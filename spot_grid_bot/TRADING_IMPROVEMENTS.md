# Trading Strategy Improvements: Strict Exit and Capital-Lock Reduction

## Purpose

This document is a technical roadmap for improving `spot_grid_bot` while staying aligned with the current codebase conventions.

The goal is not to promise guaranteed profit. The goal is to:

- prevent planned sell orders below a configurable profitable exit floor;
- reduce the chance of trapping capital in weak underwater positions;
- make averaging decisions depend on the expected post-averaging exit feasibility;
- improve entry quality with market-structure, BTC-market, and volatility context;
- make the behavior measurable in unit tests, dry-run output, and backtests.

---

## Core Principle: Strict Planned Exit Floor

The bot must not intentionally create regular sell targets below the current minimum profitable exit floor.

Use the existing codebase convention of basis points (`bps`) and snake_case config fields. The canonical minimum floor should be:

```text
minimum_exit_price =
    cost_basis_price
    * (1 + min_net_exit_profit_bps / 10_000)
    * (1 + estimated_exit_cost_bps / 10_000)
```

Where:

- `min_net_exit_profit_bps` is the desired net profit buffer above cost basis. Default: `100` for 1.0%.
- `estimated_exit_cost_bps` is the expected exit cost buffer, derived from maker fee and simulated slippage settings or configured explicitly.

This avoids the ambiguous `CostBasis * 1.01 + EstimatedFees` formula and avoids double-counting fees. If the 1.0% target is meant to be net profit, costs must be applied separately and explicitly.

This floor is a planning invariant, not a promise of guaranteed realized profit. Real execution can still differ due to partial fills, fee tier changes, rounding, venue constraints, outages, or market gaps.

---

## Phase 1: Configuration and Data Model Foundation

### Configuration Updates

Add fields using the existing dataclass style in `domain/strategy_config.py`.

Suggested fields:

- `execution.min_net_exit_profit_bps: float = 100.0`
  - Desired net profit buffer above cost basis.
- `execution.estimated_exit_cost_bps: float | None = None`
  - Optional explicit exit-cost buffer. If unset, derive from `maker_fee_bps + simulated_slippage_bps`.
- `grid.profit_space_buffer_bps: float = 50.0`
  - Extra entry-to-resistance room required beyond the exit floor.
- `grid.max_underwater_hours_compression: int = 168`
  - Time underwater before compressing regular sell grids into one focused exit target. Use hours because runtime timestamps are stored as datetimes and scheduler cadence is hourly.
- `grid.exit_compression_enabled: bool = True`
- `risk.btc_market_filter_enabled: bool = True`
- `risk.relative_strength_filter_enabled: bool = True`
- `risk.min_relative_strength_vs_btc: float = 0.0`
- `risk.recovery_quota_usage_limit_pct: float = 1.0`
  - Caps repeated recovery allocation after a symbol has consumed its tracked recovery quota.

Avoid all-caps config names such as `MIN_EXIT_MARGIN_PCT`; those do not match the current project style.

### Domain Model Extensions

Add fields where they are consumed, not only where they are convenient.

Suggested additions:

- `PreliminarySymbolAnalysis.relative_strength_vs_btc: float | None`
- `SymbolRuntimeState.position_opened_at: datetime | None`
- `SymbolRuntimeState.underwater_since: datetime | None`
- `SymbolRuntimeState.recovery_quota_used_quote: float`
- `SymbolRuntimeState.last_recovery_order_at: datetime | None`

`position_opened_at` alone is not enough for exit compression. The bot also needs `underwater_since` because a position can be open for a long time while only recently becoming underwater.

### Persistence Migration

Update `PostgresStateStore` and migration logic to persist the new runtime fields. On restore:

- clear `position_opened_at`, `underwater_since`, and recovery usage when base balance is zero;
- initialize `position_opened_at` when base balance becomes positive and no timestamp exists;
- set `underwater_since` when mark price first falls below cost basis;
- clear `underwater_since` when mark price recovers to or above cost basis;
- preserve recovery quota usage across restarts.

---

## Phase 2: Strict Exit Enforcement

### Cost Basis Resolver

Update `domain/cost_basis.py` so `minimum_exit_price()` uses explicit net-profit and cost buffers.

Required behavior:

- return `None` when cost basis is unavailable for active inventory;
- never return a value below `cost_basis_price * (1 + min_net_exit_profit_bps / 10_000)`;
- include configured or derived exit costs without double-counting;
- keep all sell planning and execution guardrails using this same function.

### Target Order Builder

Keep the existing layered sell behavior:

1. build regime-specific sell targets;
2. apply adaptive uptrend take-profit logic;
3. rebase target prices above `minimum_take_profit_price`;
4. apply RSI sell filter for regular sell grids;
5. apply no-loss floor before adding target orders.

The RSI sell filter should apply to regular range/uptrend sell grids, but not to protective de-risk exits or exit-compression orders. Protective exits exist to reduce capital lock and should remain governed by the strict no-loss floor instead.

### Dynamic Exit Compression

If a symbol has been underwater longer than `grid.max_underwater_hours_compression`, replace the tiered regular sell grid with a single focused exit target:

```text
exit_compression_price = minimum_exit_price(...)
```

Technical rules:

- only apply when `exit_compression_enabled` is true;
- only apply when base balance is positive and cost basis is known;
- bypass regular `RSI > sell_rsi_threshold` gating;
- never bypass the no-loss floor;
- use reduce-only sell intent;
- size should be venue-normalized and may be partial if venue constraints require it;
- log a distinct reason such as `exit_compression_underwater_timeout`.

---

## Phase 3: Entry and Averaging Quality

### Profit-Space Entry Filter

Before allowing buy levels, estimate whether there is enough room between the proposed entry and the nearest resistance to exit profitably.

Use a deterministic resistance source in this order:

1. nearest swing high from `StructureSnapshot`;
2. current range high from the range grid;
3. higher-timeframe recent swing high when available;
4. fallback: no resistance check if no reliable resistance exists.

Reject or reduce buy levels when:

```text
nearest_resistance_price - buy_price
    < buy_price * ((min_net_exit_profit_bps + estimated_exit_cost_bps + profit_space_buffer_bps) / 10_000)
```

This filter should run after grid geometry is built and before final size allocation.

### Averaging Depth Adjuster

Before creating underwater averaging buy orders, simulate the post-fill cost basis for each candidate buy level.

For each candidate level:

```text
new_cost_basis =
    (current_base_qty * current_cost_basis + buy_qty * effective_buy_price)
    / (current_base_qty + buy_qty)
```

Then compute the required exit price from `new_cost_basis`.

Allow the averaging order only if at least one of these is true:

- required exit price is inside current ATR-reachable bounds;
- required exit price is below or near the nearest resistance plus acceptable buffer;
- higher-timeframe regime is not bearish and symbol relative strength is acceptable.

If the simulated exit is not realistic, skip the averaging level and wait for a deeper structural level. Log a distinct reason such as `averaging_exit_unreachable`.

---

## Phase 4: Portfolio Protection

### BTC Market Filter

The BTC filter must be portfolio-level, not hidden inside a single-symbol grid builder.

Implementation path:

1. Load BTCUSDT candles for the same base and higher timeframe used by symbols.
2. Compute BTC regime and BTC momentum.
3. Add a portfolio-wide market context passed into analysis or allocation.
4. When BTC is in confirmed downtrend or risk-off:
   - block fresh entries;
   - allow protective exits;
   - allow recovery only when explicitly configured and when post-averaging exit feasibility passes.

### Relative Strength vs BTC

Compute relative strength on aligned candle windows:

```text
relative_strength_vs_btc =
    symbol_return_over_window - btc_return_over_window
```

Use this field in `PortfolioAllocator`:

- apply a budget penalty when relative strength is below `risk.min_relative_strength_vs_btc`;
- redirect fresh-entry capital toward symbols with stronger relative strength;
- avoid adding recovery budget to symbols that are both underwater and materially weaker than BTC unless a high-confidence structural reversal exists.

### Underwater Quota Lock

The current allocator is mostly stateless per cycle. To enforce "already used recovery quota", persist usage in runtime state.

Rules:

- increment `recovery_quota_used_quote` when recovery orders are confirmed or when a recovery order is placed, depending on the chosen accounting policy;
- reset or decay the quota only after a full exit or after a configured cooldown;
- block further recovery allocation when usage exceeds `risk.recovery_quota_usage_limit_pct` of the initial planned recovery quota;
- allow override only when structure confidence and relative strength both meet configured thresholds.

---

## Phase 5: Validation

### Unit Tests

Add tests for:

- `minimum_exit_price()` with explicit net-profit and cost buffers;
- no-loss blocking when cost basis is missing;
- regular sell RSI gating versus de-risk and exit-compression bypass;
- exit compression after `underwater_since` exceeds threshold;
- clearing runtime underwater fields when base balance returns to zero;
- post-averaging cost-basis simulation;
- profit-space entry filtering against swing high / range high / higher-timeframe swing high;
- BTC market filter blocking fresh entries;
- relative-strength budget penalty;
- recovery quota usage lock.

### Backtesting Scenarios

Do not use "0.0% loss rate" as the success criterion. That ignores unrealized loss and capital lock.

Track:

- realized PnL;
- unrealized PnL;
- maximum drawdown;
- underwater duration;
- capital locked in underwater positions;
- number of blocked no-loss sells;
- number of skipped averaging orders due to unreachable exit;
- recovery quota consumed;
- exit-compression events;
- fresh entries blocked by BTC filter.

Required scenarios:

- flash crash;
- slow bleed;
- prolonged BTC downtrend;
- sideways chop;
- weak structure with compressed volatility;
- sharp recovery after underwater averaging.

### Dry-Run Verification

`dry-run` output should expose reasons clearly enough for operators:

- `planner_no_loss_sell_blocked`;
- `buy_entries_blocked_due_to_rsi`;
- `sell_entries_blocked_due_to_rsi`;
- `higher_timeframe_downtrend`;
- `btc_market_filter_blocked`;
- `profit_space_insufficient`;
- `averaging_exit_unreachable`;
- `exit_compression_underwater_timeout`;
- `recovery_quota_locked`.
