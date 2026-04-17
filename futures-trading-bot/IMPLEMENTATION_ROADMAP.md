# V2 Entry Strategy Roadmap

## Goal

Implement `v2 entry logic` for the current `trend_breakout` bot so the bot gets more valid entries without becoming overfiltered and inactive.

The main design rule for this phase:
- improve entry quality
- increase practical entry coverage
- avoid adding too many hard `skip` conditions

## Current Baseline

Current live/backtest entry path:
- one active setup: `breakout_close`
- breakout over `lookback_candles`
- hard range filter
- hard volume-spike filter
- ATR-based stop
- risk-based sizing
- portfolio admission controls already implemented
- live exit lifecycle already includes breakeven, trailing, and optional regime exit

Current main limitations:
- only one entry style
- breakout often happens after expansion, so late entries are common
- H1 range state can be too restrictive inside otherwise valid H4 trend continuation
- volume filter is binary instead of context-aware
- breakout buffer is too simple

## Scope

This roadmap covers only the `entry side` of the strategy.

In scope:
- keep `breakout_close`
- add `breakout_reclaim`
- make volume threshold adaptive to trend strength
- soften H1 range rejection inside strong H4 trend
- allow reduced-risk high-vol entries instead of full skip
- make breakout confirmation buffer partly ATR-based
- preserve live/backtest parity

Out of scope:
- new portfolio rules
- new trailing logic
- Backtrader integration
- ML ranking
- funding/open-interest redesign

## Target Outcome

After this phase the bot should:
- take more trend continuation entries
- miss fewer clean moves after initial breakout
- still avoid obvious range noise
- remain simple enough to debug
- stay aligned between live and backtest engines

## Implementation Phases

## Phase 1: Extend Strategy Config

### Objective

Add all parameters required for `v2 entry logic` without changing behavior yet.

### Files

- `trading/domain/strategy_config.py`

### Changes

Extend `BreakoutTrendStrategyConfig` with:
- `strong_trend_volume_ratio`
- `medium_trend_volume_ratio`
- `weak_trend_allows_entry`
- `reclaim_enabled`
- `reclaim_tolerance_atr_fraction`
- `reclaim_stop_atr_multiplier`
- `atr_breakout_buffer_fraction`
- `allow_h1_range_in_strong_h4_trend`
- `high_vol_risk_multiplier`

Optional:
- `reclaim_max_candle_atr`
- `require_reclaim_close_in_breakout_direction`

### Initial Default Values

- `strong_trend_volume_ratio = 1.05`
- `medium_trend_volume_ratio = 1.15`
- `weak_trend_allows_entry = False`
- `reclaim_enabled = True`
- `reclaim_tolerance_atr_fraction = 0.15`
- `reclaim_stop_atr_multiplier = 1.00`
- `atr_breakout_buffer_fraction = 0.10`
- `allow_h1_range_in_strong_h4_trend = True`
- `high_vol_risk_multiplier = 0.50`

### Exit Criteria

- config can express every planned `v2` rule
- no runtime behavior changes yet

## Phase 2: Refactor Entry Pipeline

### Objective

Split the current single-path breakout builder into smaller reusable parts.

### Files

- `trading/domain/signal_generation.py`

### Changes

Extract helper functions:
- `detect_market_regime()`
- `_resolve_trend_strength()`
- `_required_volume_ratio()`
- `_build_breakout_buffer()`
- `_is_h1_range_soft_allowed()`
- `_build_breakout_close_signal()`
- `_build_breakout_reclaim_signal()`
- `_resolve_entry_risk_pct()`

### Design Rules

- keep `generate_strategy_signal()` as the main entrypoint
- preserve the current `TradeSignal` payload style
- keep setup tagging via `metadata["setup_type"]`
- keep cluster tagging
- keep `risk_distance` in metadata

### Exit Criteria

- entry pipeline is modular enough to support both setups cleanly
- no duplicated long/short logic explosion

## Phase 3: Upgrade Breakout Close Logic

### Objective

Make the current `breakout_close` setup more context-aware without reducing trade count too much.

### Files

- `trading/domain/signal_generation.py`

### Changes

#### 1. Adaptive volume filter

Replace:
- one hard threshold: `min_volume_spike_ratio`

With:
- if `trend_strength == strong`: require `>= strong_trend_volume_ratio`
- if `trend_strength == medium`: require `>= medium_trend_volume_ratio`
- if `trend_strength == weak`: allow only if `weak_trend_allows_entry == True`, otherwise skip

#### 2. ATR-aware breakout buffer

Replace:
- `breakout_buffer = close * min_breakout_close_pct`

With:
- `breakout_buffer = max(close * min_breakout_close_pct, atr * atr_breakout_buffer_fraction)`

#### 3. H1 range soft-allow path

If:
- `H1 is range`
- `H4 trend is aligned`
- `trend_strength == strong`
- `allow_h1_range_in_strong_h4_trend == True`

Then:
- do not block `breakout_close`

Still block when:
- `H4 is range`
- `trend_strength` is too weak
- direction is not aligned

#### 4. High-vol reduced-risk mode

If:
- `regime.is_high_vol == True`
- breakout is otherwise valid

Then:
- do not necessarily skip
- reduce effective risk percent using `high_vol_risk_multiplier`

### Exit Criteria

- `breakout_close` still works
- it becomes more adaptive, not more restrictive

## Phase 4: Add Breakout Reclaim Setup

### Objective

Add a second continuation entry style for missed initial breakouts.

### Files

- `trading/domain/signal_generation.py`
- optionally `trading/domain/models.py` if extra setup context is needed

### Long Rules

`breakout_reclaim` long is valid when:
- `regime == bull_trend`
- `reclaim_enabled == True`
- a recent breakout level exists from the same `lookback_candles`
- current candle trades back into the breakout area:
  `low <= breakout_level + atr * reclaim_tolerance_atr_fraction`
- current candle closes back above breakout level
- current candle closes above open
- breakout is not invalidated structurally

### Short Rules

Mirror the long rules:
- `regime == bear_trend`
- `high >= breakout_level - atr * reclaim_tolerance_atr_fraction`
- close back below breakout level
- close below open

### Stop Logic

Long:
- `stop_loss = min(reclaim_candle_low, breakout_level - atr * reclaim_stop_atr_multiplier)`

Short:
- `stop_loss = max(reclaim_candle_high, breakout_level + atr * reclaim_stop_atr_multiplier)`

### Metadata

Tag reclaim entries with:
- `setup_type = "breakout_reclaim"`
- `regime`
- `breakout_level`
- `risk_distance`
- `cluster`
- optional `volume_spike_ratio`

### Exit Criteria

- bot can enter through either `breakout_close` or `breakout_reclaim`
- reclaim does not replace close-breakout, it supplements it

## Phase 5: Entry Priority and Risk Wiring

### Objective

Define deterministic setup priority and ensure sizing reflects context.

### Files

- `trading/domain/signal_generation.py`

### Rules

Signal construction order:
1. try `breakout_close`
2. if no signal, try `breakout_reclaim`

Risk rules:
- base risk = `strategy_config.risk.primary_risk_pct`
- if `regime.is_high_vol`, apply `high_vol_risk_multiplier`
- use the resulting effective risk percent in `calculate_position_size()`

### Important Constraint

Do not add a separate portfolio admission branch here.
The resulting signal should still flow through the existing portfolio controls unchanged.

### Exit Criteria

- deterministic setup precedence
- setup type is visible in signal metadata
- risk scaling works without changing execution contract

## Phase 6: Backtest Validation

### Objective

Validate whether `v2 entry` actually improves usable opportunity capture.

### Files

- existing backtest engine only
- no structural backtest redesign required

### Validation Focus

Compare:
- current baseline
- `breakout_close` upgraded only
- `breakout_close + breakout_reclaim`

Track:
- trade count
- `net_pnl_pct`
- `expectancy_by_setup`
- `pnl_by_regime`
- `max_drawdown_by_cluster`

### Acceptance Goal

The new version should ideally:
- increase valid trade count
- keep drawdown deterioration limited
- show non-trivial contribution from `breakout_reclaim`

## Phase 7: Tests

### Objective

Cover all new entry behaviors with focused tests.

### Files

- `tests/test_strategy_signal_logic.py`
- `tests/test_backtesting_module.py` if setup analytics assertions need expansion

### Required Tests

#### Breakout Close

- long breakout still builds signal
- short breakout still builds signal
- strong trend uses softer volume threshold
- medium trend still requires stronger volume
- weak trend is skipped by default
- ATR-based breakout buffer rejects micro-breakout
- H1 range is allowed when H4 trend is strong and aligned
- H4 range still blocks

#### Breakout Reclaim

- long reclaim builds signal
- short reclaim builds signal
- reclaim skipped when candle does not recover back through level
- reclaim skipped when tolerance band is not touched
- reclaim stop is built from reclaim logic, not close-breakout logic

#### High Vol

- valid high-vol breakout reduces risk instead of always skipping
- invalid dirty high-vol breakout is still skipped

#### Priority

- if both setups could theoretically qualify, `breakout_close` wins
- `setup_type` is correct in payload

### Exit Criteria

- new logic is protected by unit tests
- existing tests remain green

## Phase 8: Documentation Refresh

### Objective

Document the new entry model after implementation.

### Files

- `README.md`
- optionally `backtesting/README.md`

### Required Changes

Update strategy docs to reflect:
- two active entry setups
- adaptive volume filter
- soft H1 range handling
- high-vol reduced-risk mode

## Recommended Execution Order

1. Phase 1: Extend Strategy Config
2. Phase 2: Refactor Entry Pipeline
3. Phase 3: Upgrade Breakout Close Logic
4. Phase 4: Add Breakout Reclaim Setup
5. Phase 5: Entry Priority and Risk Wiring
6. Phase 7: Tests
7. Phase 6: Backtest Validation
8. Phase 8: Documentation Refresh

## Validation Checklist

- no broken imports
- live signal path still returns one `TradeSignal` or `None`
- `./.venv/bin/python -m unittest discover -q`
- backtest metrics remain serializable
- `setup_type` clearly distinguishes `breakout_close` and `breakout_reclaim`

## Notes

- Do not add more hard boolean filters unless they replace an existing one.
- Prefer adaptive thresholds over additional global blocks.
- Preserve live/backtest parity by keeping all new entry decisions inside `trading/domain/signal_generation.py`.
- If `breakout_reclaim` produces too many weak entries, tighten reclaim tolerance before adding new filters.
