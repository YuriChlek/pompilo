# Stage 0 Baseline

## Purpose

This document records the pre-service candle ownership baseline before the standalone Market Data Service starts owning canonical candle ingestion.

Stage 0 does not change runtime behavior in existing applications. It documents the initial symbol/timeframe universe, rollback boundary, and smoke scenarios that prove the new service can be introduced without requiring any existing reader switch.

## Current Candle Ownership Boundary

Existing applications currently fetch and read candles through their own paths and legacy `_candles_trading_data` tables. These paths are treated as external baseline behavior for future backfill comparison only.

Market Data Service Stage 0 does not:

- change existing application schedulers;
- change existing application candle readers;
- require existing applications to read `_market_data`;
- require a schema rollback to disable existing application behavior.

## Initial Symbol Universe

The initial provider-symbol universe is:

- `BTCUSDT`
- `ETHUSDT`
- `LTCUSDT`
- `SOLUSDT`
- `SUIUSDT`
- `TAOUSDT`
- `XRPUSDT`

The canonical-symbol mapping is owned by `market_symbols` and `provider_symbols` from Stage 2 onward.

## Supported Timeframes

The initial supported timeframes are:

- `1h`
- `4h`
- `1d`

Each timeframe must have explicit close-boundary, safety-delay, sync, validation, snapshot, and event behavior before production rollout.

## Rollback Rule

Rollback for stages 0-11 means stopping Market Data Service workers and leaving existing application runtimes untouched. Because no existing application is switched to `_market_data` in this plan, rollback does not require changing existing application readers.

Schema rollback is only needed if the Market Data Service schema itself must be removed. It is not needed to restore current application behavior.

## Smoke Scenarios

### `1h`

- Sync one configured provider symbol through Market Data Service.
- Verify closed `1h` candles are normalized as `Decimal` values.
- Verify repeated insert is duplicate-safe.

### `4h`

- Sync one configured provider symbol through Market Data Service.
- Verify closed `4h` candles use UTC `00`, `04`, `08`, `12`, `16`, `20` boundaries.
- Verify missing middle intervals are detected as gaps.

### `1d`

- Sync one configured provider symbol through Market Data Service.
- Verify closed `1d` candles use UTC day boundaries.
- Verify missing latest interval is detected as incomplete.

## Stage 0 Non-Goals

- No existing application runtime switch.
- No existing application reader changes.
- No downstream readiness dispatcher.
- No trading signal generation.
- No order execution.

## Readiness Checklist

- Current candle ownership boundaries are documented.
- Initial symbols and supported timeframes are documented.
- New service rollout does not require existing application runtime changes.
- Rollback path does not require existing application reader changes.
