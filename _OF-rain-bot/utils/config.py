from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal

from utils.env import load_project_env

load_project_env()


def _get_str(name: str, default: str) -> str:
    return str(os.getenv(name, default))


def _get_int(name: str, default: int) -> int:
    return int(str(os.getenv(name, str(default))))


def _get_float(name: str, default: float) -> float:
    return float(str(os.getenv(name, str(default))))


def _get_decimal(name: str, default: str | Decimal) -> Decimal:
    if isinstance(default, Decimal):
        default_value = str(default)
    else:
        default_value = default
    return Decimal(str(os.getenv(name, default_value)))


def _get_csv_tuple(name: str, default: str, *, upper: bool = False) -> tuple[str, ...]:
    values = []
    for item in str(os.getenv(name, default)).split(","):
        normalized = item.strip()
        if not normalized:
            continue
        values.append(normalized.upper() if upper else normalized)
    return tuple(values)


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    name: str
    user: str
    password: str
    connect_timeout_seconds: float


@dataclass(frozen=True)
class BybitConfig:
    api_key: str
    api_secret: str
    market_data_api_endpoint: str
    market_data_ws_endpoint: str
    trading_api_endpoint: str
    futures_market_api_endpoint: str
    private_ws_endpoint: str
    recv_window: str


@dataclass(frozen=True)
class TelegramConfig:
    token: str
    chat_id: str
    app_timezone: str


@dataclass(frozen=True)
class OrderFlowRuntimeConfig:
    db_schema: str
    analysis_reference_exchange: str
    observation_exchanges: tuple[str, ...]
    symbols: tuple[str, ...]
    book_depth: int
    book_history_size: int
    tape_window_ms: int
    wall_lookback_ms: int
    max_wall_distance_bps: Decimal
    max_wall_distance_ticks: int
    min_wall_notional_usdt: Decimal
    min_wall_relative_size: Decimal
    min_wall_persist_ms: int
    max_wall_size_drop_pct: Decimal
    min_wall_score: Decimal
    max_pull_events: int
    max_chase_ticks: int
    max_spoof_score: Decimal
    min_test_count: int
    test_touch_ticks: int
    test_debounce_ms: int
    min_rejection_ticks: int
    min_defended_ratio: Decimal
    min_tape_pressure_ratio: Decimal
    cross_confirmation_bps: Decimal
    min_cross_exchange_confirmations: int
    book_stale_ms: int
    stop_loss_size: Decimal
    max_spread_ticks: int
    max_basis_bps: Decimal
    entry_offset_ticks: int
    invalidation_offset_ticks: int
    stop_offset_ticks: int
    take_profit_r_multiple: Decimal
    pending_wall_tolerance_ticks: int
    pending_order_max_age_seconds: int
    breakeven_arm_ticks: int
    breakeven_buffer_ticks: int
    risk_per_trade_pct: Decimal
    max_daily_loss_pct: Decimal
    max_trades_per_day: int
    max_consecutive_losses: int
    symbol_cooldown_seconds: int


@dataclass(frozen=True)
class AppConfig:
    database: DatabaseConfig
    bybit: BybitConfig
    telegram: TelegramConfig
    orderflow: OrderFlowRuntimeConfig


SYMBOLS_ROUNDING = {
    "AAVEUSDT": 2,
    "ADAUSDT": 4,
    "APTUSDT": 4,
    "AVAXUSDT": 4,
    "BNBUSDT": 2,
    "BTCUSDT": 2,
    "DOGEUSDT": 5,
    "DOTUSDT": 4,
    "ENAUSDT": 4,
    "ETHUSDT": 2,
    "JUPUSDT": 4,
    "XRPUSDT": 5,
    "SOLUSDT": 3,
    "SUIUSDT": 5,
}

POSITION_ROUNDING_RULES = {
    "BTCUSDT": lambda x: round(Decimal(str(x)), 3),
    "ETHUSDT": lambda x: round(Decimal(str(x)), 2),
    "SOLUSDT": lambda x: round(Decimal(str(x)), 1),
    "DOGEUSDT": lambda x: int(round(Decimal(str(x)), 0)),
    "XRPUSDT": lambda x: int(round(Decimal(str(x)), 0)),
    "SUIUSDT": lambda x: int(round(Decimal(str(x)), -1)),
}

ORDERFLOW_TICK_SIZES = {
    symbol: 10 ** (-precision)
    for symbol, precision in SYMBOLS_ROUNDING.items()
}


def build_app_config() -> AppConfig:
    trading_api_endpoint = _get_str("BYBIT_TRADING_API_ENDPOINT", "https://api-demo.bybit.com")
    return AppConfig(
        database=DatabaseConfig(
            host=_get_str("DB_HOST", "localhost"),
            port=_get_int("DB_PORT", 5432),
            name=_get_str("DATABASE", "pompilo_db"),
            user=_get_str("DB_USER", "admin"),
            password=os.getenv("DB_PASS", os.getenv("DB_PASSWORD", "admin_pass")),
            connect_timeout_seconds=_get_float("DB_CONNECT_TIMEOUT", 5.0),
        ),
        bybit=BybitConfig(
            api_key=_get_str("BYBIT_API_KEY", ""),
            api_secret=_get_str("BYBIT_API_SECRET", ""),
            market_data_api_endpoint=_get_str("BYBIT_MARKET_DATA_API_ENDPOINT", "https://api.bybit.com"),
            market_data_ws_endpoint=_get_str("BYBIT_MARKET_DATA_WS_ENDPOINT", "wss://stream.bybit.com/v5/public/spot"),
            trading_api_endpoint=trading_api_endpoint,
            futures_market_api_endpoint=_get_str("BYBIT_FUTURES_MARKET_API_ENDPOINT", trading_api_endpoint),
            private_ws_endpoint=_get_str("BYBIT_PRIVATE_WS_ENDPOINT", "wss://stream.bybit.com/v5/private?max_active_time=10m"),
            recv_window=_get_str("BYBIT_RECV_WINDOW", "10000"),
        ),
        telegram=TelegramConfig(
            token=_get_str("TOKEN", ""),
            chat_id=_get_str("CHAT_ID", ""),
            app_timezone=_get_str("APP_TIMEZONE", "Europe/Kyiv"),
        ),
        orderflow=OrderFlowRuntimeConfig(
            db_schema="orderflow_scalp",
            analysis_reference_exchange=_get_str("ORDERFLOW_ANALYSIS_REFERENCE_EXCHANGE", "bybit"),
            observation_exchanges=_get_csv_tuple("ORDERFLOW_OBSERVATION_EXCHANGES", "bybit,binance,okx"),
            symbols=_get_csv_tuple("ORDERFLOW_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,DOGEUSDT,XRPUSDT,SUIUSDT", upper=True),
            book_depth=_get_int("ORDERFLOW_BOOK_DEPTH", 20),
            book_history_size=_get_int("ORDERFLOW_BOOK_HISTORY_SIZE", 32),
            tape_window_ms=_get_int("ORDERFLOW_TAPE_WINDOW_MS", 3000),
            wall_lookback_ms=_get_int("ORDERFLOW_WALL_LOOKBACK_MS", 1500),
            max_wall_distance_bps=_get_decimal("ORDERFLOW_MAX_WALL_DISTANCE_BPS", "8"),
            max_wall_distance_ticks=_get_int("ORDERFLOW_MAX_WALL_DISTANCE_TICKS", 8),
            min_wall_notional_usdt=_get_decimal("ORDERFLOW_MIN_WALL_NOTIONAL_USDT", "250000"),
            min_wall_relative_size=_get_decimal("ORDERFLOW_MIN_WALL_RELATIVE_SIZE", "4.0"),
            min_wall_persist_ms=_get_int("ORDERFLOW_MIN_WALL_PERSIST_MS", 800),
            max_wall_size_drop_pct=_get_decimal("ORDERFLOW_MAX_WALL_SIZE_DROP_PCT", "0.45"),
            min_wall_score=_get_decimal("ORDERFLOW_MIN_WALL_SCORE", "65"),
            max_pull_events=_get_int("ORDERFLOW_MAX_PULL_EVENTS", 3),
            max_chase_ticks=_get_int("ORDERFLOW_MAX_CHASE_TICKS", 2),
            max_spoof_score=_get_decimal("ORDERFLOW_MAX_SPOOF_SCORE", "45"),
            min_test_count=_get_int("ORDERFLOW_MIN_TEST_COUNT", 2),
            test_touch_ticks=_get_int("ORDERFLOW_TEST_TOUCH_TICKS", 2),
            test_debounce_ms=_get_int("ORDERFLOW_TEST_DEBOUNCE_MS", 500),
            min_rejection_ticks=_get_int("ORDERFLOW_MIN_REJECTION_TICKS", 1),
            min_defended_ratio=_get_decimal("ORDERFLOW_MIN_DEFENDED_RATIO", "0.55"),
            min_tape_pressure_ratio=_get_decimal("ORDERFLOW_MIN_TAPE_PRESSURE_RATIO", "1.15"),
            cross_confirmation_bps=_get_decimal("ORDERFLOW_CROSS_CONFIRMATION_BPS", "6"),
            min_cross_exchange_confirmations=_get_int("ORDERFLOW_MIN_CROSS_EXCHANGE_CONFIRMATIONS", 1),
            book_stale_ms=_get_int("ORDERFLOW_BOOK_STALE_MS", 2500),
            stop_loss_size=_get_decimal("STOP_LOSS_SIZE", "0.5"),
            max_spread_ticks=_get_int("ORDERFLOW_MAX_SPREAD_TICKS", 3),
            max_basis_bps=_get_decimal("ORDERFLOW_MAX_BASIS_BPS", "50"),
            entry_offset_ticks=_get_int("ORDERFLOW_ENTRY_OFFSET_TICKS", 1),
            invalidation_offset_ticks=_get_int("ORDERFLOW_INVALIDATION_OFFSET_TICKS", 1),
            stop_offset_ticks=_get_int("ORDERFLOW_STOP_OFFSET_TICKS", 1),
            take_profit_r_multiple=_get_decimal("ORDERFLOW_TAKE_PROFIT_R_MULTIPLE", "1.5"),
            pending_wall_tolerance_ticks=_get_int("ORDERFLOW_PENDING_WALL_TOLERANCE_TICKS", 2),
            pending_order_max_age_seconds=_get_int("ORDERFLOW_PENDING_ORDER_MAX_AGE_SECONDS", 30),
            breakeven_arm_ticks=_get_int("ORDERFLOW_BREAKEVEN_ARM_TICKS", 3),
            breakeven_buffer_ticks=_get_int("ORDERFLOW_BREAKEVEN_BUFFER_TICKS", 1),
            risk_per_trade_pct=_get_decimal("ORDERFLOW_RISK_PER_TRADE_PCT", "0.15"),
            max_daily_loss_pct=_get_decimal("ORDERFLOW_MAX_DAILY_LOSS_PCT", "1.5"),
            max_trades_per_day=_get_int("ORDERFLOW_MAX_TRADES_PER_DAY", 30),
            max_consecutive_losses=_get_int("ORDERFLOW_MAX_CONSECUTIVE_LOSSES", 5),
            symbol_cooldown_seconds=_get_int("ORDERFLOW_SYMBOL_COOLDOWN_SECONDS", 120),
        ),
    )


APP_CONFIG = build_app_config()

DB_HOST = APP_CONFIG.database.host
DB_PORT = APP_CONFIG.database.port
DB_NAME = APP_CONFIG.database.name
DB_USER = APP_CONFIG.database.user
DB_PASS = APP_CONFIG.database.password
DB_CONNECT_TIMEOUT = APP_CONFIG.database.connect_timeout_seconds

BYBIT_API_KEY = APP_CONFIG.bybit.api_key
BYBIT_API_SECRET = APP_CONFIG.bybit.api_secret
BYBIT_MARKET_DATA_API_ENDPOINT = APP_CONFIG.bybit.market_data_api_endpoint
BYBIT_MARKET_DATA_WS_ENDPOINT = APP_CONFIG.bybit.market_data_ws_endpoint
BYBIT_TRADING_API_ENDPOINT = APP_CONFIG.bybit.trading_api_endpoint
BYBIT_FUTURES_MARKET_API_ENDPOINT = APP_CONFIG.bybit.futures_market_api_endpoint
BYBIT_PRIVATE_WS_ENDPOINT = APP_CONFIG.bybit.private_ws_endpoint
BYBIT_RECV_WINDOW = APP_CONFIG.bybit.recv_window

TELEGRAM_TOKEN = APP_CONFIG.telegram.token
TELEGRAM_CHAT_ID = APP_CONFIG.telegram.chat_id
APP_TIMEZONE = APP_CONFIG.telegram.app_timezone

ORDERFLOW_ANALYSIS_REFERENCE_EXCHANGE = APP_CONFIG.orderflow.analysis_reference_exchange
ORDERFLOW_OBSERVATION_EXCHANGES = APP_CONFIG.orderflow.observation_exchanges
ORDERFLOW_SYMBOLS = APP_CONFIG.orderflow.symbols
ORDERFLOW_BOOK_DEPTH = APP_CONFIG.orderflow.book_depth
ORDERFLOW_BOOK_HISTORY_SIZE = APP_CONFIG.orderflow.book_history_size
ORDERFLOW_TAPE_WINDOW_MS = APP_CONFIG.orderflow.tape_window_ms
ORDERFLOW_WALL_LOOKBACK_MS = APP_CONFIG.orderflow.wall_lookback_ms
ORDERFLOW_MAX_WALL_DISTANCE_BPS = float(APP_CONFIG.orderflow.max_wall_distance_bps)
ORDERFLOW_MAX_WALL_DISTANCE_TICKS = APP_CONFIG.orderflow.max_wall_distance_ticks
ORDERFLOW_MIN_WALL_NOTIONAL_USDT = float(APP_CONFIG.orderflow.min_wall_notional_usdt)
ORDERFLOW_MIN_WALL_RELATIVE_SIZE = float(APP_CONFIG.orderflow.min_wall_relative_size)
ORDERFLOW_MIN_WALL_PERSIST_MS = APP_CONFIG.orderflow.min_wall_persist_ms
ORDERFLOW_MAX_WALL_SIZE_DROP_PCT = float(APP_CONFIG.orderflow.max_wall_size_drop_pct)
ORDERFLOW_MIN_WALL_SCORE = float(APP_CONFIG.orderflow.min_wall_score)
ORDERFLOW_MAX_PULL_EVENTS = APP_CONFIG.orderflow.max_pull_events
ORDERFLOW_MAX_CHASE_TICKS = APP_CONFIG.orderflow.max_chase_ticks
ORDERFLOW_MAX_SPOOF_SCORE = float(APP_CONFIG.orderflow.max_spoof_score)
ORDERFLOW_MIN_TEST_COUNT = APP_CONFIG.orderflow.min_test_count
ORDERFLOW_TEST_TOUCH_TICKS = APP_CONFIG.orderflow.test_touch_ticks
ORDERFLOW_TEST_DEBOUNCE_MS = APP_CONFIG.orderflow.test_debounce_ms
ORDERFLOW_MIN_REJECTION_TICKS = APP_CONFIG.orderflow.min_rejection_ticks
ORDERFLOW_MIN_DEFENDED_RATIO = float(APP_CONFIG.orderflow.min_defended_ratio)
ORDERFLOW_MIN_TAPE_PRESSURE_RATIO = float(APP_CONFIG.orderflow.min_tape_pressure_ratio)
ORDERFLOW_CROSS_CONFIRMATION_BPS = float(APP_CONFIG.orderflow.cross_confirmation_bps)
ORDERFLOW_MIN_CROSS_EXCHANGE_CONFIRMATIONS = APP_CONFIG.orderflow.min_cross_exchange_confirmations
ORDERFLOW_BOOK_STALE_MS = APP_CONFIG.orderflow.book_stale_ms
ORDERFLOW_MAX_SPREAD_TICKS = APP_CONFIG.orderflow.max_spread_ticks
ORDERFLOW_MAX_BASIS_BPS = float(APP_CONFIG.orderflow.max_basis_bps)
ORDERFLOW_ENTRY_OFFSET_TICKS = APP_CONFIG.orderflow.entry_offset_ticks
ORDERFLOW_INVALIDATION_OFFSET_TICKS = APP_CONFIG.orderflow.invalidation_offset_ticks
ORDERFLOW_STOP_OFFSET_TICKS = APP_CONFIG.orderflow.stop_offset_ticks
ORDERFLOW_TAKE_PROFIT_R_MULTIPLE = float(APP_CONFIG.orderflow.take_profit_r_multiple)
ORDERFLOW_PENDING_WALL_TOLERANCE_TICKS = APP_CONFIG.orderflow.pending_wall_tolerance_ticks
ORDERFLOW_PENDING_ORDER_MAX_AGE_SECONDS = APP_CONFIG.orderflow.pending_order_max_age_seconds
ORDERFLOW_BREAKEVEN_ARM_TICKS = APP_CONFIG.orderflow.breakeven_arm_ticks
ORDERFLOW_BREAKEVEN_BUFFER_TICKS = APP_CONFIG.orderflow.breakeven_buffer_ticks
ORDERFLOW_RISK_PER_TRADE_PCT = float(APP_CONFIG.orderflow.risk_per_trade_pct)
ORDERFLOW_MAX_DAILY_LOSS_PCT = float(APP_CONFIG.orderflow.max_daily_loss_pct)
ORDERFLOW_MAX_TRADES_PER_DAY = APP_CONFIG.orderflow.max_trades_per_day
ORDERFLOW_MAX_CONSECUTIVE_LOSSES = APP_CONFIG.orderflow.max_consecutive_losses
ORDERFLOW_SYMBOL_COOLDOWN_SECONDS = APP_CONFIG.orderflow.symbol_cooldown_seconds
