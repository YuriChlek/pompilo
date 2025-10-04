import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import joblib

import torch
from dataclasses import dataclass
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from utils import TradeSignal


@dataclass
class TradeDecision:
    time: str
    signal: TradeSignal
    entry_price: float


# ХАРДКОДОВАНІ ОЗНАКИ ЯК У ТРЕНУВАЛЬНОМУ СКРИПТІ
KNOWN_REALS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
"""
UNKNOWN_REALS = [
    "open", "high", "low", "volume", "cvd", "poc", "close",
    "rsi", "macd", "macd_signal", "macd_histogram", "bb_upper",
    "bb_lower", "bb_position", "volume_ma", "volume_ratio",
    "price_change_1h", "price_change_6h", "volatility"
]
"""
UNKNOWN_REALS = [
    "open", "high", "low", "volume", "cvd", "poc", "close",
    "rsi", "bb_upper", "bb_lower", "bb_position",
    "volume_ma",
    "price_change_1h", "price_change_6h", "volatility"
]

def build_argparser():
    """
    Створює парсер аргументів командного рядка для конфігурації прогнозування.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="postgresql://admin:admin_pass@localhost:5432/pompilo_db",
                   help="SQLAlchemy URL, e.g. postgresql://user:pass@host:5432/db")
    p.add_argument("--table", default="_candles_trading_data.solusdt_p_candles_test_data",
                   help="Table name with 1h candles")
    p.add_argument("--model-base-dir", default="./tft_runs",
                   help="Base directory with saved models and scalers")
    p.add_argument("--symbol", default="SOLUSDT", help="Specific symbol to predict (e.g., SOLUSDT)")
    p.add_argument("--hours", type=int, default=6, help="Number of hours to predict")

    return p


def get_symbol_folder_name(symbol: str) -> str:
    """
    Перетворює символ торгової пари на назву папки.
    """
    return symbol.lower().replace('/', '').replace('-', '')


def fetch_latest_data(db_url: str, table: str, symbol: str = None, limit: int = 200) -> pd.DataFrame:
    """
    Завантажує останні дані з бази даних PostgreSQL.
    """
    eng = create_engine(db_url)

    if symbol:
        query = f"""
            SELECT open_time, close_time, symbol, open, close, high, low, cvd, volume, poc 
            FROM {table} 
            WHERE symbol = '{symbol}'
            ORDER BY open_time DESC 
            LIMIT {limit}
        """
    else:
        query = f"""
            SELECT open_time, close_time, symbol, open, close, high, low, cvd, volume, poc 
            FROM {table} 
            ORDER BY open_time DESC 
            LIMIT {limit}
        """

    with eng.begin() as conn:
        df = pd.read_sql(text(query), conn)

    df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")

    return df.dropna(subset=["open_time", "symbol", "close"])


def add_technical_indicators_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Додає технічні індикатори для прогнозування (аналогічно навчальному скрипту).
    """
    df = df.copy()

    # Групуємо по символам для коректних розрахунків
    for symbol in df['symbol'].unique():
        symbol_mask = df['symbol'] == symbol
        symbol_data = df[symbol_mask].copy()

        # RSI
        delta = symbol_data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        symbol_data['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        #exp12 = symbol_data['close'].ewm(span=12, adjust=False).mean()
        #exp26 = symbol_data['close'].ewm(span=26, adjust=False).mean()
        #symbol_data['macd'] = exp12 - exp26
        #symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9, adjust=False).mean()
        #symbol_data['macd_histogram'] = symbol_data['macd'] - symbol_data['macd_signal']

        # Bollinger Bands
        rolling_mean = symbol_data['close'].rolling(window=20, min_periods=1).mean()
        rolling_std = symbol_data['close'].rolling(window=20, min_periods=1).std()
        symbol_data['bb_upper'] = rolling_mean + (rolling_std * 2)
        symbol_data['bb_lower'] = rolling_mean - (rolling_std * 2)
        symbol_data['bb_position'] = (symbol_data['close'] - symbol_data['bb_lower']) / (
                symbol_data['bb_upper'] - symbol_data['bb_lower'])

        # Volume-based features
        symbol_data['volume_ma'] = symbol_data['volume'].rolling(window=20, min_periods=1).mean()
        #symbol_data['volume_ratio'] = symbol_data['volume'] / symbol_data['volume_ma']

        # Price momentum
        symbol_data['price_change_1h'] = symbol_data['close'].pct_change(1)
        symbol_data['price_change_6h'] = symbol_data['close'].pct_change(6)
        symbol_data['volatility'] = symbol_data['price_change_1h'].rolling(window=24, min_periods=1).std()

        # Оновлюємо дані в основному DataFrame
        """
        for col in ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_upper', 'bb_lower',
                    'bb_position', 'volume_ma', 'volume_ratio', 'price_change_1h',
                    'price_change_6h', 'volatility']:
            df.loc[symbol_mask, col] = symbol_data[col].values
        """
        for col in ['rsi', 'bb_upper', 'bb_lower',
                    'bb_position', 'volume_ma', 'price_change_1h',
                    'price_change_6h', 'volatility']:
            df.loc[symbol_mask, col] = symbol_data[col].values

    # Заповнюємо NaN значення, які могли виникнути
    """
    technical_cols = ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_upper', 'bb_lower',
                      'bb_position', 'volume_ma', 'volume_ratio', 'price_change_1h',
                      'price_change_6h', 'volatility']
    """
    technical_cols = ['rsi', 'bb_upper', 'bb_lower',
                      'bb_position', 'volume_ma', 'price_change_1h',
                      'price_change_6h', 'volatility']
    for col in technical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    print(f"Додано технічних індикаторів: {len(technical_cols)}")
    print(f"Доступні колонки після додавання індикаторів: {list(df.columns)}")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Додає часові ознаки до DataFrame для прогнозування.
    """
    df = df.copy()

    # Створюємо time_idx для кожної групи символів
    df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    df["time_idx"] = df.groupby("symbol").cumcount()

    df["hour"] = df["open_time"].dt.hour.astype(np.int16)
    df["dow"] = df["open_time"].dt.dayofweek.astype(np.int16)
    df["month"] = df["open_time"].dt.month.astype(np.int16)
    df["week_of_year"] = df["open_time"].dt.isocalendar().week.astype(np.int16)

    # Циклічні кодування
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def scale_data(df: pd.DataFrame, scalers: dict, features_to_scale: list) -> pd.DataFrame:
    """
    Масштабує числові ознаки в DataFrame за допомогою наданих скейлерів.
    """
    df_scaled = df.copy()
    for col in features_to_scale:
        if col in df_scaled.columns and col in scalers:
            # Замінюємо inf та NaN значення перед масштабуванням
            data_to_scale = df_scaled[[col]].replace([np.inf, -np.inf], np.nan).fillna(0)
            df_scaled[col] = scalers[col].transform(data_to_scale)
    return df_scaled


def create_prediction_dataset(df_scaled: pd.DataFrame, encoder_len: int, pred_len: int, meta: dict,
                              used_unknown_reals: list):
    """
    Створює датасет для прогнозування за допомогою TimeSeriesDataSet.
    """
    target = meta.get('target', 'relative_change')

    # ВИКОРИСТОВУЄМО ХАРДКОДОВАНІ ОЗНАКИ
    known_reals = KNOWN_REALS
    unknown_reals = used_unknown_reals  # Використовуємо той самий набір, що і при створенні моделі

    # Фільтруємо тільки ті ознаки, які є в даних
    available_known_reals = [col for col in known_reals if col in df_scaled.columns]
    available_unknown_reals = [col for col in unknown_reals if col in df_scaled.columns]

    print(f"Доступні ознаки для прогнозу:")
    print(f"Відомі: {available_known_reals} ({len(available_known_reals)})")
    print(f"Невідомі: {available_unknown_reals} ({len(available_unknown_reals)})")
    print(f"Загальна кількість: {len(available_known_reals) + len(available_unknown_reals)}")

    # Перевіряємо, чи достатньо даних
    min_required_length = encoder_len + pred_len
    if len(df_scaled) < min_required_length:
        print(f"Попередження: недостатньо даних ({len(df_scaled)}), потрібно мінімум {min_required_length}")
        return None

    try:
        # Для прогнозу заповнюємо цільові змінні нулями (тимчасово)
        df_for_prediction = df_scaled.copy()
        df_for_prediction[target] = 0  # Заповнюємо нулями для прогнозу

        dataset = TimeSeriesDataSet(
            df_for_prediction,
            time_idx="time_idx",
            target=target,
            group_ids=["symbol"],
            min_encoder_length=encoder_len,
            max_encoder_length=encoder_len,
            min_prediction_length=pred_len,
            max_prediction_length=pred_len,
            time_varying_known_reals=available_known_reals,
            time_varying_unknown_reals=available_unknown_reals,
            target_normalizer=None,
            add_relative_time_idx=True,
            add_encoder_length=True,
            allow_missing_timesteps=False,
        )
        return dataset
    except Exception as e:
        print(f"Помилка створення датасету: {e}")
        return None


def load_trained_model(model_dir: Path, meta: dict):
    """
    Завантажує навчену модель TemporalFusionTransformer.
    """
    model_path = model_dir / "trading_tft_model.pth"

    if not model_path.exists():
        print(f"Помилка: Файл моделі не знайдений: {model_path}")
        return None, None, None

    # Завантажуємо state_dict для аналізу
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=device)

    # Аналізуємо state_dict для визначення фактичної кількості ознак
    encoder_mask_size = None
    for key in state_dict.keys():
        if "encoder_variable_selection.flattened_grn.resample_norm.mask" in key:
            encoder_mask_size = state_dict[key].shape[0]
            break

    if encoder_mask_size is None:
        print("Попередження: Не вдалося визначити кількість ознак з state_dict")
        return None, None, None

    print(f"Виявлено кількість ознак у навченій моделі: {encoder_mask_size}")

    # ВИКОРИСТОВУЄМО ХАРДКОДОВАНІ ОЗНАКИ
    known_reals = KNOWN_REALS

    # Визначаємо кількість невідомих ознак, які були використані при навчанні
    num_unknown_features = encoder_mask_size - len(known_reals)

    # Беремо тільки потрібну кількість ознак з початку списку
    used_unknown_reals = UNKNOWN_REALS[:num_unknown_features]

    print(f"Очікувана кількість ознак: {len(known_reals) + len(used_unknown_reals)}")
    print(f"Фактична кількість ознак у моделі: {encoder_mask_size}")

    print(f"Фінальний набір ознак:")
    print(f"Відомі: {known_reals} ({len(known_reals)})")
    print(f"Невідомі: {used_unknown_reals} ({len(used_unknown_reals)})")
    print(f"Загальна кількість: {len(known_reals) + len(used_unknown_reals)}")

    # Створюємо модель з правильними параметрами
    quantiles = meta['quantiles']
    loss = QuantileLoss(quantiles=quantiles)

    # Створюємо dummy dataset для ініціалізації моделі
    sequence_length = meta['encoder_len'] + meta['pred_len']

    # Створюємо dummy дані з усіма необхідними ознаками
    dummy_data_dict = {
        'time_idx': range(sequence_length),
        'symbol': [meta['symbols'][0]] * sequence_length,
        'relative_change': np.random.randn(sequence_length),
    }

    # Додаємо всі необхідні ознаки
    for col in known_reals + used_unknown_reals:
        dummy_data_dict[col] = np.random.randn(sequence_length)

    dummy_data = pd.DataFrame(dummy_data_dict)

    try:
        dummy_dataset = TimeSeriesDataSet(
            dummy_data,
            time_idx="time_idx",
            target="relative_change",
            group_ids=["symbol"],
            min_encoder_length=meta['encoder_len'],
            max_encoder_length=meta['encoder_len'],
            min_prediction_length=meta['pred_len'],
            max_prediction_length=meta['pred_len'],
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=used_unknown_reals,
            target_normalizer=None,
            add_relative_time_idx=True,
            add_encoder_length=True,
            allow_missing_timesteps=False,
        )

        # Отримуємо конфігурацію моделі з метаданих
        model_config = meta.get('model_config', {
            "hidden_size": 128,
            "attention_head_size": 8,
            "dropout": 0.1,
            "hidden_continuous_size": 32,
        })

        model = TemporalFusionTransformer.from_dataset(
            dummy_dataset,
            loss=loss,
            **model_config,
            output_size=len(quantiles),
        )

        # Перевіряємо, чи архітектура співпадає
        current_encoder_mask_size = model.encoder_variable_selection.flattened_grn.resample_norm.mask.shape[0]
        print(f"Перевірка архітектури: поточна модель={current_encoder_mask_size}, навчена модель={encoder_mask_size}")

        if current_encoder_mask_size != encoder_mask_size:
            print(f"КРИТИЧНА ПОМИЛКА: Архітектури не співпадають!")
            print(f"Поточна модель: {current_encoder_mask_size}, навчена модель: {encoder_mask_size}")
            return None, None, None

        # Завантажуємо ваги
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        print(f"Модель успішно завантажена на {device}")
        return model, device, used_unknown_reals

    except Exception as e:
        print(f"Помилка завантаження моделі: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def calculate_relative_change_for_prediction(df: pd.DataFrame, pred_len: int = 6) -> pd.DataFrame:
    """
    Додає колонку relative_change з 0 для прогнозування.
    """
    df = df.copy()
    df["relative_change"] = 0  # Заповнюємо нулями для прогнозу
    return df


def model_confidence(q10, q50, q90, current_price):
    """
    Обчислює метрики впевненості прогнозу моделі.
    """
    q10 = np.array(q10)
    q50 = np.array(q50)
    q90 = np.array(q90)

    # 1. Mean Prediction Interval Width (MPIW) - ширина довірчого інтервалу
    mpiw = np.mean(q90 - q10)
    mpiw_relative = mpiw / current_price * 100  # У відсотках від поточної ціни

    # 2. Directional Confidence - впевненість у напрямку руху
    prob_bullish = np.mean(q50 > current_price)

    # 3. Sharpness - "різкість" прогнозу (обернена до ширини інтервалу)
    sharpness = 1 / (mpiw_relative + 1e-10)  # Уникнення ділення на 0

    # 4. Consistency - узгодженість прогнозів у часі
    if len(q50) > 1:
        consistency = np.corrcoef(q50[:-1], q50[1:])[0, 1]
    else:
        consistency = 1.0

    # 5. Комбінований confidence score (0-100)
    confidence_score = 100 * (
            0.4 * (1 - mpiw_relative / 50) +  # Ширина інтервалу (менше = краще)
            0.3 * (2 * np.abs(prob_bullish - 0.5)) +  # Впевненість у напрямку
            0.2 * np.tanh(sharpness / 10) +  # Різкість прогнозу
            0.1 * max(0, consistency)  # Узгодженість
    )

    # Обмежуємо між 0 і 100
    confidence_score = np.clip(confidence_score, 0, 100)

    return {
        "MPIW_absolute": mpiw,
        "MPIW_relative_percent": mpiw_relative,
        "prob_bullish": prob_bullish,
        "sharpness": sharpness,
        "consistency": consistency,
        "confidence_score": confidence_score
    }


def get_ai_signal():
    """
    Генерує торговий сигнал на основі прогнозів моделі.
    """
    try:
        args = build_argparser().parse_args()

        # Отримуємо шлях до папки з моделлю для конкретного символу
        symbol_folder = get_symbol_folder_name(args.symbol)
        model_dir = Path(args.model_base_dir) / symbol_folder

        if not model_dir.exists():
            print(f"Помилка: Папка з моделлю для символу {args.symbol} не знайдена: {model_dir}")
            return TradeDecision(
                time="",
                signal=TradeSignal.HOLD,
                entry_price=0
            )

        # Завантажуємо скейлери та метадані
        scalers_path = model_dir / "scalers.pkl"
        meta_path = model_dir / "dataset_meta.pkl"

        if not scalers_path.exists() or not meta_path.exists():
            print(f"Помилка: Файли моделі не знайдені в {model_dir}")
            return TradeDecision(
                time="",
                signal=TradeSignal.HOLD,
                entry_price=0
            )

        scalers = joblib.load(scalers_path)
        meta = joblib.load(meta_path)

        print(f"[1/7] Завантаження метаданих: encoder_len={meta['encoder_len']}, pred_len={meta['pred_len']}")
        print(f"Технічні індикатори: {len(meta.get('technical_indicators', []))}")
        print(f"Модель завантажена з: {model_dir}")

        # Отримуємо дані для прогнозу
        print("[2/7] Отримання останніх даних з бази...")
        required_length = meta['encoder_len'] + meta['pred_len'] + 50
        df = fetch_latest_data(args.db, args.table, args.symbol, limit=required_length)

        if df.empty:
            print("Помилка: Не вдалося отримати дані з бази")
            return TradeDecision(
                time="",
                signal=TradeSignal.HOLD,
                entry_price=0
            )

        print(f"Отримано {len(df)} рядків даних")

        # Додаємо технічні індикатори
        print("[3/7] Додавання технічних індикаторів...")
        df = add_technical_indicators_for_prediction(df)

        # Додаємо часові ознаки
        print("[4/7] Додавання часових ознак...")
        df = add_time_features(df)

        # Додаємо колонку relative_change (заповнену 0 для прогнозу)
        df = calculate_relative_change_for_prediction(df, meta['pred_len'])

        # Завантажуємо модель
        print("[5/7] Завантаження моделі...")
        model, device, used_unknown_reals = load_trained_model(model_dir, meta)

        if model is None:
            print("Помилка: Не вдалося завантажити модель")
            return TradeDecision(
                time="",
                signal=TradeSignal.HOLD,
                entry_price=0
            )

        # Масштабуємо дані ТІЛЬКИ ті ознаки, які використовуються в моделі
        print("[6/7] Масштабування даних...")
        features_to_scale = KNOWN_REALS + used_unknown_reals
        df_scaled = scale_data(df, scalers, features_to_scale)

        # Готуємо датасет для прогнозу
        print("[7/7] Підготовка даних для прогнозу...")
        predictions = []

        for symbol in df_scaled['symbol'].unique():
            symbol_data = df_scaled[df_scaled['symbol'] == symbol]
            original_symbol_data = df[df['symbol'] == symbol]

            # Перевіряємо, чи достатньо даних для цього символу
            if len(symbol_data) < meta['encoder_len'] + meta['pred_len']:
                print(f"Попередження: недостатньо даних для символу {symbol} ({len(symbol_data)} рядків)")
                continue

            # Беремо останній encoder_len + pred_len рядків
            latest_data = symbol_data.iloc[-(meta['encoder_len'] + meta['pred_len']):].copy()
            latest_original_data = original_symbol_data.iloc[-(meta['encoder_len'] + meta['pred_len']):].copy()

            # Створюємо датасет
            prediction_ds = create_prediction_dataset(latest_data, meta['encoder_len'], meta['pred_len'], meta,
                                                      used_unknown_reals)
            if prediction_ds is None:
                continue

            # Створюємо dataloader
            dataloader = prediction_ds.to_dataloader(train=False, batch_size=1, num_workers=0)

            close_scaler = scalers['close']

            # Прогнозуємо
            with torch.no_grad():
                for x, _ in dataloader:
                    # Переносимо дані на GPU/CPU
                    for key in x:
                        if isinstance(x[key], torch.Tensor):
                            x[key] = x[key].to(device)

                    out = model(x)
                    pred_tensor = out.prediction
                    pred_np = pred_tensor.detach().cpu().numpy()  # shape: [batch, pred_len, n_quantiles]

                    # Інверсне трансформування прогнозованих відносних змін у реальні ціни
                    pred_prices = np.zeros_like(pred_np)

                    for i in range(pred_np.shape[2]):  # для кожного квантиля
                        # relative_change -> normalized absolute price
                        pred_prices[:, :, i] = latest_data['close'].values[-1] * (1 + pred_np[:, :, i])

                    # Інверсне трансформування, якщо close була масштабована
                    pred_prices_real = close_scaler.inverse_transform(pred_prices.reshape(-1, 1)).reshape(
                        pred_prices.shape)

                    pred_entry = {
                        'symbol': symbol,
                        'predictions': pred_prices_real[0],  # беремо перший батч
                        'relative_changes': pred_np[0],  # залишаємо відносні зміни
                        'quantiles': meta['quantiles'],
                        'last_price': close_scaler.inverse_transform([[latest_data['close'].values[-1]]])[0, 0]
                        # реальна ціна
                    }

                    pred_prices = pred_prices[0]
                    q10 = pred_prices[:, pred_entry['quantiles'].index(0.1)]
                    q50 = pred_prices[:, pred_entry['quantiles'].index(0.5)]
                    q90 = pred_prices[:, pred_entry['quantiles'].index(0.9)]
                    current_price = latest_data['close'].values[-1]
                    confidence_metrics = model_confidence(q10, q50, q90, current_price)

                    pred_entry['confidence'] = confidence_metrics
                    predictions.append(pred_entry)

        # Вивід результатів та формування торгового рішення
        ai_signal = TradeSignal.HOLD
        last_open_time = None
        last_price = 0

        if not predictions:
            print("Не вдалося зробити прогноз для жодного символу")
            return TradeDecision(
                time="",
                signal=TradeSignal.HOLD,
                entry_price=0
            )

        for pred in predictions:
            last_open_time = df['open_time'].iloc[-1]
            confidence_metrics = pred['confidence']

            print(f"\nРезультати прогнозу для {pred['symbol']}:")
            print(f"Поточна ціна: {pred['last_price']:.6f}")
            print(f"Confidence Score: {confidence_metrics['confidence_score']:.1f}")
            print(f"MPIW: {confidence_metrics['MPIW_relative_percent']:.2f}%")
            print(f"Prob Bullish: {confidence_metrics['prob_bullish']:.2f}")

            last_price = pred['last_price']

            if float(confidence_metrics["confidence_score"]) > float(60):

                if confidence_metrics['prob_bullish'] < 0.4:
                    ai_signal = TradeSignal.SELL
                    print("СИГНАЛ: SELL")
                elif confidence_metrics['prob_bullish'] > 0.6:
                    ai_signal = TradeSignal.BUY
                    print("СИГНАЛ: BUY")

                # Записуємо результати в файл
                results_path = model_dir / "predictions.txt"
                with open(results_path, "a", encoding="utf-8") as f:
                    f.write("=" * 60 + "\n")
                    f.write(f"Символ: {pred['symbol']}\n")
                    f.write(f"Поточна ціна: {last_price:.6f}\n")
                    f.write(f"Час відкриття останньої свічки для прогнозу: {last_open_time}\n")
                    f.write(f"Прогноз на {meta['pred_len']} годин вперед:\n")

                    for i in range(meta['pred_len']):
                        f.write(f"  Година {i + 1}: {pred['predictions'][i, 1]:.6f} (медіана)\n")

                    if ai_signal == TradeSignal.SELL:
                        f.write(f"\nПОЗИЦІЯ: SELL\n")
                    elif ai_signal == TradeSignal.BUY:
                        f.write(f"\nПОЗИЦІЯ: BUY\n")
                    else:
                        f.write(f"\nПОЗИЦІЯ: HOLD\n")

                    f.write(f"\nМетрики впевненості:\n")
                    f.write(f"MPIW (абсолютний): {confidence_metrics['MPIW_absolute']:+.4f}\n")
                    f.write(f"MPIW (відносний): {confidence_metrics['MPIW_relative_percent']:+.2f}%\n")
                    f.write(f"Ймовірність зростання: {confidence_metrics['prob_bullish']:.3f}\n")
                    f.write(f"Різкість: {confidence_metrics['sharpness']:.4f}\n")
                    f.write(f"Узгодженість: {confidence_metrics['consistency']:.4f}\n")
                    f.write(f"Confidence Score: {confidence_metrics['confidence_score']:.1f}\n")
                    f.write("\n")

                break  # Обробляємо тільки перший символ

        return TradeDecision(
            time=str(last_open_time),
            signal=ai_signal,
            entry_price=last_price
        )

    except Exception as e:
        print(f"Критична помилка в get_ai_signal: {e}")
        import traceback
        traceback.print_exc()
        return TradeDecision(
            time="",
            signal=TradeSignal.HOLD,
            entry_price=0
        )


if __name__ == "__main__":
    decision = get_ai_signal()
    print(f"\nФінальне рішення: {decision.signal.name} по ціні {decision.entry_price:.6f}")