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


def build_argparser():
    """
        Створює парсер аргументів командного рядка для конфігурації прогнозування.

        Returns:
            argparse.ArgumentParser: Об'єкт парсера аргументів з налаштованими параметрами.

        Example:
            parser = build_argparser()
            args = parser.parse_args()
    """
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="postgresql://admin:admin_pass@localhost:5432/pompilo_db",
                   help="SQLAlchemy URL, e.g. postgresql://user:pass@host:5432/db")
    p.add_argument("--table", default="_candles_trading_data.xrpusdt_p_candles_test_data",
                   help="Table name with 1h candles")
    p.add_argument("--model-base-dir", default="./tft_runs",
                   help="Base directory with saved models and scalers")
    p.add_argument("--symbol", default="XRPUSDT", help="Specific symbol to predict (e.g., SOLUSDT)")
    p.add_argument("--hours", type=int, default=6, help="Number of hours to predict")
    return p


def get_symbol_folder_name(symbol: str) -> str:
    """
        Перетворює символ торгової пари на назву папки.

        Args:
            symbol (str): Символ торгової пари (наприклад, 'SOLUSDT').

        Returns:
            str: Нормалізована назва папки (наприклад, 'solusdt').
        Example:
            get_symbol_folder_name('SOL/USDT')
            'solusdt'
    """

    return symbol.lower().replace('/', '').replace('-', '')


def fetch_latest_data(db_url: str, table: str, symbol: str = None, limit: int = 120) -> pd.DataFrame:
    """
        Завантажує останні дані з бази даних PostgreSQL.

        Args:
            db_url (str): URL-адреса бази даних у форматі SQLAlchemy.
            table (str): Назва таблиці з даними свічок.
            symbol (str, optional): Символ торгової пари для фільтрації. Якщо None, завантажуються всі символи.
            limit (int, optional): Максимальна кількість рядків для завантаження. За замовчуванням 120.

        Returns:
            pd.DataFrame: DataFrame з даними свічок, відсортованими за часом.

        Raises:
            sqlalchemy.exc.DatabaseError: Якщо виникає помилка підключення до бази даних.
            ValueError: Якщо отримані дані містять некоректні значення часу або символів.

        Example:
            df = fetch_latest_data('postgresql://user:pass@localhost/db', 'candles', 'SOLUSDT')
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


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Додає часові ознаки до DataFrame для прогнозування.

        Args:
            df (pd.DataFrame): Вхідний DataFrame з даними свічок.

        Returns:
            pd.DataFrame: DataFrame з доданими часовими ознаками (time_idx, hour, dow, hour_sin, hour_cos, dow_sin, dow_cos).

        Raises:
            ValueError: Якщо вхідний DataFrame не містить необхідних стовпців або містить некоректні значення часу.

        Example:
            df = pd.DataFrame({'open_time': pd.to_datetime(['2023-01-01 00:00:00'])})
            df = add_time_features(df)
    """

    df = df.copy()

    # Створюємо time_idx для кожної групи символів
    df["time_idx"] = df.groupby("symbol").cumcount()

    # Безпечне отримання години та дня тижня
    df["hour"] = df["open_time"].dt.hour.astype(np.int16)
    df["dow"] = df["open_time"].dt.dayofweek.astype(np.int16)

    # Циклічне кодування
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    return df


def scale_data(df: pd.DataFrame, scalers: dict, features_to_scale: list) -> pd.DataFrame:
    """
        Масштабує числові ознаки в DataFrame за допомогою наданих скейлерів.

        Args:
            df (pd.DataFrame): Вхідний DataFrame з даними.
            scalers (dict): Словник зі скейлерами (наприклад, MinMaxScaler) для кожної ознаки.
            features_to_scale (list): Список назв стовпців для масштабування.

        Returns:
            pd.DataFrame: Масштабований DataFrame.

        Raises:
            KeyError: Якщо вказані ознаки відсутні в DataFrame або скейлерах.

        Example:
            from sklearn.preprocessing import MinMaxScaler
            scalers = {'open': MinMaxScaler()}
            df = scale_data(df, scalers, ['open'])
    """

    df_scaled = df.copy()
    for col in features_to_scale:
        if col in df_scaled.columns and col in scalers:
            df_scaled[col] = scalers[col].transform(df_scaled[[col]])
    return df_scaled


def create_prediction_dataset(df_scaled: pd.DataFrame, encoder_len: int, pred_len: int):
    """
        Створює датасет для прогнозування за допомогою TimeSeriesDataSet.

        Args:
            df_scaled (pd.DataFrame): Масштабований DataFrame з даними.
            encoder_len (int): Довжина вхідної послідовності (історичних даних).
            pred_len (int): Довжина прогнозованої послідовності.

        Returns:
            TimeSeriesDataSet or None: Об'єкт датасету для прогнозування або None у разі помилки.

        Raises:
            ValueError: Якщо недостатньо даних для створення датасету.
            Exception: Якщо виникають помилки при створенні TimeSeriesDataSet.

        Example:
            dataset = create_prediction_dataset(df_scaled, encoder_len=168, pred_len=6)
    """

    target = "relative_change"
    known_reals = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    unknown_reals = ["open", "high", "low", "volume", "cvd", "poc", "close"]

    # Перевіряємо, чи достатньо даних
    min_required_length = encoder_len + pred_len
    if len(df_scaled) < min_required_length:
        print(f"Попередження: недостатньо даних ({len(df_scaled)}), потрібно мінімум {min_required_length}")
        return None

    try:
        # Для прогнозу заповнюємо relative_change нулями (тимчасово)
        df_for_prediction = df_scaled.copy()
        df_for_prediction["relative_change"] = df_for_prediction["relative_change"].fillna(0)

        dataset = TimeSeriesDataSet(
            df_for_prediction,
            time_idx="time_idx",
            target=target,
            group_ids=["symbol"],
            min_encoder_length=encoder_len,
            max_encoder_length=encoder_len,
            min_prediction_length=pred_len,
            max_prediction_length=pred_len,
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=unknown_reals,
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
    """Завантажує навчену модель TemporalFusionTransformer.

        Args:
            model_dir (Path): Шлях до папки з файлами моделі.
            meta (dict): Словник з метаданими моделі (наприклад, encoder_len, pred_len, quantiles).

        Returns:
            tuple: Кортеж із моделлю (TemporalFusionTransformer) та пристроєм (torch.device).

        Raises:
            FileNotFoundError: Якщо файл моделі або метадані не знайдені.
            RuntimeError: Якщо не вдається завантажити ваги моделі.

        Example:
            model, device = load_trained_model(Path('./tft_runs/solusdt'), meta)
    """
    model_path = model_dir / "trading_tft_model.pth"

    # Створюємо модель з правильними параметрами
    quantiles = meta['quantiles']
    loss = QuantileLoss(quantiles=quantiles)

    # Створюємо dummy dataset для ініціалізації моделі
    dummy_data = pd.DataFrame({
        'time_idx': range(meta['encoder_len'] + meta['pred_len']),
        'symbol': [meta['symbols'][0]] * (meta['encoder_len'] + meta['pred_len']),
        'relative_change': np.random.randn(meta['encoder_len'] + meta['pred_len']),  # Змінено ціль
        'open': np.random.randn(meta['encoder_len'] + meta['pred_len']),
        'high': np.random.randn(meta['encoder_len'] + meta['pred_len']),
        'low': np.random.randn(meta['encoder_len'] + meta['pred_len']),
        'volume': np.random.randn(meta['encoder_len'] + meta['pred_len']),
        'cvd': np.random.randn(meta['encoder_len'] + meta['pred_len']),
        'poc': np.random.randn(meta['encoder_len'] + meta['pred_len']),
        'close': np.random.randn(meta['encoder_len'] + meta['pred_len']),  # Додано close як ознаку
        'hour_sin': np.sin(np.linspace(0, 2 * np.pi, meta['encoder_len'] + meta['pred_len'])),
        'hour_cos': np.cos(np.linspace(0, 2 * np.pi, meta['encoder_len'] + meta['pred_len'])),
        'dow_sin': np.sin(np.linspace(0, 2 * np.pi, meta['encoder_len'] + meta['pred_len'])),
        'dow_cos': np.cos(np.linspace(0, 2 * np.pi, meta['encoder_len'] + meta['pred_len'])),
    })

    known_reals = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    unknown_reals = ["open", "high", "low", "volume", "cvd", "poc", "close"]

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
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=None,
        add_relative_time_idx=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    model = TemporalFusionTransformer.from_dataset(
        dummy_dataset,
        loss=loss,
        hidden_size=128,
        attention_head_size=8,
        dropout=0.1,
        hidden_continuous_size=32,
        output_size=len(quantiles),
    )

    # Завантажуємо ваги
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device


def calculate_relative_change_for_prediction(df: pd.DataFrame, pred_len: int = 6) -> pd.DataFrame:
    """
        Додає колонку relative_change з NaN для прогнозування.

        Args:
            df (pd.DataFrame): Вхідний DataFrame з даними свічок.
            pred_len (int, optional): Довжина прогнозованої послідовності. За замовчуванням 6.

        Returns:
            pd.DataFrame: DataFrame з доданою колонкою relative_change, заповненою NaN.

        Example:
            df = calculate_relative_change_for_prediction(df, pred_len=6)
    """
    df = df.copy()

    # Для прогнозу нам потрібно створити колонку relative_change, але для останніх даних
    # ми не маємо майбутніх значень, тому заповнюємо NaN
    df["relative_change"] = np.nan

    return df


def model_confidence(q10, q50, q90, current_price):
    """
        Обчислює метрики впевненості прогнозу моделі.

        Args:
            q10 (np.ndarray): Прогнози для 10% квантиля.
            q50 (np.ndarray): Прогнози для 50% квантиля (медіана).
            q90 (np.ndarray): Прогнози для 90% квантиля.
            current_price (float): Поточна ціна активу.

        Returns:
            dict: Словник з метриками впевненості (MPIW, prob_bullish, sharpness, consistency, confidence_score).

        Example:
           metrics = model_confidence(q10, q50, q90, current_price=100.0)
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
    # Кореляція між послідовними прогнозами
    if len(q50) > 1:
        consistency = np.corrcoef(q50[:-1], q50[1:])[0, 1]
    else:
        consistency = 1.0  # Якщо лише один прогноз

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
    """Генерує торговий сигнал на основі прогнозів моделі.

        Returns:
            TradeDecision: Об'єкт з торговим сигналом, часом і вхідною ціною.

        Raises:
            FileNotFoundError: Якщо файли моделі або метадані не знайдені.
            ValueError: Якщо не вдалося отримати дані з бази або створити датасет для прогнозу.

        Example:
            decision = get_ai_signal()
            print(decision.signal, decision.entry_price)
    """
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

    print(f"[1/6] Завантаження метаданих: encoder_len={meta['encoder_len']}, pred_len={meta['pred_len']}")
    # print(f"Цільова змінна: {meta.get('target', 'relative_change')}")
    print(f"Модель завантажена з: {model_dir}")

    # Отримуємо дані для прогнозу
    print("[2/6] Отримання останніх даних з бази...")
    required_length = meta['encoder_len'] + meta['pred_len'] + 10
    df = fetch_latest_data(args.db, args.table, args.symbol, limit=required_length)

    if df.empty:
        print("Помилка: Не вдалося отримати дані з бази")
        return TradeDecision(
            time="",
            signal=TradeSignal.HOLD,
            entry_price=0
        )

    print(f"Отримано {len(df)} рядків даних")
    # print(f"Символи: {df['symbol'].unique()}")

    # Додаємо часові ознаки
    print("[3/6] Додавання часових ознак...")
    df = add_time_features(df)

    # Додаємо колонку relative_change (заповнену NaN для прогнозу)
    df = calculate_relative_change_for_prediction(df, meta['pred_len'])

    # Масштабуємо дані (тільки ті ознаки, для яких є скейлери)
    print("[4/6] Масштабування даних...")
    features_to_scale = [f for f in scalers.keys() if f != "relative_change"]
    df_scaled = scale_data(df, scalers, features_to_scale)

    # Додаємо колонку close до масштабованих даних (якщо вона не масштабувалася)
    if 'close' not in scalers:
        df_scaled['close'] = df['close'].values

    # Завантажуємо модель
    print("[5/6] Завантаження моделі...")
    model, device = load_trained_model(model_dir, meta)

    # Готуємо датасет для прогнозу
    print("[6/6] Підготовка даних для прогнозу...")
    predictions = []

    for symbol in df_scaled['symbol'].unique():
        symbol_data = df_scaled[df_scaled['symbol'] == symbol]

        # Перевіряємо, чи достатньо даних для цього символу
        if len(symbol_data) < meta['encoder_len'] + meta['pred_len']:
            print(f"Попередження: недостатньо даних для символу {symbol} ({len(symbol_data)} рядків)")
            continue

        # Беремо останній encoder_len + pred_len рядків
        latest_data = symbol_data.iloc[-(meta['encoder_len'] + meta['pred_len']):].copy()

        # Створюємо датасет
        prediction_ds = create_prediction_dataset(latest_data, meta['encoder_len'], meta['pred_len'])
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
                pred_prices_real = close_scaler.inverse_transform(pred_prices.reshape(-1, 1)).reshape(pred_prices.shape)

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
        last_open_time = latest_data['open_time'].iloc[-1]
        confidence_metrics = pred['confidence']

        if (float(confidence_metrics["MPIW_relative_percent"]) < float(4.5) and
                float(confidence_metrics["confidence_score"]) > float(60)):

            last_price = pred['last_price']

            if float(confidence_metrics['prob_bullish']) < float(0.4):
                ai_signal = TradeSignal.SELL
            elif float(confidence_metrics['prob_bullish']) > float(0.6):
                ai_signal = TradeSignal.BUY

            # Записуємо результати в файл
            results_path = model_dir / "predictions.txt"
            with open(results_path, "a", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write(f"Символ: {pred['symbol']}\n")
                f.write(f"Поточна ціна: {last_price:.6f}\n")
                f.write(f"Час відкриття останньої свічки для прогнозу: {last_open_time}\n")
                f.write(f"Прогноз на {meta['pred_len']} годин вперед:\n")

                if ai_signal == TradeSignal.SELL:
                    f.write(f"\nPosition: SELL\n")
                elif ai_signal == TradeSignal.BUY:
                    f.write(f"\nPosition: BUY\n")
                else:
                    f.write(f"\nPosition: HOLD\n")

                f.write(f"\nMPIW_absolute: {confidence_metrics['MPIW_absolute']:+.2f}%")
                f.write(f"\nMPIW_relative_percent: {confidence_metrics['MPIW_relative_percent']:+.2f}%")
                f.write(f"\nprob_bullish: {confidence_metrics['prob_bullish']:.2f}")
                f.write(f"\nsharpness: {confidence_metrics['sharpness']}")
                f.write(f"\nconsistency: {confidence_metrics['consistency']}")
                f.write(f"\nconfidence_score: {confidence_metrics['confidence_score']}\n")
                f.write("\n")

            break  # Обробляємо тільки перший символ

    return TradeDecision(
        time=str(last_open_time),
        signal=ai_signal,
        entry_price=last_price
    )


if __name__ == "__main__":
    decision = get_ai_signal()