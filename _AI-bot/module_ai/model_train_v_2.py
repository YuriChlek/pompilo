import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.preprocessing import MinMaxScaler
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric

from lightning.pytorch import Trainer
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks import Callback
import gc

target = "relative_change"
# Відомі часові ознаки (не залежать від майбутнього)
KNOWN_REALS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
# Невідомі ознаки (залежать від майбутнього)

UNKNOWN_REALS = [
    "open", "high", "low", "volume", "cvd", "poc", "close",
    "rsi", "bb_upper", "bb_lower", "bb_position", "volume_ma",
    "price_change_1h", "price_change_6h", "volatility"
]


class EnhancedTemporalLoss(MultiHorizonMetric):
    """
    Комбінована loss function, яка включає:
    1. Quantile Loss для розподілу прогнозів
    2. DTW (Dynamic Time Warping) для часового вирівнювання
    3. Phase penalty для штрафування фазового зсуву
    """

    def __init__(self, quantiles=[0.1, 0.5, 0.9], alpha=0.3, beta=0.2, gamma=0.1, **kwargs):
        super().__init__(**kwargs)
        self.quantile_loss = QuantileLoss(quantiles)
        self.alpha = alpha  # Вага для DTW
        self.beta = beta  # Вага для phase penalty
        self.gamma = gamma  # Вага для directional penalty
        self.quantiles = quantiles

    def dtw_distance(self, preds, targets):
        """
        Обчислює Dynamic Time Warping distance між прогнозами та цілями.
        Спрощена версія для батчів.
        """
        batch_size, seq_len = preds.shape

        # Створюємо матрицю відстаней
        dtw_matrix = torch.zeros(batch_size, seq_len, seq_len, device=preds.device)

        for i in range(seq_len):
            for j in range(seq_len):
                dtw_matrix[:, i, j] = torch.abs(preds[:, i] - targets[:, j])

        # Accumulated cost matrix
        acc_matrix = torch.zeros_like(dtw_matrix)
        acc_matrix[:, 0, 0] = dtw_matrix[:, 0, 0]

        for i in range(1, seq_len):
            acc_matrix[:, i, 0] = acc_matrix[:, i - 1, 0] + dtw_matrix[:, i, 0]

        for j in range(1, seq_len):
            acc_matrix[:, 0, j] = acc_matrix[:, 0, j - 1] + dtw_matrix[:, 0, j]

        for i in range(1, seq_len):
            for j in range(1, seq_len):
                acc_matrix[:, i, j] = dtw_matrix[:, i, j] + torch.min(
                    torch.stack([
                        acc_matrix[:, i - 1, j],
                        acc_matrix[:, i, j - 1],
                        acc_matrix[:, i - 1, j - 1]
                    ]), dim=0
                )[0]

        return acc_matrix[:, -1, -1].mean()

    def phase_penalty(self, preds, targets):
        """
        Штраф за фазовий зсув між прогнозами та цілями.
        Використовує кореляцію та різницю в градієнтах.
        """
        # Обчислюємо градієнти (похідні)
        pred_gradients = preds[:, 1:] - preds[:, :-1]
        target_gradients = targets[:, 1:] - targets[:, :-1]

        # Штраф за різницю в градієнтах
        gradient_penalty = F.mse_loss(pred_gradients, target_gradients)

        # Штраф за кореляцію (менша кореляція = більший штраф)
        batch_size = preds.shape[0]
        correlation_penalty = 0

        for i in range(batch_size):
            pred_series = preds[i] - preds[i].mean()
            target_series = targets[i] - targets[i].mean()

            if pred_series.norm() > 1e-8 and target_series.norm() > 1e-8:
                correlation = torch.dot(pred_series, target_series) / (
                        pred_series.norm() * target_series.norm() + 1e-8)
                # Перетворюємо кореляцію [-1, 1] у штраф [0, 2]
                correlation_penalty += (1 - correlation) / 2
            else:
                correlation_penalty += 1  # Максимальний штраф для постійних рядів

        correlation_penalty /= batch_size

        return gradient_penalty + correlation_penalty

    def directional_penalty(self, preds, targets):
        """
        Штраф за неправильний напрямок руху.
        """
        pred_direction = (preds[:, -1] - preds[:, 0]) > 0
        target_direction = (targets[:, -1] - targets[:, 0]) > 0

        # Бінарна втрата за напрямком
        direction_loss = F.binary_cross_entropy_with_logits(
            (preds[:, -1] - preds[:, 0]).float(),
            target_direction.float()
        )

        return direction_loss

    def loss(self, y_pred, y_actual):
        """
        Обчислює комбіновану втрату.

        Args:
            y_pred: Прогнози моделі [batch_size, seq_len, n_quantiles]
            y_actual: Цільові значення [batch_size, seq_len]
        """
        # Базовий quantile loss
        quant_loss = self.quantile_loss(y_pred, y_actual)

        # Витягуємо медіанний прогноз (quantile 0.5)
        median_preds = y_pred[..., 1]  # [batch_size, seq_len]

        # DTW loss
        dtw_loss = self.dtw_distance(median_preds, y_actual)

        # Phase penalty
        phase_loss = self.phase_penalty(median_preds, y_actual)

        # Directional penalty
        direction_loss = self.directional_penalty(median_preds, y_actual)

        # Комбінована втрата
        total_loss = (quant_loss +
                      self.alpha * dtw_loss +
                      self.beta * phase_loss +
                      self.gamma * direction_loss)

        return total_loss

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Конвертує прогнози у точкові прогнози (медіану).
        """
        if y_pred.ndim == 3:
            return y_pred[..., 1]  # медіана
        elif y_pred.ndim == 2:
            return y_pred
        else:
            return y_pred

    def to_quantiles(self, y_pred: torch.Tensor, quantiles=None) -> torch.Tensor:
        """
        Конвертує прогнози у квантилі.
        """
        if quantiles is None:
            quantiles = self.quantiles

        if y_pred.ndim == 3:
            return y_pred
        else:
            # Якщо вхід вже квантилі, повертаємо як є
            return y_pred.unsqueeze(-1).expand(-1, -1, len(quantiles))


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, help="SQLAlchemy URL, e.g. postgresql://user:pass@host:5432/db")
    p.add_argument("--table", required=True, help="Table name with 1h candles")
    p.add_argument("--save-dir", default="./tft_runs", help="Base directory to save models & scalers")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--encoder-len", type=int, default=168)
    p.add_argument("--pred-len", type=int, default=6,
                   help="Prediction length (horizon)")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Last fraction of data as validation")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--sample-ratio", type=float, default=1.0, help="Fraction of data to use for training (0.1-1.0)")
    p.add_argument("--use-temporal-loss", action="store_true",
                   help="Use enhanced temporal loss with DTW and phase penalties")
    p.add_argument("--dtw-alpha", type=float, default=0.3, help="Weight for DTW loss component")
    p.add_argument("--phase-beta", type=float, default=0.2, help="Weight for phase penalty")
    p.add_argument("--direction-gamma", type=float, default=0.1, help="Weight for directional penalty")
    return p


def read_postgres_chunked(db_url: str, table: str, chunk_size: int = 5000) -> pd.DataFrame:
    eng = create_engine(db_url)
    chunks = []

    with eng.begin() as conn:
        total_count = pd.read_sql(text(f"SELECT COUNT(*) FROM {table}"), conn).iloc[0, 0]

        for offset in range(0, total_count, chunk_size):
            chunk = pd.read_sql(
                text(f"""
                    SELECT open_time, close_time, symbol, open, close, high, low, cvd, volume, poc 
                    FROM {table} 
                    ORDER BY open_time ASC
                    LIMIT {chunk_size} OFFSET {offset}
                """),
                conn
            )
            chunk["open_time"] = pd.to_datetime(chunk["open_time"], utc=True, errors="coerce")
            chunk["close_time"] = pd.to_datetime(chunk["close_time"], utc=True, errors="coerce")

            chunks.append(chunk)
            print(f"Loaded chunk {offset // chunk_size + 1}/{(total_count - 1) // chunk_size + 1}")

    result_df = pd.concat(chunks, ignore_index=True)
    return result_df.dropna(subset=["open_time", "symbol", "close"])


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = ['open', 'high', 'low', 'close', 'volume', 'cvd', 'poc']
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')

    df['symbol'] = df['symbol'].astype('category')
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Додає технічні індикатори для покращення прогнозів.
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

        # Bollinger Bands
        rolling_mean = symbol_data['close'].rolling(window=20, min_periods=1).mean()
        rolling_std = symbol_data['close'].rolling(window=20, min_periods=1).std()
        symbol_data['bb_upper'] = rolling_mean + (rolling_std * 2)
        symbol_data['bb_lower'] = rolling_mean - (rolling_std * 2)
        symbol_data['bb_position'] = (symbol_data['close'] - symbol_data['bb_lower']) / (
                symbol_data['bb_upper'] - symbol_data['bb_lower'])

        # Volume-based features
        symbol_data['volume_ma'] = symbol_data['volume'].rolling(window=20, min_periods=1).mean()

        # Price momentum
        symbol_data['price_change_1h'] = symbol_data['close'].pct_change(1)
        symbol_data['price_change_6h'] = symbol_data['close'].pct_change(6)
        symbol_data['volatility'] = symbol_data['price_change_1h'].rolling(window=24, min_periods=1).std()

        # Оновлюємо дані в основному DataFrame
        for col in ['rsi', 'bb_upper', 'bb_lower',
                    'bb_position', 'volume_ma', 'price_change_1h',
                    'price_change_6h', 'volatility']:
            df.loc[symbol_mask, col] = symbol_data[col].values

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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


def calculate_enhanced_targets(df: pd.DataFrame, pred_len: int = 6) -> pd.DataFrame:
    """
    Розраховує розширені цільові змінні для багатозадачного навчання.
    """
    df = df.copy()

    # Базова відносна зміна
    df["relative_change"] = df.groupby("symbol")["close"].shift(-pred_len) / df["close"] - 1

    # Додаткові цілі для багатозадачного навчання
    df["future_high_ratio"] = df.groupby("symbol")["high"].shift(-pred_len) / df["close"] - 1
    df["future_low_ratio"] = df.groupby("symbol")["low"].shift(-pred_len) / df["close"] - 1

    # Волатильність як додаткова ціль
    df["future_volatility"] = (df["future_high_ratio"] - df["future_low_ratio"]).abs()

    # Бінарна ціль для напрямку руху
    df["direction_target"] = (df["relative_change"] > 0).astype(np.float32)

    # Видаляємо рядки з NaN значеннями цільових змінних
    target_columns = ["relative_change", "future_high_ratio", "future_low_ratio", "future_volatility",
                      "direction_target"]
    df = df.dropna(subset=target_columns)

    return df


def split_train_valid(df: pd.DataFrame, val_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dfs = []
    val_dfs = []
    print(f"Validation ratio: {val_ratio}")

    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("open_time")
        n = len(group)
        cut_idx = int(n * (1 - val_ratio))

        train_dfs.append(group.iloc[:cut_idx])
        val_dfs.append(group.iloc[cut_idx:])

    return pd.concat(train_dfs), pd.concat(val_dfs)


def fit_transform_scalers(train_df: pd.DataFrame, val_df: pd.DataFrame, save_dir: Path, features: list[str]) -> tuple[
    pd.DataFrame, pd.DataFrame]:
    # Не масштабуємо цільові змінні
    target_columns = ["relative_change", "future_high_ratio", "future_low_ratio", "future_volatility",
                      "direction_target"]
    features_to_scale = [f for f in features if f not in target_columns]

    scalers = {}

    for col in features_to_scale:
        sc = MinMaxScaler()
        train_data = train_df[[col]].replace([np.inf, -np.inf], np.nan).fillna(0)
        sc.fit(train_data)
        train_df[col] = sc.transform(train_df[[col]].replace([np.inf, -np.inf], np.nan).fillna(0))
        val_df[col] = sc.transform(val_df[[col]].replace([np.inf, -np.inf], np.nan).fillna(0))
        scalers[col] = sc

    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scalers, save_dir / "scalers.pkl")
    print(f"Saved scalers for {len(scalers)} features")

    return train_df, val_df


def build_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, encoder_len: int, pred_len: int,
                   sample_ratio: float = 1.0):
    # Основна цільова змінна

    combined_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    training_cutoff_values = combined_df.groupby("symbol")["time_idx"].max() - pred_len
    combined_df["symbol_str"] = combined_df["symbol"].astype(str)

    if sample_ratio < 1.0:
        if combined_df['symbol'].nunique() == 1:
            symbol = combined_df['symbol'].iloc[0]
            cutoff = training_cutoff_values[symbol]
            train_mask = combined_df["time_idx"] <= cutoff
            available_train = combined_df[train_mask]
            n_samples = max(encoder_len + pred_len, int(len(available_train) * sample_ratio))
            train_df_sampled = available_train.tail(n_samples)
        else:
            unique_symbols = list(combined_df['symbol'].unique())
            selected_symbols = np.random.choice(
                unique_symbols,
                size=int(len(unique_symbols) * sample_ratio),
                replace=False
            )
            train_mask = combined_df["symbol"].isin(selected_symbols)
            for symbol in selected_symbols:
                cutoff = training_cutoff_values[symbol]
                train_mask &= (combined_df["symbol"] == symbol) & (combined_df["time_idx"] <= cutoff)
            train_df_sampled = combined_df[train_mask]
    else:
        train_mask = pd.Series(False, index=combined_df.index)
        for symbol, cutoff in training_cutoff_values.items():
            train_mask |= (combined_df["symbol"] == symbol) & (combined_df["time_idx"] <= cutoff)
        train_df_sampled = combined_df[train_mask]

    training = TimeSeriesDataSet(
        train_df_sampled,
        time_idx="time_idx",
        target=target,
        group_ids=["symbol_str"],
        min_encoder_length=encoder_len,
        max_encoder_length=encoder_len,
        min_prediction_length=pred_len,
        max_prediction_length=pred_len,
        time_varying_known_reals=KNOWN_REALS,
        time_varying_unknown_reals=UNKNOWN_REALS,
        target_normalizer=None,
        add_relative_time_idx=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, combined_df, predict=True, stop_randomization=True
    )

    return training, validation, combined_df


def create_optimized_tft_model(training_ds, dataset_size: int, args):
    """
    Створює TFT модель з оптимізованими параметрами на основі розміру датасету.
    """
    quantiles = [0.1, 0.5, 0.9]

    # Вибираємо loss function
    if args.use_temporal_loss:
        loss = EnhancedTemporalLoss(
            quantiles=quantiles,
            alpha=args.dtw_alpha,
            beta=args.phase_beta,
            gamma=args.direction_gamma,
            reduction="mean"
        )
        print(f"Using enhanced temporal loss with DTW (alpha={args.dtw_alpha}), "
              f"phase penalty (beta={args.phase_beta}), direction penalty (gamma={args.direction_gamma})")
    else:
        loss = QuantileLoss(quantiles=quantiles)
        print("Using standard Quantile Loss")

    config = {
        "hidden_size": 128,
        "attention_head_size": 8,
        "dropout": 0.1,
        "hidden_continuous_size": 32,
        "learning_rate": 8e-4,
    }

    model = TemporalFusionTransformer.from_dataset(
        training_ds,
        loss=loss,
        **config,
        output_size=len(quantiles),
        log_interval=50,
        reduce_on_plateau_patience=4,
    )

    print(f"Expected prediction shape: [batch_size, {args.pred_len}, {len(quantiles)}]")
    return model


def create_optimized_dataloaders(training_ds, validation_ds, dataset_size: int):
    """
    Створює оптимізовані DataLoader-и з адаптивним batch size.
    """
    # Адаптивний batch size на основі розміру датасету

    if dataset_size < 10000:
        batch_size = 32
    elif dataset_size < 50000:
        batch_size = 64
    else:
        batch_size = 128

    print(f"Using batch size: {batch_size} for dataset size: {dataset_size}")

    train_loader = training_ds.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
        drop_last=True
    )

    val_loader = validation_ds.to_dataloader(
        train=False,
        batch_size=batch_size * 2,
        num_workers=2,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, batch_size


class MemoryCleanupCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LossMonitoringCallback(Callback):
    """Callback для моніторингу компонентів loss function"""

    def on_train_epoch_end(self, trainer, pl_module):
        # Для спрощення, ми не будемо моніторити компоненти окремо
        # Можна додати цю функціональність пізніше
        pass


class EnhancedTrainingCallbacks:
    """Покращені callback-и для навчання"""

    @staticmethod
    def get_callbacks(save_dir: Path, use_temporal_loss=False):
        ckpt_dir = save_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=12,
                mode="min",
                min_delta=0.001
            ),

            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="tft-{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_weights_only=False,
                every_n_epochs=1
            ),

            LearningRateMonitor(logging_interval="epoch"),
            MemoryCleanupCallback(),
        ]

        if use_temporal_loss:
            callbacks.append(LossMonitoringCallback())

        return callbacks


def get_symbol_from_data(df: pd.DataFrame) -> str:
    """Отримує символ з даних (якщо один символ) або створює загальну назву для кількох символів"""
    symbols = df['symbol'].unique()
    if len(symbols) == 1:
        return symbols[0].lower().replace('/', '').replace('-', '')
    else:
        return f"multi_{len(symbols)}_symbols"


def main():
    args = build_argparser().parse_args()
    seed_everything(args.seed)

    print("[1/8] Loading data from PostgreSQL (chunked)…")
    df = read_postgres_chunked(args.db, args.table)
    df = optimize_dataframe(df)

    if "hight" in df.columns and "high" not in df.columns:
        df = df.rename(columns={"hight": "high"})

    # Отримуємо символ для структури папок
    symbol_name = get_symbol_from_data(df)
    save_dir = Path(args.save_dir) / symbol_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {save_dir}")
    torch.set_float32_matmul_precision('medium')

    df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    print(f"Dataset stats: Rows: {len(df)}, Symbols: {df['symbol'].nunique()}")

    print("[2/8] Building time features…")
    df = add_time_features(df)

    print("[2.5/8] Adding technical indicators…")
    df = add_technical_indicators(df)

    print("[2.7/8] Calculating enhanced targets…")
    df = calculate_enhanced_targets(df, args.pred_len)

    print("[3/8] Train/validation split…")
    train_df, val_df = split_train_valid(df, val_ratio=args.val_ratio)
    all_symbols = sorted(pd.concat([train_df, val_df])["symbol"].unique())

    # Розширений список фіч для масштабування
    features_to_scale = [
        "open", "high", "low", "volume", "cvd", "poc", "close",
        "rsi", "bb_upper",
        "bb_lower", "bb_position", "volume_ma",
        "price_change_1h", "price_change_6h", "volatility"
    ]

    print("[4/8] Fit scalers on TRAIN and transform…")
    train_df, val_df = fit_transform_scalers(train_df, val_df, save_dir, features_to_scale)

    print("[5/8] Build TFT datasets…")
    training_ds, validation_ds, combined_df = build_datasets(
        train_df, val_df, args.encoder_len, args.pred_len, args.sample_ratio
    )

    print("[5.5/8] Create optimized dataloaders…")
    train_loader, val_loader, batch_size = create_optimized_dataloaders(
        training_ds, validation_ds, len(combined_df)
    )

    print("[6/8] Configure optimized TFT model…")
    tft = create_optimized_tft_model(training_ds, len(combined_df), args)

    print("[6.5/8] Setup enhanced training…")
    callbacks = EnhancedTrainingCallbacks.get_callbacks(save_dir, args.use_temporal_loss)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    accumulate_grad_batches = max(1, 64 // batch_size)

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=0.1,
        accumulate_grad_batches=accumulate_grad_batches,
        deterministic=True,
        precision="16-mixed" if accelerator == "gpu" else "32-true",
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(save_dir), name="tft_logs"),
        enable_progress_bar=True,
    )

    print("[7/8] Training…")
    trainer.fit(tft, train_loader, val_loader)

    best_model_path = trainer.checkpoint_callback.best_model_path or str(save_dir / "checkpoints" / "last.ckpt")
    print(f"Loading best model from: {best_model_path}")

    try:
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    except Exception as e:
        print(f"Error loading best model: {e}, using final model instead")
        best_tft = tft

    best_tft_path = save_dir / "trading_tft_model.pth"
    torch.save(best_tft.state_dict(), best_tft_path)
    print(f"Saved best TFT model to {best_tft_path}")

    meta = {
        "encoder_len": args.encoder_len,
        "pred_len": args.pred_len,
        "features_scaled": features_to_scale,
        "known_reals": KNOWN_REALS,
        "unknown_reals": UNKNOWN_REALS,
        "quantiles": [0.1, 0.5, 0.9],
        "symbols": [str(s) for s in all_symbols],
        "best_ckpt": best_model_path,
        "target": "relative_change",
        "symbol_name": symbol_name,
        "dataset_size": len(combined_df),
        "technical_indicators": [
            "rsi", "bb_upper",
            "bb_lower", "bb_position", "volume_ma",
            "price_change_1h", "price_change_6h", "volatility"
        ],
        "model_config": {
            "hidden_size": tft.hparams.hidden_size,
            "attention_head_size": tft.hparams.attention_head_size,
            "dropout": tft.hparams.dropout,
            "hidden_continuous_size": tft.hparams.hidden_continuous_size,
        },
        "temporal_loss_config": {
            "use_temporal_loss": args.use_temporal_loss,
            "dtw_alpha": args.dtw_alpha,
            "phase_beta": args.phase_beta,
            "direction_gamma": args.direction_gamma
        } if args.use_temporal_loss else {}
    }
    joblib.dump(meta, save_dir / "dataset_meta.pkl")
    print(f"Training completed. Results saved to {save_dir}")

    # Вивід статистики
    print(f"\nTraining Summary:")
    print(f"- Symbols: {len(all_symbols)}")
    print(f"- Total samples: {len(combined_df)}")
    print(f"- Features: {len(features_to_scale)}")
    print(f"- Technical indicators: {len(meta['technical_indicators'])}")
    print(f"- Temporal loss: {'Enabled' if args.use_temporal_loss else 'Disabled'}")
    if args.use_temporal_loss:
        print(f"- DTW weight: {args.dtw_alpha}")
        print(f"- Phase penalty weight: {args.phase_beta}")
        print(f"- Direction penalty weight: {args.direction_gamma}")
    print(f"- Best validation loss: {trainer.checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()