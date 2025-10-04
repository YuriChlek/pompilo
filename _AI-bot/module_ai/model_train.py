"""
    Temporal Fusion Transformer (TFT) Training Pipeline for Crypto Forecasting

    Перероблено для прогнозування відносної зміни ціни замість абсолютного значення.
"""

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
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from lightning.pytorch import Trainer
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks import Callback
import gc


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, help="SQLAlchemy URL, e.g. postgresql://user:pass@host:5432/db")
    p.add_argument("--table", required=True, help="Table name with 1h candles")
    p.add_argument("--save-dir", default="./tft_runs", help="Base directory to save models & scalers")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--encoder-len", type=int, default=168)
    p.add_argument("--pred-len", type=int, default=6)
    p.add_argument("--val-ratio", type=float, default=0.2, help="Last fraction of data as validation")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--sample-ratio", type=float, default=1.0, help="Fraction of data to use for training (0.1-1.0)")
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


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    df["time_idx"] = df.groupby("symbol").cumcount()

    df["hour"] = df["open_time"].dt.hour.astype(np.int16)
    df["dow"] = df["open_time"].dt.dayofweek.astype(np.int16)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    return df


def calculate_relative_change(df: pd.DataFrame, pred_len: int = 6) -> pd.DataFrame:
    """
    Розраховує відносну зміну ціни для прогнозу.

    Args:
        df: DataFrame з даними
        pred_len: Кількість майбутніх свічок для прогнозу

    Returns:
        DataFrame з доданою цільовою змінною 'relative_change'
    """
    df = df.copy()

    # Групуємо по символам для коректного розрахунку
    grouped = df.groupby("symbol")

    # Створюємо цільову змінну - відносна зміна через pred_len свічок
    df["relative_change"] = grouped["close"].shift(-pred_len) / df["close"] - 1

    # Видаляємо рядки, де значення цільової змінної NaN (останні pred_len рядків для кожного символу)
    df = df.dropna(subset=["relative_change"])

    return df


def split_train_valid(df: pd.DataFrame, val_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dfs = []
    val_dfs = []

    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("open_time")
        n = len(group)
        cut_idx = int(n * (1 - val_ratio))

        train_dfs.append(group.iloc[:cut_idx])
        val_dfs.append(group.iloc[cut_idx:])

    return pd.concat(train_dfs), pd.concat(val_dfs)


def fit_transform_scalers(train_df: pd.DataFrame, val_df: pd.DataFrame, save_dir: Path, features: list[str]) -> tuple[
    pd.DataFrame, pd.DataFrame]:
    features = [f for f in features if f != "relative_change"]  # Не масштабуємо цільову змінну

    scalers = {}

    for col in features:
        sc = MinMaxScaler()
        sc.fit(train_df[[col]])
        train_df[col] = sc.transform(train_df[[col]])
        val_df[col] = sc.transform(val_df[[col]])
        scalers[col] = sc

    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scalers, save_dir / "scalers.pkl")

    return train_df, val_df


def build_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, encoder_len: int, pred_len: int,
                   sample_ratio: float = 1.0):
    # Змінюємо цільову змінну на relative_change
    target = "relative_change"
    known_reals = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    unknown_reals = ["open", "high", "low", "volume", "cvd", "poc", "close"]  # Додаємо close до ознак

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
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=None,
        add_relative_time_idx=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, combined_df, predict=True, stop_randomization=True
    )

    return training, validation, combined_df


class MemoryCleanupCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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

    # Спочатку завантажуємо дані, щоб визначити символ
    print("[1/7] Loading data from PostgreSQL (chunked)…")
    df = read_postgres_chunked(args.db, args.table)
    df = optimize_dataframe(df)

    if "hight" in df.columns and "high" not in df.columns:
        df = df.rename(columns={"hight": "high"})

    # Отримуємо символ для структури папок
    symbol_name = get_symbol_from_data(df)

    # Створюємо шлях для збереження: tft_runs/symbol/
    save_dir = Path(args.save_dir) / symbol_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {save_dir}")

    torch.set_float32_matmul_precision('medium')

    df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    print("Rows:", len(df), "Symbols:", df["symbol"].nunique())

    print("[2/7] Building time features…")
    df = add_time_features(df)

    print("[2.5/7] Calculating relative price change…")
    df = calculate_relative_change(df, args.pred_len)

    print("[3/7] Train/validation split…")
    train_df, val_df = split_train_valid(df, val_ratio=args.val_ratio)
    all_symbols = sorted(pd.concat([train_df, val_df])["symbol"].unique())

    features_to_scale = ["open", "high", "low", "volume", "cvd", "poc", "close"]

    print("[4/7] Fit scalers on TRAIN and transform…")
    train_df, val_df = fit_transform_scalers(train_df, val_df, save_dir, features_to_scale)

    print("[5/7] Build TFT datasets…")
    training_ds, validation_ds, combined_df = build_datasets(
        train_df, val_df, args.encoder_len, args.pred_len, args.sample_ratio
    )

    batch_size = min(args.batch_size, 128)
    print("batch_size", batch_size)
    train_loader = training_ds.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=2,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = validation_ds.to_dataloader(
        train=False,
        batch_size=batch_size * 2,
        num_workers=1,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available()
    )

    del df, train_df, val_df, combined_df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[6/7] Configure TFT model & trainer…")
    quantiles = [0.1, 0.5, 0.9]
    loss = QuantileLoss(quantiles=quantiles)

    """
        hidden_size=32
        attention_head_size=2
        hidden_continuous_size=16

        hidden_size=64,
        attention_head_size=4,
        hidden_continuous_size=32,

        hidden_size=64-128
        attention_head_size=4-8
        hidden_continuous_size=32-64
        dropout=0.1
        
        малий датасет
        
        tft = TemporalFusionTransformer.from_dataset(
        training_ds,
        loss=loss,
        learning_rate=1e-3,
        hidden_size=64,              # менший розмір LSTM
        attention_head_size=4,       # ділиться на 64
        dropout=0.2,                 # трохи більше регуляризації
        hidden_continuous_size=16,   # компактні ембеддинги
        output_size=len(quantiles),
        log_interval=50,
        reduce_on_plateau_patience=6,
    )
    
        середній датасет
    
        tft = TemporalFusionTransformer.from_dataset(
            training_ds,
            loss=loss,
            learning_rate=1e-3,
            hidden_size=128,
            attention_head_size=8,
            dropout=0.1,
            hidden_continuous_size=32,
            output_size=len(quantiles),
            log_interval=50,
            reduce_on_plateau_patience=4,
        )
    
        великий датасет
    
        ft = TemporalFusionTransformer.from_dataset(
            training_ds,
            loss=loss,
            learning_rate=5e-4,          # нижчий LR для стабільності
            hidden_size=256,             # більше пам’яті і параметрів
            attention_head_size=8,       # ділиться на 256
            dropout=0.2,                 # щоб уникнути оверфіту
            hidden_continuous_size=64,   # потужніші ембеддинги
            output_size=len(quantiles),
            log_interval=100,            # рідше логувати, бо довше навчання
            reduce_on_plateau_patience=8,
        )
    
    """
    """
        tft = TemporalFusionTransformer.from_dataset(
            training_ds,
            loss=loss,
            learning_rate=1e-3,
            hidden_size=128,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=32,
            output_size=len(quantiles),
            log_interval=50,
            reduce_on_plateau_patience=4,
        )
    """

    tft = TemporalFusionTransformer.from_dataset(
        training_ds,
        loss=loss,
        learning_rate=1e-3,
        hidden_size=128,
        attention_head_size=8,
        dropout=0.1,
        hidden_continuous_size=32,
        output_size=len(quantiles),
        log_interval=50,
        reduce_on_plateau_patience=4,
    )

    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    lr_logger = LearningRateMonitor(logging_interval="step")
    early_stop = EarlyStopping(monitor="val_loss", patience=8, mode="min")
    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=2
    )

    logger = CSVLogger(save_dir=str(save_dir), name="tft_logs")
    memory_callback = MemoryCleanupCallback()

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
        callbacks=[early_stop, checkpoint, lr_logger, memory_callback],
        logger=logger,
        enable_progress_bar=True,
    )

    print("[7/7] Training…")
    trainer.fit(tft, train_loader, val_loader)

    best_model_path = checkpoint.best_model_path or str(ckpt_dir / "last.ckpt")
    print(f"Loading best model from: {best_model_path}")

    try:
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    except:
        best_tft = tft

    best_tft_path = save_dir / "trading_tft_model.pth"
    torch.save(best_tft.state_dict(), best_tft_path)
    print(f"Saved best TFT model to {best_tft_path}")

    meta = {
        "encoder_len": args.encoder_len,
        "pred_len": args.pred_len,
        "features_scaled": features_to_scale,
        "quantiles": quantiles,
        "symbols": [str(s) for s in all_symbols],
        "best_ckpt": best_model_path,
        "target": "relative_change",
        "symbol_name": symbol_name,  # Додаємо інформацію про символ
    }
    joblib.dump(meta, save_dir / "dataset_meta.pkl")
    print(f"Training completed. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
