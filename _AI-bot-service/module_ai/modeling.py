from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from module_ai.symbols import normalize_training_symbols

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - optional dependency
    CatBoostRegressor = None


QUANTILES = (0.1, 0.5, 0.9)
MODEL_ARTIFACT_NAME = "forecast_model.joblib"
MODEL_BACKEND_GRADIENT_BOOSTING = "gradient_boosting"
MODEL_BACKEND_CATBOOST = "catboost"
COMPUTE_DEVICE_AUTO = "auto"
COMPUTE_DEVICE_CPU = "cpu"
COMPUTE_DEVICE_GPU = "gpu"
SUPPORTED_COMPUTE_DEVICES = (
    COMPUTE_DEVICE_AUTO,
    COMPUTE_DEVICE_CPU,
    COMPUTE_DEVICE_GPU,
)
SUPPORTED_MODEL_BACKENDS = (
    MODEL_BACKEND_GRADIENT_BOOSTING,
    MODEL_BACKEND_CATBOOST,
)


@dataclass(frozen=True)
class ForecastResult:
    symbol: str
    origin_timestamp: str
    horizon_h: int
    q10_target_return_h: float
    q50_target_return_h: float
    q90_target_return_h: float
    last_real_close: float | None = None
    projected_price_h_q10: float | None = None
    projected_price_h_q50: float | None = None
    projected_price_h_q90: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "origin_timestamp": self.origin_timestamp,
            "horizon_h": self.horizon_h,
            "q10_target_return_h": self.q10_target_return_h,
            "q50_target_return_h": self.q50_target_return_h,
            "q90_target_return_h": self.q90_target_return_h,
            "last_real_close": self.last_real_close,
            "projected_price_h_q10": self.projected_price_h_q10,
            "projected_price_h_q50": self.projected_price_h_q50,
            "projected_price_h_q90": self.projected_price_h_q90,
        }


def detect_nvidia_gpu() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return {
            "available": False,
            "driver": None,
            "devices": [],
            "reason": "nvidia_smi_not_found",
        }

    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:
        return {
            "available": False,
            "driver": nvidia_smi,
            "devices": [],
            "reason": f"nvidia_smi_query_failed:{type(exc).__name__}",
        }

    devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {
        "available": bool(devices),
        "driver": nvidia_smi,
        "devices": devices,
        "reason": "ok" if devices else "no_visible_devices",
    }


def resolve_compute_device(
    *,
    model_backend: str,
    compute_device: str,
    gpu_device_id: int = 0,
) -> dict[str, Any]:
    requested = str(compute_device).strip().lower()
    if requested not in SUPPORTED_COMPUTE_DEVICES:
        raise ValueError(f"unsupported compute_device: {compute_device}")

    gpu_probe = detect_nvidia_gpu()
    backend_supports_gpu = model_backend == MODEL_BACKEND_CATBOOST

    if requested == COMPUTE_DEVICE_CPU:
        return {
            "requested": requested,
            "resolved": COMPUTE_DEVICE_CPU,
            "gpu_probe": gpu_probe,
            "gpu_device_id": int(gpu_device_id),
            "reason": "forced_cpu",
        }

    if requested == COMPUTE_DEVICE_GPU:
        if not backend_supports_gpu:
            raise ValueError(f"model_backend={model_backend} does not support GPU training")
        if not gpu_probe["available"]:
            raise ValueError("GPU training was requested but no NVIDIA GPU is available")
        return {
            "requested": requested,
            "resolved": COMPUTE_DEVICE_GPU,
            "gpu_probe": gpu_probe,
            "gpu_device_id": int(gpu_device_id),
            "reason": "forced_gpu",
        }

    if backend_supports_gpu and gpu_probe["available"]:
        return {
            "requested": requested,
            "resolved": COMPUTE_DEVICE_GPU,
            "gpu_probe": gpu_probe,
            "gpu_device_id": int(gpu_device_id),
            "reason": "auto_gpu_available",
        }

    return {
        "requested": requested,
        "resolved": COMPUTE_DEVICE_CPU,
        "gpu_probe": gpu_probe,
        "gpu_device_id": int(gpu_device_id),
        "reason": "auto_cpu_fallback",
    }


def _validate_supervised_arrays(X: np.ndarray, y: np.ndarray) -> None:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    if len(X) == 0:
        raise ValueError("training data is empty")


def _train_gradient_boosting_quantile_models(
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    compute_device_info: dict[str, Any],
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for quantile in QUANTILES:
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=quantile,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
        )
        model.fit(X, y)
        models[f"q{int(quantile * 100):02d}"] = model

    return {
        "quantiles": list(QUANTILES),
        "models": models,
        "model_type": "gradient_boosting_quantile",
        "model_backend": MODEL_BACKEND_GRADIENT_BOOSTING,
        "compute_device_requested": compute_device_info["requested"],
        "compute_device_resolved": COMPUTE_DEVICE_CPU,
        "gpu_probe": compute_device_info["gpu_probe"],
        "model_params": {
            "random_state": random_state,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "compute_device_reason": compute_device_info["reason"],
        },
    }


def _train_catboost_quantile_models(
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    compute_device_info: dict[str, Any],
) -> dict[str, Any]:
    if CatBoostRegressor is None:
        raise ImportError(
            "catboost is not installed. Install project dependencies from requirements.txt to use --model-backend catboost."
        )

    models: dict[str, Any] = {}
    resolved_device = compute_device_info["resolved"]
    resolved_reason = compute_device_info["reason"]
    for quantile in QUANTILES:
        model_kwargs = {
            "loss_function": f"Quantile:alpha={quantile}",
            "random_seed": random_state,
            "iterations": n_estimators,
            "depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "verbose": False,
            "allow_writing_files": False,
        }
        if resolved_device == COMPUTE_DEVICE_GPU:
            model_kwargs["task_type"] = "GPU"
            model_kwargs["devices"] = str(compute_device_info["gpu_device_id"])

        model = CatBoostRegressor(**model_kwargs)
        try:
            model.fit(X, y)
        except Exception:
            if compute_device_info["requested"] == COMPUTE_DEVICE_AUTO and resolved_device == COMPUTE_DEVICE_GPU:
                fallback_kwargs = dict(model_kwargs)
                fallback_kwargs.pop("task_type", None)
                fallback_kwargs.pop("devices", None)
                model = CatBoostRegressor(**fallback_kwargs)
                model.fit(X, y)
                resolved_device = COMPUTE_DEVICE_CPU
                resolved_reason = "auto_gpu_failed_fallback_to_cpu"
            else:
                raise
        models[f"q{int(quantile * 100):02d}"] = model

    return {
        "quantiles": list(QUANTILES),
        "models": models,
        "model_type": "catboost_quantile",
        "model_backend": MODEL_BACKEND_CATBOOST,
        "compute_device_requested": compute_device_info["requested"],
        "compute_device_resolved": resolved_device,
        "gpu_probe": compute_device_info["gpu_probe"],
        "model_params": {
            "random_state": random_state,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "gpu_device_id": compute_device_info["gpu_device_id"],
            "compute_device_reason": resolved_reason,
        },
    }


def train_quantile_models(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_backend: str = MODEL_BACKEND_GRADIENT_BOOSTING,
    random_state: int = 1337,
    n_estimators: int = 300,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    compute_device: str = COMPUTE_DEVICE_AUTO,
    gpu_device_id: int = 0,
) -> dict[str, Any]:
    _validate_supervised_arrays(X, y)
    if model_backend not in SUPPORTED_MODEL_BACKENDS:
        raise ValueError(f"unsupported model_backend: {model_backend}")
    compute_device_info = resolve_compute_device(
        model_backend=model_backend,
        compute_device=compute_device,
        gpu_device_id=gpu_device_id,
    )

    if model_backend == MODEL_BACKEND_GRADIENT_BOOSTING:
        return _train_gradient_boosting_quantile_models(
            X,
            y,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            compute_device_info=compute_device_info,
        )

    if model_backend == MODEL_BACKEND_CATBOOST:
        return _train_catboost_quantile_models(
            X,
            y,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            compute_device_info=compute_device_info,
        )

    raise ValueError(f"unsupported model_backend: {model_backend}")


def predict_quantiles(model_bundle: dict[str, Any], X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    models = model_bundle.get("models", {})
    required_model_keys = ["q10", "q50", "q90"]
    missing = [key for key in required_model_keys if key not in models]
    if missing:
        raise ValueError(f"model bundle missing quantile estimators: {missing}")

    q10 = models["q10"].predict(X)
    q50 = models["q50"].predict(X)
    q90 = models["q90"].predict(X)
    stacked = np.column_stack([q10, q50, q90])
    stacked.sort(axis=1)
    return stacked


def save_model_bundle(path: str | Path, model_bundle: dict[str, Any]) -> None:
    bundle_path = Path(path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, bundle_path)


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, dict):
        raise ValueError("invalid model artifact format")
    if "models" not in bundle:
        raise ValueError("model artifact missing models")
    return bundle


def validate_model_bundle_against_dataset_metadata(model_bundle: dict[str, Any], dataset_metadata: dict[str, Any]) -> None:
    model_symbol = str(model_bundle.get("symbol", "")).lower().replace("/", "").replace("-", "")
    dataset_symbol = str(dataset_metadata["symbol"]).lower().replace("/", "").replace("-", "")
    if model_symbol and model_symbol != dataset_symbol:
        raise ValueError("model artifact symbol does not match dataset metadata symbol")
    model_training_symbols = model_bundle.get("training_symbols")
    dataset_training_symbols = dataset_metadata.get("training_symbols")
    if model_training_symbols is not None and dataset_training_symbols is not None:
        normalized_model_training_symbols = sorted(normalize_training_symbols(list(model_training_symbols)))
        normalized_dataset_training_symbols = sorted(normalize_training_symbols(list(dataset_training_symbols)))
        if normalized_model_training_symbols != normalized_dataset_training_symbols:
            raise ValueError("model artifact training_symbols do not match dataset metadata")
    if int(model_bundle.get("encoder_len", dataset_metadata["encoder_len"])) != int(dataset_metadata["encoder_len"]):
        raise ValueError("model artifact encoder_len does not match dataset metadata")
    if int(model_bundle.get("horizon_h", dataset_metadata["horizon_h"])) != int(dataset_metadata["horizon_h"]):
        raise ValueError("model artifact horizon_h does not match dataset metadata")


__all__ = [
    "ForecastResult",
    "MODEL_BACKEND_CATBOOST",
    "MODEL_BACKEND_GRADIENT_BOOSTING",
    "MODEL_ARTIFACT_NAME",
    "COMPUTE_DEVICE_AUTO",
    "COMPUTE_DEVICE_CPU",
    "COMPUTE_DEVICE_GPU",
    "QUANTILES",
    "SUPPORTED_MODEL_BACKENDS",
    "SUPPORTED_COMPUTE_DEVICES",
    "detect_nvidia_gpu",
    "load_model_bundle",
    "predict_quantiles",
    "resolve_compute_device",
    "save_model_bundle",
    "train_quantile_models",
    "validate_model_bundle_against_dataset_metadata",
]
