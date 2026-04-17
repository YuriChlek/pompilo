"""
Active ML package surface.

Canonical active modules:
- module_ai.data_access
- module_ai.data_pipeline
- module_ai.windowing
- module_ai.modeling
- module_ai.runtime
- module_ai.splits
- module_ai.train
- module_ai.forecast
- module_ai.evaluate
- module_ai.evaluate_run
- module_ai.benchmark_launcher
- module_ai.signal_calibration

Important:
- The package does not re-export legacy decision-returning inference helpers.
- Legacy files remain in the repository for reference only and are not part of
  the active public workflow.
"""

from .data_pipeline import (
    DATASET_CONTRACT_VERSION,
    DECODER_KNOWN_FEATURES,
    DEPRECATED_FEATURES,
    ENCODER_FEATURES,
    EVALUATION_CONTRACT_VERSION,
    FEATURE_VERSION,
    SUPPORTED_INFERENCE_ENTRYPOINT,
    TARGET_NAME,
)
from .forecast import get_return_forecast
from .evaluate import (
    evaluate_forecasts,
    forecast_to_signal,
    run_final_test_evaluation,
    run_walk_forward_evaluation,
    simulate_trades,
)
from .modeling import ForecastResult
from .benchmark_launcher import benchmark_log, benchmark_status, benchmark_stop, launch_benchmark
from .signal_calibration import main as calibrate_thresholds_main
from .splits import SplitConfig, generate_walk_forward_folds, split_dataset
from .training_universe import TrainingUniverseBundle, load_training_universe_bundle, summarize_training_universe

__all__ = [
    "DATASET_CONTRACT_VERSION",
    "DECODER_KNOWN_FEATURES",
    "DEPRECATED_FEATURES",
    "ENCODER_FEATURES",
    "EVALUATION_CONTRACT_VERSION",
    "FEATURE_VERSION",
    "ForecastResult",
    "SUPPORTED_INFERENCE_ENTRYPOINT",
    "TARGET_NAME",
    "TrainingUniverseBundle",
    "SplitConfig",
    "benchmark_log",
    "benchmark_status",
    "benchmark_stop",
    "launch_benchmark",
    "calibrate_thresholds_main",
    "evaluate_forecasts",
    "forecast_to_signal",
    "generate_walk_forward_folds",
    "get_return_forecast",
    "load_training_universe_bundle",
    "run_final_test_evaluation",
    "run_walk_forward_evaluation",
    "simulate_trades",
    "split_dataset",
    "summarize_training_universe",
]
