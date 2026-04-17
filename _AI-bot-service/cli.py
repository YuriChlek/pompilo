from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from module_ai.artifacts import default_artifact_dir


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return _serialize(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    return value


def _format_bool(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "n/a"


def _format_symbol_preview(symbols: list[str], *, limit: int = 8) -> str:
    if not symbols:
        return "n/a"
    if len(symbols) <= limit:
        return ", ".join(symbols)
    return f"{', '.join(symbols[:limit])} ... (+{len(symbols) - limit} more)"


def _load_artifact_bundle(artifact_dir: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    from module_ai.data_pipeline import (
        load_scaler_artifact,
        validate_artifact_metadata,
        validate_scaler_metadata_against_dataset_metadata,
    )
    from module_ai.modeling import (
        MODEL_ARTIFACT_NAME,
        load_model_bundle,
        validate_model_bundle_against_dataset_metadata,
    )
    import joblib

    artifact_path = Path(artifact_dir)
    meta_path = artifact_path / "dataset_meta.pkl"
    if not meta_path.exists():
        raise ValueError(f"artifact metadata not found: {meta_path}")

    dataset_meta = joblib.load(meta_path)
    validate_artifact_metadata(dataset_meta)

    scalers, scaler_meta = load_scaler_artifact(artifact_path / "scalers.pkl")
    validate_scaler_metadata_against_dataset_metadata(scaler_meta, dataset_meta)

    model_bundle = load_model_bundle(artifact_path / MODEL_ARTIFACT_NAME)
    validate_model_bundle_against_dataset_metadata(model_bundle, dataset_meta)

    return artifact_path, dataset_meta, scaler_meta, model_bundle


def _default_artifact_dir(symbol: str) -> str:
    return str(default_artifact_dir(symbol))


def _handle_train_model(args: argparse.Namespace) -> int:
    from module_ai.train import main as train_main

    return int(train_main(args))


def _handle_evaluate_artifact(args: argparse.Namespace) -> int:
    from module_ai.evaluate_run import main as evaluate_main

    if not args.artifact_dir:
        args.artifact_dir = _default_artifact_dir(args.symbol)
    return int(evaluate_main(args))


def _handle_forecast(args: argparse.Namespace) -> int:
    from module_ai.forecast import get_return_forecast

    forecast = get_return_forecast(
        model_dir=args.model_dir or _default_artifact_dir(args.symbol),
        symbol=args.symbol,
        limit=args.limit,
    )
    print(json.dumps(_serialize(forecast), ensure_ascii=False, indent=2))
    return 0


def _handle_start_bot(args: argparse.Namespace) -> int:
    from module_signal.bot import main as bot_main

    if not args.artifact_dir:
        args.artifact_dir = _default_artifact_dir(args.symbol)
    return int(bot_main(args))


def _handle_benchmark_models(args: argparse.Namespace) -> int:
    from module_ai.benchmark_run import main as benchmark_main

    return int(benchmark_main(args))


def _handle_launch_benchmark(args: argparse.Namespace) -> int:
    from module_ai.benchmark_launcher import launch_benchmark

    return int(launch_benchmark(args))


def _handle_benchmark_status(args: argparse.Namespace) -> int:
    from module_ai.benchmark_launcher import benchmark_status

    return int(benchmark_status(args))


def _handle_benchmark_log(args: argparse.Namespace) -> int:
    from module_ai.benchmark_launcher import benchmark_log

    return int(benchmark_log(args))


def _handle_benchmark_stop(args: argparse.Namespace) -> int:
    from module_ai.benchmark_launcher import benchmark_stop

    return int(benchmark_stop(args))


def _handle_calibrate_thresholds(args: argparse.Namespace) -> int:
    from module_ai.signal_calibration import main as calibrate_main

    return int(calibrate_main(args))


def _handle_artifact_info(args: argparse.Namespace) -> int:
    artifact_path, dataset_meta, _, model_bundle = _load_artifact_bundle(args.artifact_dir)
    evaluation_summary = dataset_meta.get("evaluation_summary") or {}
    training_symbols = list(dataset_meta.get("training_symbols", []))
    approval_gate_results = evaluation_summary.get("approval_gate_results") or {}
    validation_failures = approval_gate_results.get("per_symbol_validation_failures") or []
    final_test_failures = approval_gate_results.get("per_symbol_final_test_failures") or []
    file_presence = {
        "dataset_meta.pkl": (artifact_path / "dataset_meta.pkl").exists(),
        "scalers.pkl": (artifact_path / "scalers.pkl").exists(),
        "forecast_model.joblib": (artifact_path / "forecast_model.joblib").exists(),
        "evaluation_report.pkl": (artifact_path / "evaluation_report.pkl").exists(),
    }

    lines = [
        f"artifact_dir: {artifact_path}",
        f"symbol: {dataset_meta['symbol']}",
        f"symbol_count: {dataset_meta.get('symbol_count', 'n/a')}",
        f"training_symbols: {_format_symbol_preview(training_symbols)}",
        f"target_name: {dataset_meta['target_name']}",
        f"horizon_h: {dataset_meta['horizon_h']}",
        f"encoder_len: {dataset_meta['encoder_len']}",
        f"dataset_contract_version: {dataset_meta['dataset_contract_version']}",
        f"feature_version: {dataset_meta['feature_version']}",
        f"evaluation_contract_version: {dataset_meta['evaluation_contract_version']}",
        f"evaluation_status: {dataset_meta.get('evaluation_status', 'n/a')}",
        f"evaluation_approved: {_format_bool(dataset_meta.get('evaluation_approved'))}",
        f"deployable: {_format_bool(dataset_meta.get('deployable'))}",
        f"supported_inference_entrypoint: {dataset_meta['supported_inference_entrypoint']}",
        f"model_type: {model_bundle.get('model_type', 'n/a')}",
        f"model_backend: {model_bundle.get('model_backend', dataset_meta.get('model_backend', 'n/a'))}",
        f"compute_device_requested: {model_bundle.get('compute_device_requested', dataset_meta.get('compute_device', 'n/a'))}",
        f"compute_device_resolved: {model_bundle.get('compute_device_resolved', 'n/a')}",
        f"scaled_features_count: {len(dataset_meta.get('scaled_features', []))}",
        f"scaled_features: {', '.join(dataset_meta.get('scaled_features', []))}",
        f"scaling_scope: {dataset_meta.get('scaling_scope', 'n/a')}",
        f"train_range: {dataset_meta.get('train_start_time')} -> {dataset_meta.get('train_end_time')}",
        f"validation_range: {dataset_meta.get('validation_start_time')} -> {dataset_meta.get('validation_end_time')}",
        f"test_range: {dataset_meta.get('test_start_time')} -> {dataset_meta.get('test_end_time')}",
        f"evaluation_summary_present: {_format_bool(bool(evaluation_summary))}",
        f"per_symbol_validation_failures: {', '.join(validation_failures) if validation_failures else 'none'}",
        f"per_symbol_final_test_failures: {', '.join(final_test_failures) if final_test_failures else 'none'}",
        "files:",
    ]
    for filename, present in file_presence.items():
        lines.append(f"  - {filename}: {'present' if present else 'missing'}")

    print("\n".join(lines))
    return 0


def _handle_validate_artifact(args: argparse.Namespace) -> int:
    try:
        artifact_path, dataset_meta, scaler_meta, model_bundle = _load_artifact_bundle(args.artifact_dir)
        evaluation_summary = dataset_meta.get("evaluation_summary") or {}
        approval_gate_results = evaluation_summary.get("approval_gate_results") or {}
        approval_state = {
            "evaluation_status": dataset_meta.get("evaluation_status"),
            "evaluation_approved": dataset_meta.get("evaluation_approved"),
            "deployable": dataset_meta.get("deployable"),
            "supported_inference_entrypoint": dataset_meta.get("supported_inference_entrypoint"),
            "model_type": model_bundle.get("model_type"),
            "model_backend": model_bundle.get("model_backend", dataset_meta.get("model_backend")),
            "compute_device_requested": model_bundle.get("compute_device_requested", dataset_meta.get("compute_device")),
            "compute_device_resolved": model_bundle.get("compute_device_resolved"),
            "scaled_features_count": len(scaler_meta.get("scaled_features", [])),
            "symbol_count": dataset_meta.get("symbol_count"),
            "scaling_scope": dataset_meta.get("scaling_scope"),
            "global_validation_gate_passed": approval_gate_results.get("global_validation_gate_passed"),
            "global_final_test_gate_passed": approval_gate_results.get("global_final_test_gate_passed"),
            "per_symbol_validation_gate_passed": approval_gate_results.get("per_symbol_validation_gate_passed"),
            "per_symbol_final_test_gate_passed": approval_gate_results.get("per_symbol_final_test_gate_passed"),
        }

        print("VALIDATION RESULT: PASS")
        print(f"artifact_dir: {artifact_path}")
        for key, value in approval_state.items():
            print(f"{key}: {value}")
        return 0
    except Exception as exc:
        print("VALIDATION RESULT: FAIL")
        print(f"reason: {exc}")
        return 1


def _add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--symbol",
        default="ALL",
        help="Торговий символ для single-symbol експерименту або ALL для training universe",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Список символів через кому для multi-symbol train. Перекриває --symbol",
    )
    parser.add_argument("--save-dir", default="./artifacts", help="Базова директорія для артефактів")
    parser.add_argument("--encoder-len", type=int, default=48, help="Довжина encoder-вікна")
    parser.add_argument("--pred-len", type=int, default=6, help="Горизонт H для target_return_h")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Частка train")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Частка validation")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Частка test")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--model-backend",
        default="gradient_boosting",
        choices=["gradient_boosting", "catboost"],
        help="Backend моделі для quantile forecast",
    )
    parser.add_argument("--n-estimators", type=int, default=300, help="Кількість estimators")
    parser.add_argument("--max-depth", type=int, default=3, help="Максимальна глибина дерев")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate моделі")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample для gradient boosting")
    parser.add_argument(
        "--compute-device",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Пристрій навчання: auto, cpu або gpu",
    )
    parser.add_argument("--gpu-device-id", type=int, default=0, help="Індекс NVIDIA GPU для backend-ів з GPU support")


def _add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", default="ALL", help="Торговий символ або ALL для shared artifact evaluation")
    parser.add_argument("--artifact-dir", default="", help="Директорія кандидатного артефакту. За замовчуванням ./artifacts/<symbol>")
    parser.add_argument("--signal-mode", default="quantile_barrier", help="Політика перетворення forecast -> signal")
    parser.add_argument("--cost-threshold", type=float, default=0.0, help="Поріг cost_threshold")
    parser.add_argument("--buy-threshold", type=float, default=0.0, help="Поріг для BUY у median_with_width")
    parser.add_argument("--sell-threshold", type=float, default=0.0, help="Поріг для SELL у median_with_width")
    parser.add_argument("--max-width", type=float, default=0.05, help="Максимальна ширина інтервалу")
    parser.add_argument("--execution-rule", default="close_to_close", help="Правило виконання для trade simulation")
    parser.add_argument("--transaction-cost", type=float, default=0.0, help="Транзакційні витрати")
    parser.add_argument("--slippage", type=float, default=0.0, help="Slippage")
    parser.add_argument("--approval-max-validation-rmse", type=float, default=0.08, help="Максимальний RMSE на validation")
    parser.add_argument("--approval-max-final-test-rmse", type=float, default=0.08, help="Максимальний RMSE на final test")
    parser.add_argument("--approval-min-validation-coverage", type=float, default=0.50, help="Мінімальний coverage на validation")
    parser.add_argument("--approval-min-final-test-coverage", type=float, default=0.50, help="Мінімальний coverage на final test")
    parser.add_argument("--approval-max-per-symbol-validation-rmse", type=float, default=0.12, help="Максимальний RMSE на validation для кожного символу")
    parser.add_argument("--approval-max-per-symbol-final-test-rmse", type=float, default=0.12, help="Максимальний RMSE на final test для кожного символу")
    parser.add_argument("--approval-min-per-symbol-validation-coverage", type=float, default=0.10, help="Мінімальний coverage на validation для кожного символу")
    parser.add_argument("--approval-min-per-symbol-final-test-coverage", type=float, default=0.10, help="Мінімальний coverage на final test для кожного символу")


def _add_common_benchmark_args(parser: argparse.ArgumentParser) -> None:
    _add_common_train_args(parser)
    parser.add_argument("--model-backends", default="gradient_boosting,catboost", help="Список backend-ів через кому для benchmark")
    parser.add_argument("--output-dir", default="./artifacts/benchmarks", help="Куди зберігати benchmark report")
    parser.add_argument(
        "--compare-per-symbol-baseline",
        action="store_true",
        help="Додатково порівняти shared model з окремими per-symbol baseline моделями",
    )
    parser.add_argument("--signal-mode", default="quantile_barrier", help="Політика перетворення forecast -> signal")
    parser.add_argument("--cost-threshold", type=float, default=0.0, help="Поріг cost_threshold")
    parser.add_argument("--buy-threshold", type=float, default=0.0, help="Поріг для BUY у median_with_width")
    parser.add_argument("--sell-threshold", type=float, default=0.0, help="Поріг для SELL у median_with_width")
    parser.add_argument("--max-width", type=float, default=0.05, help="Максимальна ширина інтервалу")
    parser.add_argument("--execution-rule", default="close_to_close", help="Правило виконання для trade simulation")
    parser.add_argument("--transaction-cost", type=float, default=0.0, help="Транзакційні витрати")
    parser.add_argument("--slippage", type=float, default=0.0, help="Slippage")
    parser.add_argument("--approval-max-validation-rmse", type=float, default=0.08, help="Максимальний RMSE на validation")
    parser.add_argument("--approval-max-final-test-rmse", type=float, default=0.08, help="Максимальний RMSE на final test")
    parser.add_argument("--approval-min-validation-coverage", type=float, default=0.50, help="Мінімальний coverage на validation")
    parser.add_argument("--approval-min-final-test-coverage", type=float, default=0.50, help="Мінімальний coverage на final test")
    parser.add_argument("--approval-max-per-symbol-validation-rmse", type=float, default=0.12, help="Максимальний RMSE на validation для кожного символу")
    parser.add_argument("--approval-max-per-symbol-final-test-rmse", type=float, default=0.12, help="Максимальний RMSE на final test для кожного символу")
    parser.add_argument("--approval-min-per-symbol-validation-coverage", type=float, default=0.10, help="Мінімальний coverage на validation для кожного символу")
    parser.add_argument("--approval-min-per-symbol-final-test-coverage", type=float, default=0.10, help="Мінімальний coverage на final test для кожного символу")


def _add_common_calibration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", default="ALL", help="Торговий символ або ALL для shared artifact calibration")
    parser.add_argument("--artifact-dir", default="", help="Директорія candidate/shared artifact")
    parser.add_argument("--signal-modes", default="quantile_barrier,median_with_width", help="Список signal policy через кому")
    parser.add_argument("--cost-threshold-grid", default="", help="Grid для quantile_barrier cost_threshold")
    parser.add_argument("--buy-threshold-grid", default="", help="Grid для median_with_width buy_threshold")
    parser.add_argument("--sell-threshold-grid", default="", help="Grid для median_with_width sell_threshold")
    parser.add_argument("--max-width-grid", default="", help="Grid для median_with_width max_width")
    parser.add_argument("--top-k", type=int, default=12, help="Скільки найкращих candidate-ів зберегти")
    parser.add_argument("--min-signal-share", type=float, default=0.002, help="Мінімальна частка BUY+SELL")
    parser.add_argument("--max-signal-share", type=float, default=0.35, help="Максимальна частка BUY+SELL")
    parser.add_argument("--target-signal-share", type=float, default=0.05, help="Цільова частка BUY+SELL для scoring")
    parser.add_argument("--min-trade-count", type=int, default=25, help="Мінімальна кількість trades")
    parser.add_argument("--execution-rule", default="close_to_close", help="Правило виконання для trade simulation")
    parser.add_argument("--transaction-cost", type=float, default=0.0, help="Транзакційні витрати")
    parser.add_argument("--slippage", type=float, default=0.0, help="Slippage")
    parser.add_argument("--output-path", default="", help="Куди зберігати calibration report")


def _add_benchmark_management_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default="./artifacts/benchmarks/real_db_run", help="Каталог benchmark run")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Єдиний CLI для активного ML/strategy workflow проєкту."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train-model",
        aliases=["train"],
        help="Навчити активну ML-модель і створити кандидатний артефакт.",
        description="Запускає активне навчання та створює кандидатний артефакт. Артефакт не стає deployable автоматично.",
    )
    _add_common_train_args(train_parser)
    train_parser.set_defaults(handler=_handle_train_model)

    eval_parser = subparsers.add_parser(
        "evaluate-artifact",
        aliases=["eval"],
        help="Перевірити кандидатний артефакт і прийняти рішення approve/reject.",
        description="Запускає канонічний approval gate для вже збереженого кандидатного артефакту.",
    )
    _add_common_eval_args(eval_parser)
    eval_parser.set_defaults(handler=_handle_evaluate_artifact)

    benchmark_parser = subparsers.add_parser(
        "benchmark-models",
        aliases=["benchmark"],
        help="Порівняти кілька model backend-ів на одному shared split.",
        description="Запускає benchmark кількох backend-ів і зберігає recommendation report.",
    )
    _add_common_benchmark_args(benchmark_parser)
    benchmark_parser.set_defaults(handler=_handle_benchmark_models)

    launch_benchmark_parser = subparsers.add_parser(
        "launch-benchmark",
        help="Запустити benchmark у background через Python launcher.",
        description="Запускає benchmark у background через Python launcher і одразу повертає prompt.",
    )
    _add_common_benchmark_args(launch_benchmark_parser)
    launch_benchmark_parser.add_argument("--startup-wait-seconds", type=float, default=3.0, help="Скільки чекати після старту, щоб перевірити що процес не впав")
    launch_benchmark_parser.set_defaults(handler=_handle_launch_benchmark)

    benchmark_status_parser = subparsers.add_parser(
        "benchmark-status",
        help="Показати статус background benchmark run.",
    )
    _add_benchmark_management_args(benchmark_status_parser)
    benchmark_status_parser.set_defaults(handler=_handle_benchmark_status)

    benchmark_log_parser = subparsers.add_parser(
        "benchmark-log",
        help="Показати останні рядки benchmark log через Python CLI.",
    )
    _add_benchmark_management_args(benchmark_log_parser)
    benchmark_log_parser.add_argument("--lines", type=int, default=80, help="Скільки останніх рядків показати")
    benchmark_log_parser.set_defaults(handler=_handle_benchmark_log)

    benchmark_stop_parser = subparsers.add_parser(
        "benchmark-stop",
        help="Зупинити background benchmark run.",
    )
    _add_benchmark_management_args(benchmark_stop_parser)
    benchmark_stop_parser.add_argument("--force", action="store_true", help="Надіслати SIGKILL замість SIGTERM")
    benchmark_stop_parser.set_defaults(handler=_handle_benchmark_stop)

    calibrate_parser = subparsers.add_parser(
        "calibrate-thresholds",
        aliases=["calibrate"],
        help="Підібрати forecast->signal thresholds на validation і підтвердити їх на final test.",
        description="Запускає signal policy calibration поверх уже навченого artifact без зміни model backend.",
    )
    _add_common_calibration_args(calibrate_parser)
    calibrate_parser.set_defaults(handler=_handle_calibrate_thresholds)

    forecast_parser = subparsers.add_parser(
        "forecast",
        help="Побудувати forecast з approved-артефакту.",
        description="Запускає активний forecast-only inference. Працює лише з approved/deployable артефактами.",
    )
    forecast_parser.add_argument("--model-dir", default="", help="Директорія approved-артефакту. За замовчуванням ./artifacts/<symbol>")
    forecast_parser.add_argument("--symbol", required=True, help="Торговий символ")
    forecast_parser.add_argument("--limit", type=int, default=256, help="Скільки останніх свічок завантажити")
    forecast_parser.set_defaults(handler=_handle_forecast)

    artifact_info_parser = subparsers.add_parser(
        "artifact-info",
        aliases=["info"],
        help="Показати практичну інформацію про артефакт.",
        description="Читає артефакт і друкує операторську зведену інформацію про метадані, версії та наявність файлів.",
    )
    artifact_info_parser.add_argument("--artifact-dir", required=True, help="Директорія артефакту")
    artifact_info_parser.set_defaults(handler=_handle_artifact_info)

    validate_artifact_parser = subparsers.add_parser(
        "validate-artifact",
        aliases=["validate"],
        help="Перевірити артефакт на внутрішню узгодженість.",
        description="Валідує dataset metadata, scaler artifact, model artifact та їхню взаємну узгодженість для активного workflow.",
    )
    validate_artifact_parser.add_argument("--artifact-dir", required=True, help="Директорія артефакту")
    validate_artifact_parser.set_defaults(handler=_handle_validate_artifact)

    bot_parser = subparsers.add_parser(
        "start-bot",
        aliases=["bot"],
        help="Запустити активний strategy/bot path.",
        description="Запускає strategy-layer decision flow. ML всередині дає forecast, фінальне рішення формується на рівні strategy.",
    )
    bot_parser.add_argument("--symbol", required=True, help="Торговий символ")
    bot_parser.add_argument("--artifact-dir", default="", help="Явна директорія approved-артефакту. Якщо не задано, використовується ./artifacts/<symbol>")
    bot_parser.add_argument("--limit", type=int, default=256, help="Скільки останніх свічок завантажувати для forecast")
    bot_parser.add_argument("--policy-mode", default="quantile_barrier", help="Політика forecast -> ML signal")
    bot_parser.add_argument("--cost-threshold", type=float, default=0.0, help="Поріг для quantile_barrier")
    bot_parser.add_argument("--buy-threshold", type=float, default=0.0, help="Поріг BUY для median_with_width")
    bot_parser.add_argument("--sell-threshold", type=float, default=0.0, help="Поріг SELL для median_with_width")
    bot_parser.add_argument("--max-width", type=float, default=0.05, help="Максимальна ширина інтервалу")
    bot_parser.add_argument("--use-advanced-heuristics", action="store_true", help="Використовувати advanced order-flow heuristics")
    bot_parser.add_argument("--decision-details", action="store_true", help="Повернути розширені деталі strategy decision")
    bot_parser.add_argument("--interval-seconds", type=float, default=0.0, help="Якщо > 0, повторювати запуск циклічно")
    bot_parser.add_argument("--max-iterations", type=int, default=None, help="Обмеження кількості циклів у loop-режимі")
    bot_parser.set_defaults(handler=_handle_start_bot)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    exit_code = args.handler(args)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
