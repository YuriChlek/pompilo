from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


LOG_FILENAME = "benchmark.log"
PID_FILENAME = "benchmark.pid"
REPORT_FILENAME = "model_benchmark_report.json"
STATE_FILENAME = "launcher_state.json"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _output_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "output_dir": root,
        "log_path": root / LOG_FILENAME,
        "pid_path": root / PID_FILENAME,
        "report_path": root / REPORT_FILENAME,
        "state_path": root / STATE_FILENAME,
    }


def _is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _load_pid(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    raw = pid_path.read_text().strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _build_benchmark_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "cli.py",
        "benchmark-models",
        "--symbols",
        str(args.symbols),
        "--model-backends",
        str(args.model_backends),
        "--output-dir",
        str(args.output_dir),
        "--encoder-len",
        str(args.encoder_len),
        "--pred-len",
        str(args.pred_len),
        "--train-ratio",
        str(args.train_ratio),
        "--val-ratio",
        str(args.val_ratio),
        "--test-ratio",
        str(args.test_ratio),
        "--seed",
        str(args.seed),
        "--n-estimators",
        str(args.n_estimators),
        "--max-depth",
        str(args.max_depth),
        "--learning-rate",
        str(args.learning_rate),
        "--subsample",
        str(args.subsample),
        "--compute-device",
        str(args.compute_device),
        "--gpu-device-id",
        str(args.gpu_device_id),
        "--signal-mode",
        str(args.signal_mode),
        "--cost-threshold",
        str(args.cost_threshold),
        "--buy-threshold",
        str(args.buy_threshold),
        "--sell-threshold",
        str(args.sell_threshold),
        "--max-width",
        str(args.max_width),
        "--execution-rule",
        str(args.execution_rule),
        "--transaction-cost",
        str(args.transaction_cost),
        "--slippage",
        str(args.slippage),
        "--approval-max-validation-rmse",
        str(args.approval_max_validation_rmse),
        "--approval-max-final-test-rmse",
        str(args.approval_max_final_test_rmse),
        "--approval-min-validation-coverage",
        str(args.approval_min_validation_coverage),
        "--approval-min-final-test-coverage",
        str(args.approval_min_final_test_coverage),
        "--approval-max-per-symbol-validation-rmse",
        str(args.approval_max_per_symbol_validation_rmse),
        "--approval-max-per-symbol-final-test-rmse",
        str(args.approval_max_per_symbol_final_test_rmse),
        "--approval-min-per-symbol-validation-coverage",
        str(args.approval_min_per_symbol_validation_coverage),
        "--approval-min-per-symbol-final-test-coverage",
        str(args.approval_min_per_symbol_final_test_coverage),
    ]
    if getattr(args, "compare_per_symbol_baseline", False):
        cmd.append("--compare-per-symbol-baseline")
    return cmd


def launch_benchmark(args: argparse.Namespace) -> int:
    paths = _output_paths(args.output_dir)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    existing_pid = _load_pid(paths["pid_path"])
    if existing_pid is not None and _is_process_running(existing_pid):
        print("Benchmark is already running for this output directory.")
        print(f"PID: {existing_pid}")
        print(f"Log: {paths['log_path']}")
        print(f"State: {paths['state_path']}")
        print(f"Check status: python cli.py benchmark-status --output-dir {paths['output_dir']}")
        print(f"Stop process: python cli.py benchmark-stop --output-dir {paths['output_dir']}")
        return 1

    if paths["report_path"].exists():
        paths["report_path"].unlink()

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_json(
        paths["state_path"],
        {
            "status": "launching",
            "started_at": started_at,
            "symbols": args.symbols,
            "model_backends": args.model_backends,
            "compare_per_symbol_baseline": bool(args.compare_per_symbol_baseline),
            "compute_device": args.compute_device,
            "gpu_device_id": int(args.gpu_device_id),
            "output_dir": str(paths["output_dir"]),
            "log_path": str(paths["log_path"]),
            "pid_path": str(paths["pid_path"]),
            "report_path": str(paths["report_path"]),
        },
    )

    command = _build_benchmark_command(args)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with paths["log_path"].open("w", encoding="utf-8") as log_file:
        log_file.write("=== benchmark launcher ===\n")
        log_file.write(f"started_at={started_at}\n")
        log_file.write(f"symbols={args.symbols}\n")
        log_file.write(f"model_backends={args.model_backends}\n")
        log_file.write(f"compare_per_symbol_baseline={bool(args.compare_per_symbol_baseline)}\n")
        log_file.write(f"compute_device={args.compute_device}\n")
        log_file.write(f"gpu_device_id={int(args.gpu_device_id)}\n")
        log_file.write(f"output_dir={paths['output_dir']}\n")
        log_file.write(f"command={' '.join(command)}\n\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=_project_root(),
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    paths["pid_path"].write_text(str(process.pid))
    time.sleep(float(args.startup_wait_seconds))

    if process.poll() is not None:
        _write_json(
            paths["state_path"],
            {
                "status": "failed_to_start",
                "started_at": started_at,
                "pid": process.pid,
                "output_dir": str(paths["output_dir"]),
                "log_path": str(paths["log_path"]),
                "pid_path": str(paths["pid_path"]),
                "report_path": str(paths["report_path"]),
            },
        )
        if paths["pid_path"].exists():
            paths["pid_path"].unlink()
        print("Benchmark process exited right after startup.")
        print(f"Log: {paths['log_path']}")
        return 1

    _write_json(
        paths["state_path"],
        {
            "status": "running",
            "started_at": started_at,
            "pid": process.pid,
            "symbols": args.symbols,
            "model_backends": args.model_backends,
            "compare_per_symbol_baseline": bool(args.compare_per_symbol_baseline),
            "compute_device": args.compute_device,
            "gpu_device_id": int(args.gpu_device_id),
            "output_dir": str(paths["output_dir"]),
            "log_path": str(paths["log_path"]),
            "pid_path": str(paths["pid_path"]),
            "report_path": str(paths["report_path"]),
        },
    )

    print("Benchmark started in background.")
    print(f"PID: {process.pid}")
    print(f"Log: {paths['log_path']}")
    print(f"PID file: {paths['pid_path']}")
    print(f"Expected report: {paths['report_path']}")
    print(f"Launcher state: {paths['state_path']}")
    print()
    print(f"Check status: python cli.py benchmark-status --output-dir {paths['output_dir']}")
    print(f"Show recent log lines: python cli.py benchmark-log --output-dir {paths['output_dir']}")
    print(f"Stop process: python cli.py benchmark-stop --output-dir {paths['output_dir']}")
    return 0


def benchmark_status(args: argparse.Namespace) -> int:
    paths = _output_paths(args.output_dir)
    state = _load_json(paths["state_path"]) or {}
    report = _load_json(paths["report_path"]) or {}
    pid = _load_pid(paths["pid_path"])
    running = _is_process_running(pid) if pid is not None else False

    print(f"output_dir: {paths['output_dir']}")
    print(f"pid: {pid if pid is not None else 'n/a'}")
    print(f"process_running: {running}")
    print(f"launcher_status: {state.get('status', 'n/a')}")
    print(f"log_path: {paths['log_path']}")
    print(f"report_path: {paths['report_path']}")
    print(f"report_exists: {paths['report_path'].exists()}")
    if report:
        print(f"report_status: {report.get('status', 'n/a')}")
        print(f"benchmarked_backends: {report.get('benchmarked_backends', [])}")
        print(f"backend_results_count: {len(report.get('backend_results', []))}")
        print(f"per_symbol_baseline_results_count: {len(report.get('per_symbol_baseline_results', []))}")
        print(f"recommendation: {report.get('recommendation')}")
    else:
        print("report_status: not_ready")
    return 0


def benchmark_log(args: argparse.Namespace) -> int:
    paths = _output_paths(args.output_dir)
    if not paths["log_path"].exists():
        print(f"log not found: {paths['log_path']}")
        return 1
    lines = paths["log_path"].read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-max(int(args.lines), 1) :]
    print("\n".join(tail))
    return 0


def benchmark_stop(args: argparse.Namespace) -> int:
    paths = _output_paths(args.output_dir)
    pid = _load_pid(paths["pid_path"])
    if pid is None:
        print("No benchmark PID file found.")
        return 1
    if not _is_process_running(pid):
        print("Benchmark process is not running.")
        if paths["pid_path"].exists():
            paths["pid_path"].unlink()
        return 1

    sig = signal.SIGKILL if args.force else signal.SIGTERM
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        os.kill(pid, sig)

    _write_json(
        paths["state_path"],
        {
            "status": "stop_requested",
            "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "pid": pid,
            "signal": sig.name,
            "output_dir": str(paths["output_dir"]),
            "log_path": str(paths["log_path"]),
            "pid_path": str(paths["pid_path"]),
            "report_path": str(paths["report_path"]),
        },
    )
    print(f"Stop signal sent: {sig.name}")
    print(f"PID: {pid}")
    print(f"Output dir: {paths['output_dir']}")
    return 0
