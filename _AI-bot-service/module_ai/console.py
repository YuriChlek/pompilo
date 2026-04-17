from __future__ import annotations

import itertools
import sys
import threading
import time
from contextlib import contextmanager


class ConsoleSpinner:
    def __init__(self, message: str):
        self.message = message
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled = sys.stdout.isatty()

    def start(self) -> None:
        if not self._enabled:
            print(f"[...] {self.message}", flush=True)
            return

        def _run() -> None:
            for frame in itertools.cycle("|/-\\"):
                if self._stop_event.is_set():
                    break
                sys.stdout.write(f"\r[{frame}] {self.message}")
                sys.stdout.flush()
                time.sleep(0.1)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self, *, success: bool, elapsed_s: float) -> None:
        status = "OK" if success else "FAIL"
        if not self._enabled:
            print(f"[{status}] {self.message} ({elapsed_s:.1f}s)", flush=True)
            return

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        sys.stdout.write(f"\r[{status}] {self.message} ({elapsed_s:.1f}s)\n")
        sys.stdout.flush()


@contextmanager
def stage(message: str):
    spinner = ConsoleSpinner(message)
    started_at = time.perf_counter()
    spinner.start()
    try:
        yield
    except Exception:
        spinner.stop(success=False, elapsed_s=time.perf_counter() - started_at)
        raise
    spinner.stop(success=True, elapsed_s=time.perf_counter() - started_at)


class ProgressBar:
    def __init__(self, total: int, label: str):
        self.total = max(int(total), 1)
        self.label = label
        self.current = 0
        self._enabled = sys.stdout.isatty()

    def advance(self, item_label: str) -> None:
        self.current += 1
        if not self._enabled:
            print(f"[progress] {self.label}: {self.current}/{self.total} -> {item_label}", flush=True)
            return

        width = 28
        ratio = min(max(self.current / self.total, 0.0), 1.0)
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        sys.stdout.write(f"\r[{bar}] {self.label}: {self.current}/{self.total} -> {item_label}")
        sys.stdout.flush()
        if self.current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()


def log_step(message: str) -> None:
    print(message, flush=True)


__all__ = [
    "ConsoleSpinner",
    "ProgressBar",
    "log_step",
    "stage",
]
