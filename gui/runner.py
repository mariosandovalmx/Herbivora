"""Execution of subprocesses with queued logs (worker thread)."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

from gui.paths import REPO_ROOT


class JobStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobResult:
    status: JobStatus
    returncode: int = 0
    message: str = ""


class JobRunner:
    """Runs a Python command in the background and emits log lines."""

    def __init__(self, on_log: Callable[[str], None], on_done: Callable[[JobResult], None]) -> None:
        self._on_log = on_log
        self._on_done = on_done
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._status = JobStatus.IDLE
        self._lock = threading.Lock()
        self._cancel_requested = False

    @property
    def status(self) -> JobStatus:
        with self._lock:
            return self._status

    @property
    def is_running(self) -> bool:
        return self.status == JobStatus.RUNNING

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._on_log(f"[{ts}] {msg}")

    def run_script(
        self,
        script: Path | str,
        args: list[str],
        *,
        title: str = "",
        cwd: Path | None = None,
    ) -> None:
        if self.is_running:
            self.log("A job is already running.")
            return
        work_cwd = cwd or REPO_ROOT
        cmd = [sys.executable, str(script), *args]
        with self._lock:
            self._status = JobStatus.RUNNING
        if title:
            self.log(f"{'=' * 50}")
            self.log(title)
            self.log(f"{'=' * 50}")
        self.log(f"Command: {' '.join(cmd)}")
        self.log(f"CWD: {work_cwd}")

        def worker() -> None:
            result = self._run_process(cmd, work_cwd)
            with self._lock:
                self._status = result.status
            self._on_done(result)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def run_callable(
        self,
        fn: Callable[[Callable[[str], None], Callable[[], bool]], None],
        *,
        title: str = "",
    ) -> None:
        """Run an in-process callable on a worker thread (models stay in memory).

        ``fn(log_fn, should_cancel)`` should perform the work. Cancellation is
        cooperative via ``should_cancel()``.
        """
        if self.is_running:
            self.log("A job is already running.")
            return
        self._cancel_requested = False
        with self._lock:
            self._status = JobStatus.RUNNING
        if title:
            self.log(f"{'=' * 50}")
            self.log(title)
            self.log(f"{'=' * 50}")

        def worker() -> None:
            try:
                fn(self.log, lambda: self._cancel_requested)
                if self._cancel_requested:
                    result = JobResult(JobStatus.CANCELLED, message="Cancelled by the user")
                else:
                    self.log("Completed successfully.")
                    result = JobResult(JobStatus.SUCCESS)
            except Exception as e:
                self.log(f"Exception: {e}")
                result = JobResult(JobStatus.FAILED, -1, str(e))
            with self._lock:
                self._status = result.status
            self._on_done(result)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def run_sequence(
        self,
        steps: list[tuple[str, Path | str, list[str]]],
        *,
        cwd: Path | None = None,
    ) -> None:
        """Runs multiple scripts sequentially (e.g. full pipeline)."""
        if self.is_running:
            self.log("A job is already running.")
            return
        work_cwd = cwd or REPO_ROOT

        def worker() -> None:
            final = JobResult(JobStatus.SUCCESS)
            for title, script, args in steps:
                if self._cancel_requested:
                    final = JobResult(JobStatus.CANCELLED, message="Cancelled by the user")
                    break
                self.log(f"{'=' * 50}")
                self.log(title)
                cmd = [sys.executable, str(script), *args]
                step_result = self._run_process(cmd, work_cwd)
                if step_result.status != JobStatus.SUCCESS:
                    final = step_result
                    break
            with self._lock:
                self._status = final.status
            self._on_done(final)

        self._cancel_requested = False
        with self._lock:
            self._status = JobStatus.RUNNING
        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def _run_process(self, cmd: list[str], cwd: Path) -> JobResult:
        self._cancel_requested = False
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env={**os.environ, "PYTHONUTF8": "1"},
            )
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                if self._cancel_requested:
                    self._proc.terminate()
                    break
                self._on_log(line.rstrip("\n\r"))
            code = self._proc.wait()
            self._proc = None
            if self._cancel_requested:
                return JobResult(JobStatus.CANCELLED, code, "Process cancelled")
            if code == 0:
                self.log("Completed successfully.")
                return JobResult(JobStatus.SUCCESS, code)
            self.log(f"Error: exit code {code}")
            return JobResult(JobStatus.FAILED, code, f"Exit code {code}")
        except FileNotFoundError as e:
            self.log(f"Error: {e}")
            return JobResult(JobStatus.FAILED, -1, str(e))
        except Exception as e:
            self.log(f"Exception: {e}")
            return JobResult(JobStatus.FAILED, -1, str(e))

    def cancel(self) -> None:
        self._cancel_requested = True
        with self._lock:
            proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except OSError:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except OSError:
                    pass
