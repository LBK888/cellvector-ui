from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import os
from pathlib import Path
import subprocess
from threading import Event

from .contracts import RunRecord, RunState


class NnUNetRunner:
    """Execute one allow-listed argv contract without invoking a shell."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | Path,
        env: Mapping[str, str] | None = None,
        cancel_event: Event | None = None,
        expected_artifacts: Mapping[Path, str] | None = None,
    ) -> RunRecord:
        command = tuple(str(value) for value in argv)
        if not command:
            raise ValueError("argv must not be empty")
        if cancel_event is not None and cancel_event.is_set():
            return RunRecord(argv=command, state=RunState.CANCELLED)
        workdir = Path(cwd).resolve()
        process_env = os.environ.copy()
        process_env.update(dict(env or {}))
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        )
        try:
            process = subprocess.Popen(
                list(command),
                cwd=workdir,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
                creationflags=creationflags,
            )
        except FileNotFoundError as error:
            return RunRecord(
                argv=command,
                state=RunState.FAILED,
                error_code="NNUNET_NOT_INSTALLED",
                stderr=str(error),
            )

        cancelled = False
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    self._terminate_tree(process)
                    stdout, stderr = process.communicate()
                    break
        if cancelled:
            return RunRecord(
                argv=command,
                state=RunState.CANCELLED,
                return_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        if process.returncode != 0:
            return RunRecord(
                argv=command,
                state=RunState.FAILED,
                return_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                error_code=self._map_error(stderr),
            )

        artifact_hashes: dict[str, str] = {}
        for relative_path, expected_hash in (expected_artifacts or {}).items():
            artifact = (workdir / relative_path).resolve()
            try:
                display_path = artifact.relative_to(workdir).as_posix()
            except ValueError:
                return RunRecord(
                    argv=command,
                    state=RunState.FAILED,
                    return_code=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    error_code="ARTIFACT_PATH_INVALID",
                )
            if not artifact.is_file():
                return RunRecord(
                    argv=command,
                    state=RunState.FAILED,
                    return_code=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    error_code="ARTIFACT_MISSING",
                )
            actual_hash = sha256(artifact.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                return RunRecord(
                    argv=command,
                    state=RunState.FAILED,
                    return_code=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    error_code="ARTIFACT_CHECKSUM_MISMATCH",
                )
            artifact_hashes[display_path] = actual_hash
        return RunRecord(
            argv=command,
            state=RunState.SUCCEEDED,
            return_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            artifact_sha256=artifact_hashes,
        )

    @staticmethod
    def _map_error(stderr: str) -> str:
        lowered = stderr.lower()
        if "cuda out of memory" in lowered or "cublas_status_alloc_failed" in lowered:
            return "GPU_OUT_OF_MEMORY"
        if "no space left on device" in lowered:
            return "INSUFFICIENT_STORAGE"
        return "TRAINING_FAILED"

    @staticmethod
    def _terminate_tree(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
            )
        else:
            process.terminate()

