from __future__ import annotations

from collections.abc import Callable
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import platform
from pathlib import Path
import shutil
import sys
from typing import Any

from pydantic import Field

from .contracts import ContractModel


class AIEnvironmentReport(ContractModel):
    ready: bool
    python_version: str
    torch_version: str | None = None
    nnunet_version: str | None = None
    cuda_available: bool = False
    gpu_name: str | None = None
    errors: tuple[str, ...] = Field(default_factory=tuple)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def check_ai_environment(
    *,
    command_finder: Callable[[str], str | None] = shutil.which,
    module_finder: Callable[[str], Any] = importlib.util.find_spec,
    executable_path: str | Path | None = None,
) -> AIEnvironmentReport:
    errors: list[str] = []
    torch_version = None
    nnunet_version = None
    cuda_available = False
    gpu_name = None
    if module_finder("torch") is None:
        errors.append("PYTORCH_NOT_INSTALLED")
    else:
        torch_version = _package_version("torch")
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = str(torch.cuda.get_device_name(0))
        else:
            errors.append("CUDA_UNAVAILABLE")
    if module_finder("nnunetv2") is None:
        errors.append("NNUNET_NOT_INSTALLED")
    else:
        nnunet_version = _package_version("nnunetv2")
    required = (
        "nnUNetv2_plan_and_preprocess",
        "nnUNetv2_train",
        "nnUNetv2_predict",
    )
    scripts_dir = Path(executable_path or sys.executable).resolve().parent

    def command_exists(command: str) -> bool:
        return bool(
            command_finder(command)
            or (scripts_dir / command).is_file()
            or (scripts_dir / f"{command}.exe").is_file()
        )

    if any(not command_exists(command) for command in required):
        if "NNUNET_NOT_INSTALLED" not in errors:
            errors.append("NNUNET_COMMANDS_MISSING")
    return AIEnvironmentReport(
        ready=not errors,
        python_version=platform.python_version(),
        torch_version=torch_version,
        nnunet_version=nnunet_version,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        errors=tuple(errors),
    )
