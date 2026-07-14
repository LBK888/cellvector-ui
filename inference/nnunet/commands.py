from __future__ import annotations

from pathlib import Path

from .contracts import ExperimentSpec, InferenceProfile


def build_plan_command(spec: ExperimentSpec) -> list[str]:
    command = [
        "nnUNetv2_plan_and_preprocess",
        "-d",
        str(spec.dataset_id),
        "-c",
        spec.configuration,
    ]
    if spec.planner:
        command.extend(["-pl", spec.planner])
    command.append("--verify_dataset_integrity")
    return command


def build_train_command(spec: ExperimentSpec, *, fold: int) -> list[str]:
    if fold not in spec.folds:
        raise ValueError(f"fold {fold} is outside experiment folds {spec.folds}")
    command = [
        "nnUNetv2_train",
        str(spec.dataset_id),
        spec.configuration,
        str(fold),
    ]
    if spec.plans:
        command.extend(["-p", spec.plans])
    command.append("--npz")
    return command


def build_predict_command(
    spec: ExperimentSpec,
    *,
    input_dir: Path,
    output_dir: Path,
    profile: InferenceProfile,
) -> list[str]:
    command = [
        "nnUNetv2_predict",
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-d",
        str(spec.dataset_id),
        "-c",
        spec.configuration,
    ]
    if spec.plans:
        command.extend(["-p", spec.plans])
    if profile == InferenceProfile.PREVIEW:
        command.extend(["-f", str(spec.folds[0])])
    return command

