from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Mapping
from uuid import UUID

import tifffile

from cellvector.domain.models import AnnotationDocument
from cellvector.io.images import import_frames

from .models import DatasetError, DatasetSnapshot, SplitName
from .rasterize import rasterize_nnunet_labels


@dataclass(frozen=True)
class NnUNetExportReport:
    dataset_root: Path
    dataset_json: dict[str, object]
    artifact_sha256: dict[str, str]


def export_nnunet_dataset(
    snapshot: DatasetSnapshot,
    documents: Mapping[UUID, AnnotationDocument],
    output_root: str | Path,
    *,
    dataset_id: int,
    dataset_name: str,
    membrane_width_px: int = 1,
    microridge_width_px: int = 1,
) -> NnUNetExportReport:
    if not 1 <= dataset_id <= 999:
        raise DatasetError("INVALID_DATASET_ID", "dataset ID must be between 1 and 999")
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", dataset_name) is None:
        raise DatasetError("INVALID_DATASET_NAME", "dataset name is not nnU-Net safe")
    dataset_root = Path(output_root) / f"Dataset{dataset_id:03d}_{dataset_name}"
    for folder in ("imagesTr", "labelsTr", "imagesTs"):
        (dataset_root / folder).mkdir(parents=True, exist_ok=True)

    training_cases = 0
    for sample in snapshot.samples:
        if sample.split is None:
            raise DatasetError("SPLIT_MISSING", f"sample {sample.sample_id} has no split")
        document = documents.get(sample.revision_id)
        if document is None:
            raise DatasetError(
                "ANNOTATION_REVISION_MISSING",
                f"revision {sample.revision_id} was not supplied",
            )
        if document.source.sha256 != sample.source_sha256:
            raise DatasetError("CHECKSUM_MISMATCH", f"sample {sample.sample_id} source differs")
        frames = import_frames(sample.image_uri)
        if sample.frame_index >= len(frames):
            raise DatasetError("FRAME_INDEX_INVALID", f"sample {sample.sample_id} frame is absent")
        frame = frames[sample.frame_index]
        if frame.array.shape != (sample.height_px, sample.width_px):
            raise DatasetError("SOURCE_DIMENSION_MISMATCH", sample.sample_id)
        if sample.split == SplitName.FROZEN_TEST:
            image_path = dataset_root / "imagesTs" / f"{sample.sample_id}_0000.tif"
            tifffile.imwrite(image_path, frame.array)
            continue
        training_cases += 1
        image_path = dataset_root / "imagesTr" / f"{sample.sample_id}_0000.tif"
        label_path = dataset_root / "labelsTr" / f"{sample.sample_id}.tif"
        tifffile.imwrite(image_path, frame.array)
        labels = rasterize_nnunet_labels(
            document,
            membrane_width_px=membrane_width_px,
            microridge_width_px=microridge_width_px,
        )
        tifffile.imwrite(label_path, labels)

    dataset_json: dict[str, object] = {
        "channel_names": {"0": "actin_microridge"},
        "labels": {
            "background": 0,
            "cell_region": 1,
            "cell_membrane": 2,
            "microridge": 3,
        },
        "numTraining": training_cases,
        "file_ending": ".tif",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    dataset_json_path = dataset_root / "dataset.json"
    dataset_json_path.write_text(
        json.dumps(dataset_json, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifact_hashes = {
        path.relative_to(dataset_root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(dataset_root.rglob("*"))
        if path.is_file()
    }
    dataset_report_path = dataset_root / "dataset-report.json"
    dataset_report_path.write_text(
        json.dumps(
            {
                "snapshot_hash": snapshot.identity_hash(),
                "snapshot_schema_version": snapshot.schema_version,
                "software_smoke_test": snapshot.software_smoke_test,
                "label_policy": {
                    "version": snapshot.label_policy_version,
                    "membrane_width_px": membrane_width_px,
                    "microridge_width_px": microridge_width_px,
                    "priority": [
                        "microridge",
                        "cell_membrane",
                        "cell_region",
                        "background",
                    ],
                },
                "artifacts": artifact_hashes,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifact_hashes["dataset-report.json"] = sha256(
        dataset_report_path.read_bytes()
    ).hexdigest()
    return NnUNetExportReport(dataset_root, dataset_json, artifact_hashes)
