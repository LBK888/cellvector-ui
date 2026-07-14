"""Import already-projected single-channel TIFF and PNG images."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from numpy.typing import NDArray
from PIL import Image

from cellvector.domain.models import SourceImage


class UnsupportedImageFormat(ValueError):
    def __init__(self, suffix: str) -> None:
        self.suffix = suffix
        super().__init__(f"unsupported image format: {suffix}")


@dataclass(frozen=True)
class ImportedFrame:
    array: NDArray[np.generic]
    source: SourceImage
    stack_metadata: dict[str, Any]


def import_frames(path: str | Path) -> list[ImportedFrame]:
    """Import one image or every independent projected frame in a TIFF stack."""

    source_path = Path(path).resolve()
    suffix = source_path.suffix.lower()
    checksum = sha256(source_path.read_bytes()).hexdigest()

    if suffix == ".png":
        arrays = [np.asarray(Image.open(source_path))]
        metadata: dict[str, Any] = {}
    elif suffix in {".tif", ".tiff"}:
        arrays, metadata = _read_tiff(source_path)
    else:
        raise UnsupportedImageFormat(suffix)

    frames: list[ImportedFrame] = []
    for index, array in enumerate(arrays):
        frame = np.asarray(array)
        if frame.ndim != 2:
            raise ValueError(
                f"CellVector accepts single-channel 2D frames; frame {index} has shape {frame.shape}"
            )
        frames.append(
            ImportedFrame(
                array=frame,
                source=SourceImage(
                    image_uri=str(source_path),
                    sha256=checksum,
                    frame_index=index,
                    width_px=int(frame.shape[1]),
                    height_px=int(frame.shape[0]),
                    dtype=str(frame.dtype),
                ),
                stack_metadata=dict(metadata),
            )
        )
    return frames


def _read_tiff(path: Path) -> tuple[list[NDArray[np.generic]], dict[str, Any]]:
    with tifffile.TiffFile(path) as tif:
        metadata = {
            str(key): _metadata_value(value)
            for key, value in (tif.imagej_metadata or {}).items()
        }
        array = np.asarray(tif.asarray())

    if array.ndim == 2:
        return [array], metadata
    if array.ndim == 3:
        return [np.asarray(frame) for frame in array], metadata
    raise ValueError(
        f"CellVector accepts 2D frames or a one-axis stack of 2D frames; TIFF shape is {array.shape}"
    )


def _metadata_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    return str(value)
