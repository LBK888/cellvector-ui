"""Reconstruction of the published Fiji microridge-length pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology

from cellvector.domain.models import PredictionProposal
from cellvector.io.images import ImportedFrame

from .base import empty_proposal, normalize_percentile, proposal_from_skeleton


@dataclass(frozen=True)
class FijiParameters:
    smoothing_passes: int = 3
    smoothing_size_px: int = 3
    laplacian_radius_px: int = 1
    min_branch_px: float = 4.0


class FijiReconstructionPipeline:
    method = "classical_fiji_reconstruction"

    def analyze(
        self,
        frame: ImportedFrame,
        parameters: FijiParameters | None = None,
    ) -> PredictionProposal:
        config = parameters or FijiParameters()
        values, warnings = normalize_percentile(frame.array)
        if warnings:
            return empty_proposal(
                frame,
                method=self.method,
                parameters=asdict(config),
                warnings=warnings,
            )
        smoothed = values
        for _ in range(config.smoothing_passes):
            smoothed = ndi.uniform_filter(
                smoothed,
                size=config.smoothing_size_px,
                mode="nearest",
            )
        size = 2 * config.laplacian_radius_px + 1
        footprint = np.ones((size, size), dtype=bool)
        laplacian = (
            morphology.dilation(smoothed, footprint)
            + morphology.erosion(smoothed, footprint)
            - 2.0 * smoothed
        )
        response = np.abs(laplacian)
        threshold = filters.threshold_triangle(response)
        binary = morphology.remove_small_objects(response > threshold, max_size=2)
        skeleton = morphology.skeletonize(binary)
        return proposal_from_skeleton(
            frame,
            skeleton,
            method=self.method,
            parameters=asdict(config),
            warnings=warnings,
            min_path_px=config.min_branch_px,
        )
