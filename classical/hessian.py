"""Reconstruction of the published Gaussian/Hessian/vectorization method."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from skimage import filters, morphology

from cellvector.domain.models import PredictionProposal
from cellvector.io.images import ImportedFrame

from .base import empty_proposal, normalize_percentile, proposal_from_skeleton


@dataclass(frozen=True)
class HessianParameters:
    gaussian_sigma_px: float = 0.8
    ridge_sigmas_px: tuple[float, ...] = (1.0, 2.0, 3.0)
    min_component_px: int = 6
    min_branch_px: float = 4.0


class HessianVectorPipeline:
    method = "classical_hessian"

    def analyze(
        self,
        frame: ImportedFrame,
        parameters: HessianParameters | None = None,
    ) -> PredictionProposal:
        config = parameters or HessianParameters()
        values, warnings = normalize_percentile(frame.array)
        if warnings:
            return empty_proposal(
                frame,
                method=self.method,
                parameters=asdict(config),
                warnings=warnings,
            )
        smoothed = filters.gaussian(
            values,
            sigma=config.gaussian_sigma_px,
            preserve_range=True,
        )
        response = filters.sato(
            smoothed,
            sigmas=config.ridge_sigmas_px,
            black_ridges=False,
        )
        threshold = filters.threshold_triangle(response)
        binary = morphology.remove_small_objects(
            response > threshold,
            max_size=config.min_component_px - 1,
        )
        skeleton = morphology.skeletonize(binary)
        return proposal_from_skeleton(
            frame,
            skeleton,
            method=self.method,
            parameters=asdict(config),
            warnings=warnings,
            min_path_px=config.min_branch_px,
        )
