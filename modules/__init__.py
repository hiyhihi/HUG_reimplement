"""Modules package for HUG"""

from .losses import (
    HUGLoss,
    HolisticContrastiveLoss,
    FineGrainedContrastiveLoss,
    MultiModalCoordinationLoss,
    uncertainty_aware_distance,
    pairwise_uncertainty_distance
)
from .dynamic_weighting import DynamicWeighting
from .metrics import CIRMetrics, compute_recall_at_k

__all__ = [
    'HUGLoss',
    'HolisticContrastiveLoss',
    'FineGrainedContrastiveLoss',
    'MultiModalCoordinationLoss',
    'uncertainty_aware_distance',
    'pairwise_uncertainty_distance',
    'DynamicWeighting',
    'CIRMetrics',
    'compute_recall_at_k'
]
