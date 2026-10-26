"""MOSAIC-FL model architectures and baseline reference classifiers."""

from mosaic.models.baselines import MajorityClassClassifier, MotionEnergyClassifier
from mosaic.models.temporal_baseline import BaselineTemporalClassifier

__all__ = [
    "BaselineTemporalClassifier",
    "MajorityClassClassifier",
    "MotionEnergyClassifier",
]
