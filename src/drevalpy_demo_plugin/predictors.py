"""Predictor example: ridge regression on the design matrix.

``MatrixPredictor`` hands the predictor one row per sample-drug pair: the sample features
and the drug features side by side. It only has to fit and score that matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from drevalpy.plugin import Accelerator, MatrixPredictor, register_predictor


@register_predictor(
    "demoRidge",
    description="Ridge regression solved with the normal equations.",
    supports_accelerator=Accelerator.NONE,
    license="MIT",
)
class Ridge(MatrixPredictor):
    """Penalised least squares with an unpenalised intercept."""

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Declare ``penalty``; HPO draws it on a log scale.

        Returns:
            The search space; the values in effect are in ``self._hyperparameters``.
        """
        return {"penalty": {"type": "float", "low": 1e-3, "high": 1e3, "log": True, "default": 1.0}}

    def _fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
        """Solve for the weights; whatever is stored on ``self`` is saved with the model."""
        self._center = x.mean(axis=0)
        self._intercept = float(y.mean())
        centered = x - self._center
        gram = centered.T @ centered + float(self._hyperparameters["penalty"]) * np.eye(x.shape[1])
        self._weights = np.linalg.solve(gram, centered.T @ (y - self._intercept))

    def _predict_matrix(self, x: np.ndarray) -> np.ndarray:
        """Score one row per pair."""
        return (x - self._center) @ self._weights + self._intercept
