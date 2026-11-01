"""Featurizer example: keep the most variable genes of the training samples.

A featurizer turns one side of the dataset (here: the samples) into features. ``fit`` sees
only the training samples, so anything it learns, such as which genes to keep, cannot leak
information from the test samples.
"""

from __future__ import annotations

from typing import Any, ClassVar, Self

import numpy as np
from drevalpy.plugin import (
    FeatureSource,
    Featurizer,
    FeaturizerResult,
    Side,
    dense_result,
    register_featurizer,
    view_matrix,
)

VIEW = "gene_expression"


@register_featurizer("demoTopGenes", description="The n most variable genes of the training samples.")
class TopVarianceGenes(Featurizer):
    """Select the genes whose expression varies most across the training samples."""

    sides: ClassVar[frozenset[Side]] = frozenset({"sample"})
    input_views: ClassVar[tuple[str, ...]] = (VIEW,)

    def __init__(self, n_genes: int = 100) -> None:
        """Create an unfitted featurizer.

        Args:
            n_genes: How many genes to keep.
        """
        self.n_genes = n_genes
        self._genes: tuple[str, ...] | None = None

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Declare ``n_genes``; the ``default`` is used when nothing is tuned.

        Returns:
            The search space, one entry per ``__init__`` keyword.
        """
        return {"n_genes": {"type": "int", "low": 10, "high": 1000, "default": 100}}

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Pick the genes with the highest variance across *train_ids*.

        Returns:
            This featurizer, fitted.
        """
        _ = train_pair_ids, es_pair_ids
        names = np.asarray(source.feature_names(VIEW))
        variance = np.asarray(source.view(VIEW, train_ids)).var(axis=0)
        # A stable sort keeps the choice independent of ties' positions in the view.
        keep = np.sort(np.argsort(-variance, kind="stable")[: self.n_genes])
        self._genes = tuple(names[keep])
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Return the selected genes of *ids* as one numeric block named after the view.

        Returns:
            The features of *ids*, in the order of *ids*.

        Raises:
            RuntimeError: If the featurizer is not fitted.
        """
        if self._genes is None:
            msg = "demoTopGenes must be fitted before transforming"
            raise RuntimeError(msg)
        return dense_result(ids, VIEW, view_matrix(source, VIEW, ids, self._genes), self._genes)
