"""Featurizer examples, one per side: a sample featurizer and a drug featurizer.

A featurizer turns one side of the dataset, the samples or the drugs, into features. Which side
is fixed when it is registered (``side=Side.SAMPLE`` or ``side=Side.DRUG``), and a model can use
it only in the slot of that side. ``fit`` sees only the training entities, so anything it
learns, such as which genes to keep, cannot leak information from the test entities.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import Any, ClassVar, Self

import numpy as np
from drevalpy.plugin import (
    BlockSpec,
    FeatureFormat,
    FeatureSource,
    Featurizer,
    FeaturizerResult,
    Side,
    dense_result,
    register_featurizer,
    view_matrix,
)

VIEW = "gene_expression"


@register_featurizer("demoTopGenes", side=Side.SAMPLE, description="The n most variable genes of the training samples.")
class TopVarianceGenes(Featurizer):
    """Select the genes whose expression varies most across the training samples."""

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


ATOM_COUNTS = "demo_atom_counts"
# A bracket atom such as [Na+] or [nH] names its element first; outside brackets, the two-letter
# halogens come before the one-letter elements, and lowercase letters are aromatic atoms.
_ATOM = re.compile(r"\[\d*([A-Z][a-z]?|[a-z]{1,2})|(Br|Cl|[BCNOSPFI]|[bcnosp])")


def count_atoms(smiles: str) -> Counter[str]:
    """Count the heavy atoms of each element in a SMILES string.

    Args:
        smiles: A SMILES string, such as ``"CC(=O)Oc1ccccc1C(=O)O"``.

    Returns:
        The number of atoms per element symbol; hydrogens outside brackets are implicit and not counted.
    """
    return Counter((match.group(1) or match.group(2)).capitalize() for match in _ATOM.finditer(smiles))


@register_featurizer(
    "demoAtomCounts", side=Side.DRUG, description="How many atoms of each element a drug's SMILES contains."
)
class AtomCounts(Featurizer):
    """Count the atoms of each element in the drug's SMILES.

    It reads no view, only the SMILES the dataset stores for every drug (``source.smiles``).
    """

    def __init__(self) -> None:
        """Create an unfitted featurizer."""
        self._elements: tuple[str, ...] | None = None

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Declare the one numeric block.

        The default derives the block from the view a featurizer reads; this one reads none.

        Returns:
            The blocks ``transform`` emits.
        """
        _ = config
        return (BlockSpec(ATOM_COUNTS, FeatureFormat.NUMERIC_MATRIX),)

    def valid_mask(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        """Mark the drugs that have a SMILES; drevalpy leaves the others out of fit and transform.

        Returns:
            One boolean per ID.
        """
        smiles = source.smiles(ids)
        return np.zeros(len(ids), dtype=bool) if smiles is None else smiles.notna().to_numpy()

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Collect the elements that occur in the training drugs; they become the columns.

        Returns:
            This featurizer, fitted.
        """
        _ = train_pair_ids, es_pair_ids
        seen: set[str] = set()
        for smiles in _smiles(source, train_ids):
            seen.update(count_atoms(smiles))
        self._elements = tuple(sorted(seen))
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Return the atom counts of *ids*, one column per element seen in training.

        Returns:
            The features of *ids*, in the order of *ids*.

        Raises:
            RuntimeError: If the featurizer is not fitted.
        """
        if self._elements is None:
            msg = "demoAtomCounts must be fitted before transforming"
            raise RuntimeError(msg)
        counts = [count_atoms(smiles) for smiles in _smiles(source, ids)]
        matrix = np.array([[row[element] for element in self._elements] for row in counts], dtype=float)
        return dense_result(ids, ATOM_COUNTS, matrix.reshape(len(ids), len(self._elements)), self._elements)


def _smiles(source: FeatureSource, ids: np.ndarray) -> list[str]:
    """Return the SMILES of *ids*, which ``valid_mask`` has checked all have one.

    Returns:
        One SMILES per ID.

    Raises:
        ValueError: If the dataset stores no SMILES.
    """
    smiles = source.smiles(ids)
    if smiles is None:
        msg = "demoAtomCounts needs SMILES, and the dataset has none"
        raise ValueError(msg)
    return [str(value) for value in smiles]
