"""Splitter example: leave-sample-out without shuffling.

A splitter is a function that returns one ``Fold`` per fold: boolean train, validation and
test masks over the sample-by-drug response matrix. ``validation="LSO"`` makes drevalpy check
every fold for a sample that is in both train and test, so a bug cannot inflate the scores.
"""

from __future__ import annotations

import numpy as np
from drevalpy.plugin import CurveQuality, Dataset, Fold, PairMask, observed_mask, register_splitter


@register_splitter(
    "DEMO_LSO",
    "Leave-Sample-Out: sample i is tested in fold i mod n_splits, so folds follow the dataset order",
    validation="LSO",
)
def round_robin(
    dataset: Dataset,
    n_splits: int = 5,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    *,
    quality: CurveQuality,
) -> list[Fold]:
    """Test every sample once; draw the validation samples from the rest at random.

    Args:
        dataset: Dataset whose response matrix is split.
        n_splits: Number of folds.
        validation_ratio: Share of the training samples moved to validation.
        random_state: Seed for the validation draw.
        quality: The run's curve-quality rule; only pairs passing it are used.

    Returns:
        One fold per test group.
    """
    usable = observed_mask(dataset, quality)
    samples = np.arange(usable.shape[0])
    rng = np.random.default_rng(random_state)
    folds = []
    for fold_index in range(n_splits):
        test = samples[samples % n_splits == fold_index]
        rest = rng.permutation(samples[samples % n_splits != fold_index])
        n_val = max(1, round(validation_ratio * len(rest)))
        folds.append(
            Fold(
                train=PairMask(_rows(usable, rest[n_val:])),
                val=PairMask(_rows(usable, rest[:n_val])),
                test=PairMask(_rows(usable, test)),
            )
        )
    return folds


def _rows(usable: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Keep the usable pairs of *rows* and nothing else."""
    mask = np.zeros_like(usable)
    mask[rows] = usable[rows]
    return mask
