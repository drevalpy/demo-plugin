"""Shared fixtures: an in-memory dataset, so the tests need no files or network."""

from __future__ import annotations

import pytest
from drevalpy import synthetic
from drevalpy.types.data.dataset import Dataset


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    """60 samples, 8 drugs and a ``gene_expression`` view."""
    return synthetic()
