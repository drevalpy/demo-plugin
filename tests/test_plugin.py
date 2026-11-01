"""Check each component with drevalpy's conformance checks, and the plugin as installed."""

from __future__ import annotations

import drevalpy
import numpy as np
import pytest
from drevalpy import ModelSpec
from drevalpy.plugin import CurveQuality, ModelResult, save_png
from drevalpy.registry import splitter
from drevalpy.testing import FEATURIZER_CHECKS, PREDICTOR_CHECKS, build_synthetic_batch, check_plugin
from drevalpy.types.results import RunResult

from drevalpy_demo_plugin.featurizers import TopVarianceGenes
from drevalpy_demo_plugin.plots import PredictionScatter
from drevalpy_demo_plugin.predictors import Ridge


def test_plugin_registers_every_component():
    report = check_plugin("demo")
    assert report.components == {
        "featurizer": ("demoTopGenes",),
        "predictor": ("demoRidge",),
        "splitter": ("DEMO_LSO",),
        "visualization": ("demoScatter",),
    }


def test_plugin_registers_the_preset():
    assert ModelSpec.from_name("DemoModel").config.predictor.name == "demoRidge"


@pytest.mark.parametrize("check", FEATURIZER_CHECKS)
def test_featurizer_conforms(check, dataset):
    check(TopVarianceGenes, dataset)


@pytest.mark.parametrize("check", PREDICTOR_CHECKS)
def test_predictor_conforms(check, dataset):
    check(Ridge, build_synthetic_batch(dataset, sample_block_names=("gene_expression",)))


def test_splitter_tests_every_sample_once(dataset):
    # Resolving through the registry runs drevalpy's leakage check on every fold.
    folds = splitter.get("DEMO_LSO")(dataset, n_splits=3, quality=CurveQuality.default())
    tested = np.sum([fold.test.mask.any(axis=1) for fold in folds], axis=0)
    assert len(folds) == 3
    assert set(tested) <= {0, 1}


def test_plot_draws_a_model_result(tmp_path):
    rng = np.random.default_rng(0)
    run = RunResult(
        model_name="demo",
        dataset_name="SYNTHETIC",
        fold_index=0,
        predictions=rng.normal(size=16),
        ground_truth=rng.normal(size=16),
        sample_ids=np.array([f"S{index}" for index in range(16)]),
        drug_ids=np.array(["D"] * 16),
    )
    result = ModelResult(model_name="demo", dataset_name="SYNTHETIC", runs=[run])
    save_png(PredictionScatter(), result, tmp_path / "scatter.png")
    assert (tmp_path / "scatter.png").stat().st_size > 0


def test_preset_runs_end_to_end_on_the_plugin_splitter(dataset):
    result = drevalpy.run(
        [ModelSpec.from_name("DemoModel")], dataset, "DEMO_LSO", hyperparameter_tuning=False, on_error="raise"
    )
    assert result.model_names == ["DemoModel"]
    assert len(result.models[0].runs) == 5
