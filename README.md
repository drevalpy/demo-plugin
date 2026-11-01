# drevalpy-demo-plugin

A minimal [DrEvalPy](https://github.com/daisybio/drevalpy) plugin. It contains one small,
working example of each thing a plugin can add to DrEvalPy, so you can copy it as a starting
point for your own.

| What              | Name           | File                                                        | What it does                                                      |
| ----------------- | -------------- | ----------------------------------------------------------- | ----------------------------------------------------------------- |
| Sample featurizer | `demoTopGenes` | [`featurizers.py`](src/drevalpy_demo_plugin/featurizers.py) | Keeps the `n_genes` most variable genes of the training samples   |
| Predictor         | `demoRidge`    | [`predictors.py`](src/drevalpy_demo_plugin/predictors.py)   | Ridge regression, with a tunable `penalty`                        |
| Split mode        | `DEMO_LSO`     | [`splitters.py`](src/drevalpy_demo_plugin/splitters.py)     | Leave-sample-out; sample _i_ is tested in fold _i_ mod `n_splits` |
| Plot              | `demoScatter`  | [`plots.py`](src/drevalpy_demo_plugin/plots.py)             | Predicted against observed response, one figure per model         |
| Model preset      | `DemoModel`    | [`zoo.yaml`](src/drevalpy_demo_plugin/zoo.yaml)             | `demoTopGenes` + built-in `identity` drug features + `demoRidge`  |
| Registration      |                | [`components.py`](src/drevalpy_demo_plugin/components.py)   | The entry point: imports the four modules and loads the preset    |
| Tests             |                | [`tests/test_plugin.py`](tests/test_plugin.py)              | DrEvalPy's conformance checks, plus one end-to-end run            |

The components are deliberately simple. Each file explains in its docstring what DrEvalPy
expects from that kind of component; the DrEvalPy docs (_How-to → Extend DrEvalPy_) cover
every option.

## How a plugin works

A plugin is an ordinary Python package that declares an entry point in the `drevalpy.plugins`
group ([`pyproject.toml`](pyproject.toml)):

```toml
[project.entry-points."drevalpy.plugins"]
demo = "drevalpy_demo_plugin.components"
```

Once the package is installed in the same environment as DrEvalPy, every `drevalpy` command
and every Python session that imports `drevalpy` imports that module. Importing it runs the
`@register_featurizer`, `@register_predictor`, `@register_splitter` and
`@register_visualization` decorators, and `load_extensions(zoo_files=[...])` adds the presets.
There is no flag to pass and nothing to configure: from then on the names in the table above
work wherever a built-in name does.

## Install

The repository is private, so install it over SSH (or with a token) into the environment that
has DrEvalPy:

```bash
pip install "drevalpy-demo-plugin @ git+ssh://git@github.com/drevalpy/demo-plugin.git"
# or, in a uv project:
uv add "drevalpy-demo-plugin @ git+ssh://git@github.com/drevalpy/demo-plugin.git"
```

The package depends on DrEvalPy 2.0 from its `v2-13-docs-reference` branch until 2.0 is on
PyPI. DrEvalPy needs PyTorch, whose build you choose yourself: install DrEvalPy with the `cpu`,
`cu126` or `cu130` extra, or install `torch` before the plugin.

Check that the plugin loaded:

```bash
drevalpy list plugins             # demo   loaded   drevalpy_demo_plugin.components
drevalpy list zoo-presets         # ... DemoModel   demoTopGenes:identity:demoRidge ...
drevalpy list splitters           # ... DEMO_LSO ...
drevalpy list visualizations      # ... demoScatter ...
```

If the status is `failed`, `drevalpy list plugins --traceback` shows why.

## Use it from the command line

Make a small dataset to try things on (or pass a registered dataset name such as `GDSC2`):

```bash
drevalpy data synthetic --output toy.h5mu
```

Cross-validate the preset on the plugin's split mode. `--no-hpo` skips hyperparameter tuning to
keep the run short; leave it out to tune `n_genes` and `penalty`:

```bash
drevalpy run DemoModel --dataset toy.h5mu --split-mode DEMO_LSO --no-hpo -o results
```

The report includes the plugin's plot for every model, next to the built-in plots:

```bash
drevalpy report results -o report
```

Plugin components mix freely with built-in ones. This grid runs every valid combination of the
given featurizers and predictors, here the plugin's ridge against the built-in elastic net on
the same features:

```bash
drevalpy run --dataset toy.h5mu --split-mode DEMO_LSO --no-hpo \
  -S demoTopGenes -D identity -P demoRidge -P elasticNet -o grid
```

Train a final model on all pairs and predict with it later:

```bash
drevalpy train DemoModel --dataset toy.h5mu --no-hpo -o model
drevalpy predict model --dataset toy.h5mu -o predictions.npz
```

The split mode also works on its own. To run the folds in parallel, for example as cluster jobs,
write them to files and train one fold per job:

```bash
drevalpy data split toy.h5mu folds --mode DEMO_LSO
drevalpy single DemoModel toy.h5mu folds/fold_0.npz fold_0 --no-hpo
```

## Use it from Python

Nothing needs importing from the plugin: importing `drevalpy` loads it.

```python
import drevalpy
from drevalpy import DRPModel, ModelSpec

dataset = drevalpy.load("toy.h5mu")
folds = drevalpy.split(dataset, "DEMO_LSO", n_splits=5)

spec = ModelSpec.from_name("DemoModel")
result = drevalpy.run([spec], dataset, "DEMO_LSO", hyperparameter_tuning=False)
print(result)

# Fix hyperparameters yourself; keys are <slot>.<component>.<name>.
smaller = spec.with_hyperparameters({"sample_featurizer.demoTopGenes.n_genes": 20})

model = DRPModel(smaller)
model.train(dataset, folds[0].train)
model.save("demo_model.zip")
predictions = DRPModel.load("demo_model.zip").predict(dataset, folds[0].test)
```

A saved model stores the plugin's classes by their import path, so the plugin must be installed
wherever the model is loaded. Load only model files you trust: loading unpickles them.

## Make your own plugin from this one

1. Copy the repository and rename the package: the directory `src/drevalpy_demo_plugin`, the
   `name` in `pyproject.toml`, the entry point (name and module path) and the imports.
2. Give every component a name of your own. Names are global across all installed plugins and
   DrEvalPy's built-ins, and registering a taken name raises an error.
3. Replace the example code, and delete the files for kinds of component you don't need; also
   drop them from the imports in `components.py`. Keep the tests for the rest.
4. Keep every import from `drevalpy.plugin`: that module is the API DrEvalPy promises to keep
   stable. Deeper module paths may change between releases.

Model presets are the only YAML a plugin should load itself. Dataset sources
(`drevalpy.registry.dataset.register_source`) are saved to the user's configuration file, so a
plugin should not register them on import; ship them as a YAML file that users load once with
`drevalpy -e DIR`.

## Develop

```bash
uv sync                      # installs DrEvalPy, CPU PyTorch and the dev tools
uv run pytest                # about 10 seconds; needs no data files or network
uv run prek install          # run ruff, ty and the tests before each commit
```

`uv.lock` pins DrEvalPy to one commit of its branch. To test against the newest one, run
`uv lock --upgrade-package drevalpy`.

The tests use DrEvalPy's conformance checks (`drevalpy.testing`): the featurizer and the
predictor must produce what they declare and survive a pickle round trip, the splitter must not
leak samples between train and test, and `check_plugin("demo")` must find all four components
through the installed entry point. CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml))
runs them with `DREVALPY_STRICT_PLUGINS=1`, which makes a plugin that fails to import an error
instead of a warning.
