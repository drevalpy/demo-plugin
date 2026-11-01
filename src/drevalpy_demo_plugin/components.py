"""Entry-point target: importing this module registers everything the plugin provides.

drevalpy imports it once per process, when ``drevalpy.registry`` is first imported. The
``@register_*`` decorators run as the component modules are imported; the presets are
loaded afterwards, because they name those components.
"""

from __future__ import annotations

from pathlib import Path

from drevalpy.plugin import load_extensions

from drevalpy_demo_plugin import featurizers, plots, predictors, splitters

# directories=[] stops load_extensions from also reading DREVALPY_EXTENSIONS_DIR.
load_extensions(zoo_files=[str(Path(__file__).with_name("zoo.yaml"))], directories=[])

__all__ = ["featurizers", "plots", "predictors", "splitters"]
