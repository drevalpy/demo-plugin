"""Plot example: predicted against observed response, one figure per model.

The report runs every registered plot whose requirements the experiment meets.
``result_type = ModelResult`` makes it draw this one once per model.
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from drevalpy.plugin import ModelResult, Plot, register_visualization


@register_visualization("demoScatter", "Predicted against observed response, pooled over a model's folds.")
class PredictionScatter(Plot):
    """Scatter every test prediction of one model against its observed value."""

    result_type = ModelResult

    def figure(self, result: ModelResult) -> matplotlib.figure.Figure:
        """Draw the scatter of *result*.

        Args:
            result: The model whose runs are pooled.

        Returns:
            A Matplotlib figure, which the report embeds as a PNG.
        """
        observed = np.concatenate([run.ground_truth for run in result.runs])
        predicted = np.concatenate([run.predictions for run in result.runs])
        figure, axes = plt.subplots(figsize=(4, 4))
        axes.scatter(observed, predicted, s=4, alpha=0.5)
        axes.axline((0, 0), slope=1, color="grey", linewidth=0.8)
        axes.set_xlabel("observed")
        axes.set_ylabel("predicted")
        axes.set_title(result.model_name)
        figure.tight_layout()
        return figure
