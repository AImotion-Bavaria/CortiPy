"""Chart data model + rendering primitives for the Streamlit UI.

Extracted from apps/streamlit_app.py (modularization). Holds the self-contained
data model, downsampling, and the matplotlib renderer. The chart *builders*
(``_chart_from_*``) stay in the app for now since they are tangled with evaluation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


@dataclass(frozen=True)
class ChartSeries:
    name: str
    x: np.ndarray
    y: np.ndarray


@dataclass
class ChartData:
    key: str
    title: str
    x_label: str
    y_label: str
    series: List[ChartSeries]
    description: Optional[str] = None


def downsample_series(x: np.ndarray, y: np.ndarray, max_points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    step = max(1, math.ceil(len(x) / max_points))
    return x[::step], y[::step]


def render_chart(chart: ChartData) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    for series in chart.series:
        ax.plot(series.x, series.y, label=series.name)
    ax.set_title(chart.title)
    ax.set_xlabel(chart.x_label)
    ax.set_ylabel(chart.y_label)
    if len(chart.series) > 1:
        ax.legend(loc="best")
    if chart.description:
        ax.text(
            0.01,
            0.02,
            chart.description,
            transform=ax.transAxes,
            fontsize=8,
            color="gray",
            ha="left",
        )
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
