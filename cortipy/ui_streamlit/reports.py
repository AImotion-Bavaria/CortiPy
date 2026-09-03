"""Chart collection and report rendering for the Streamlit UI."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from cortipy.ui_streamlit.charts import (
    ChartData,
    _chart_from_alpha_power,
    _chart_from_average_signals,
    _chart_from_eval_fft,
    _chart_from_eval_psd,
    _chart_from_metric_vector,
    _chart_from_psd,
    _chart_from_raw_data,
    render_chart,
)
from cortipy.ui_streamlit.fields import coerce_number
from cortipy.ui_streamlit.plot_windows import render_matplotlib_window_launcher, safe_window_key


LOGGER = logging.getLogger(__name__)


def autoevaluate_if_needed(params: Dict[str, Any], data: Optional[np.ndarray]) -> Dict[str, Any]:
    if data is None:
        return params.get("Evaluation") or {}

    data_array = np.asarray(data)

    if data_array.size == 0 or (data_array.ndim >= 2 and data_array.shape[1] == 0):
        LOGGER.warning("Skipping auto-evaluation: empty data buffer", extra={"shape": data_array.shape})
        return params.get("Evaluation") or {}

    evaluation: Dict[str, Any] = {}
    method_name = str(params.get("Method", "")).lower()
    evaluator_map = {
        "alpha": "AlphaEvaluator",
        "ssvep": "SsvepEvaluator",
        "assr": "AssrEvaluator",
        "vep": "VepEvaluator",
        "p300": "P300Evaluator",
        "bera": "BeraEvaluator",
    }
    evaluator_name = evaluator_map.get(method_name)
    if not evaluator_name:
        return evaluation

    try:  # pragma: no cover - runtime convenience
        from cortipy.core.context import ModuleContext

        if evaluator_name == "AlphaEvaluator":
            from cortipy.evaluation.alpha import AlphaEvaluator as EvalCls
        elif evaluator_name == "SsvepEvaluator":
            from cortipy.evaluation.ssvep import SsvepEvaluator as EvalCls
        elif evaluator_name == "AssrEvaluator":
            from cortipy.evaluation.assr import AssrEvaluator as EvalCls
        elif evaluator_name == "VepEvaluator":
            from cortipy.evaluation.vep import VepEvaluator as EvalCls
        elif evaluator_name == "P300Evaluator":
            from cortipy.evaluation.p300 import P300Evaluator as EvalCls
        elif evaluator_name == "BeraEvaluator":
            from cortipy.evaluation.bera import BeraEvaluator as EvalCls
        else:
            return evaluation

        temp_params = _params_for_evaluation(params)
        temp_params["Parameters"] = dict(params.get("Parameters", {}))
        temp_params["data"] = data_array
        ctx = ModuleContext(temp_params)
        EvalCls(show_plots=False).evaluate(ctx)
        evaluation = ctx.params.get("Evaluation") or {}
    except Exception as exc:
        LOGGER.warning("Evaluation generation failed for method %s: %s", method_name, exc)
        return params.get("Evaluation") or {}
    return evaluation


def _params_for_evaluation(params: Dict[str, Any]) -> Dict[str, Any]:
    temp_params = {key: value for key, value in (params or {}).items() if key not in {"Evaluation", "data"}}
    temp_params["Parameters"] = dict((params or {}).get("Parameters", {}) or {})
    if isinstance((params or {}).get("Channels"), list):
        temp_params["Channels"] = [
            dict(row) if isinstance(row, dict) else row
            for row in (params or {}).get("Channels", [])
        ]
    metadata = (params or {}).get("Metadata")
    if isinstance(metadata, dict):
        temp_params["Metadata"] = dict(metadata)
    return temp_params


def render_evaluation_figures(params: Dict[str, Any], data: Optional[np.ndarray]) -> List[plt.Figure]:
    if data is None:
        return []
    method_name = str(params.get("Method", "")).lower()
    evaluator_map = {
        "alpha": "AlphaEvaluator",
        "ssvep": "SsvepEvaluator",
        "assr": "AssrEvaluator",
        "vep": "VepEvaluator",
        "p300": "P300Evaluator",
        "bera": "BeraEvaluator",
    }
    evaluator_name = evaluator_map.get(method_name)
    if not evaluator_name:
        return []

    try:  # pragma: no cover - UI rendering only
        from cortipy.core.context import ModuleContext

        if evaluator_name == "AlphaEvaluator":
            from cortipy.evaluation.alpha import AlphaEvaluator as EvalCls
        elif evaluator_name == "SsvepEvaluator":
            from cortipy.evaluation.ssvep import SsvepEvaluator as EvalCls
        elif evaluator_name == "AssrEvaluator":
            from cortipy.evaluation.assr import AssrEvaluator as EvalCls
        elif evaluator_name == "VepEvaluator":
            from cortipy.evaluation.vep import VepEvaluator as EvalCls
        elif evaluator_name == "P300Evaluator":
            from cortipy.evaluation.p300 import P300Evaluator as EvalCls
        elif evaluator_name == "BeraEvaluator":
            from cortipy.evaluation.bera import BeraEvaluator as EvalCls
        else:
            return []

        before = set(plt.get_fignums())
        temp_params = _params_for_evaluation(params)
        temp_params["ReportAnalyzer"] = False
        temp_params["data"] = data
        EvalCls(show_plots=True).evaluate(ModuleContext(temp_params))
        new_nums = [num for num in plt.get_fignums() if num not in before]
        return [plt.figure(num) for num in new_nums]
    except Exception as exc:
        LOGGER.debug("Evaluation figure rendering failed: %s", exc)
        return []


def collect_chart_data(label: str, params: Dict[str, Any], data: Optional[np.ndarray], aggregate: bool = False) -> Dict[str, ChartData]:
    charts: Dict[str, ChartData] = {}

    if data is not None:
        raw_chart = _chart_from_raw_data(label, params, data, aggregate=aggregate)
        if raw_chart:
            charts.setdefault("raw", raw_chart)
        psd_chart = _chart_from_psd(label, params, data, aggregate=aggregate)
        if psd_chart:
            charts.setdefault("psd", psd_chart)

    evaluation = autoevaluate_if_needed(params, data)
    if evaluation:
        alpha_chart = _chart_from_alpha_power(label, evaluation.get("alphaPower", {}), aggregate=aggregate, params=params)
        if alpha_chart:
            charts.setdefault("alpha_power", alpha_chart)
        eval_psd_chart = _chart_from_eval_psd(label, evaluation.get("PSD", {}), aggregate=aggregate, params=params)
        if eval_psd_chart:
            charts.setdefault("eval_psd", eval_psd_chart)
        fft_chart = _chart_from_eval_fft(label, evaluation.get("fft", {}), aggregate=aggregate, params=params)
        if fft_chart:
            charts.setdefault("eval_fft", fft_chart)
        avg_chart = _chart_from_average_signals(label, evaluation.get("average_signals", {}), aggregate=aggregate, params=params)
        if avg_chart:
            charts.setdefault(avg_chart.key, avg_chart)

        peaks = evaluation.get("peak", {})
        if isinstance(peaks, dict):
            for peak_name, peak_vals in peaks.items():
                val_chart = _chart_from_metric_vector(label, f"{peak_name} amplitude", peak_vals, unit="Amplitude (uV)")
                if val_chart:
                    charts.setdefault(f"peak_{peak_name}", val_chart)
                time_key = f"{peak_name}Time"
                time_vals = evaluation.get(time_key) or evaluation.get(f"{peak_name}_time")
                time_chart = _chart_from_metric_vector(
                    label,
                    f"{peak_name} latency",
                    time_vals,
                    unit="Time (s)",
                )
                if time_chart:
                    charts.setdefault(f"peak_{peak_name}_latency", time_chart)

        t_res = evaluation.get("tRes")
        if isinstance(t_res, dict) and "h" in t_res:
            tres_chart = _chart_from_metric_vector(label, "tRes", t_res.get("h"), unit="tRes h")
            if tres_chart:
                charts.setdefault("metric_tRes", tres_chart)

        metric_map = {
            "SNR": "SNR",
            "SNR_dB": "SNR (dB)",
            "R_AM": "R_AM",
            "R_AM_dB": "R_AM (dB)",
            "Fsp": "Fsp",
            "Fmp": "Fmp",
            "RN_elberlingDon": "Residual noise (Elberling)",
            "RN_eclipse": "Residual noise (Eclipse)",
            "rho": "Correlation",
            "f_CCA": "F_CCA",
            "T2circ": "T2 circ",
            "p_values": "p-values",
            "SNR_2_45Hz": "SNR 2-45 Hz",
            "SNR_max": "SNR max",
            "amp_V": "Amplitude",
            "SNR_time": "SNR time",
            "SNR_Peak": "SNR peak",
            "RN_micV": "Residual noise (uV)",
        }
        for key, label_name in metric_map.items():
            metric_chart = _chart_from_metric_vector(label, label_name, evaluation.get(key), unit=label_name)
            if metric_chart:
                charts.setdefault(metric_chart.key, metric_chart)

    eval_keys = list(evaluation.keys()) if isinstance(evaluation, dict) else None
    chart_keys = sorted(charts.keys()) if charts else []
    LOGGER.info(
        "Chart collection complete label=%s method=%s data_present=%s eval_keys=%s charts=%s",
        label,
        str(params.get("Method", "")).lower(),
        bool(data is not None),
        eval_keys,
        chart_keys,
    )
    return charts


def render_chart_section(label: str, params: Dict[str, Any], data: Optional[np.ndarray], aggregate: bool = False) -> None:
    eval_figures = render_evaluation_figures(params, data)
    LOGGER.debug(
        "Render chart section",
        extra={
            "label": label,
            "method": params.get("Method"),
            "eval_figures": len(eval_figures),
        },
    )
    charts = collect_chart_data(label, params or {}, data, aggregate=aggregate)
    if not charts:
        st.info("No charts available for this session yet.")
        return
    st.subheader(f"Charts - {label}")
    if eval_figures:
        st.caption("Evaluation plots")
        for idx, fig in enumerate(eval_figures, start=1):
            render_matplotlib_window_launcher(
                fig,
                f"{label} evaluation plot {idx}",
                f"eval_{safe_window_key(label)}_{idx}",
                button_label="Open evaluation plot",
            )
        plt.close("all")
    for key in sorted(charts.keys()):
        render_chart(charts[key])


def _safe_stat_value(func, array: np.ndarray) -> Optional[float]:
    try:
        value = float(func(array))
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _summarize_session_stats(label: str, params: Dict[str, Any], data: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    if data is None:
        return None
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr[:, None]

    n_samples, n_channels = arr.shape[0], arr.shape[1]
    flat = arr.reshape(-1)
    fs_value = coerce_number(params.get("Parameters", {}).get("fs") if params else None)
    fs = float(fs_value) if fs_value else 0.0
    duration = (n_samples / fs) if fs > 0 else None

    return {
        "Session": label,
        "Samples": n_samples,
        "Channels": n_channels,
        "Duration (s)": round(duration, 2) if duration is not None and math.isfinite(duration) else None,
        "Median": _safe_stat_value(np.nanmedian, flat),
        "Mean": _safe_stat_value(np.nanmean, flat),
        "Std": _safe_stat_value(np.nanstd, flat),
        "P25": _safe_stat_value(lambda a: np.nanpercentile(a, 25), flat),
        "P75": _safe_stat_value(lambda a: np.nanpercentile(a, 75), flat),
        "Min": _safe_stat_value(np.nanmin, flat),
        "Max": _safe_stat_value(np.nanmax, flat),
    }


def _summarize_channel_stats(label: str, params: Dict[str, Any], data: Optional[np.ndarray]) -> List[Dict[str, Any]]:
    if data is None:
        return []
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        arr = arr[:, None]

    n_channels = arr.shape[1]
    labels: List[str] = []
    for entry in params.get("Channels") or []:
        name = entry.get("Channel") or entry.get("Label")
        if name:
            labels.append(str(name))
    if len(labels) < n_channels:
        labels.extend([f"Ch {idx + 1}" for idx in range(len(labels), n_channels)])

    rows: List[Dict[str, Any]] = []
    for idx in range(n_channels):
        ch_data = arr[:, idx]
        rows.append(
            {
                "Session": label,
                "Channel": labels[idx] if idx < len(labels) else f"Ch {idx + 1}",
                "Median": _safe_stat_value(np.nanmedian, ch_data),
                "Mean": _safe_stat_value(np.nanmean, ch_data),
                "Std": _safe_stat_value(np.nanstd, ch_data),
                "P25": _safe_stat_value(lambda a: np.nanpercentile(a, 25), ch_data),
                "P75": _safe_stat_value(lambda a: np.nanpercentile(a, 75), ch_data),
                "Min": _safe_stat_value(np.nanmin, ch_data),
                "Max": _safe_stat_value(np.nanmax, ch_data),
            }
        )
    return rows


def render_comparison_charts(payloads: List[tuple[str, Dict[str, Any], Optional[np.ndarray]]]) -> None:
    session_stats: List[Dict[str, Any]] = []
    channel_stats: List[Dict[str, Any]] = []
    merged: Dict[str, ChartData] = {}
    for label, params, data in payloads:
        stats_row = _summarize_session_stats(label, params or {}, data)
        if stats_row:
            session_stats.append(stats_row)
        channel_stats.extend(_summarize_channel_stats(label, params or {}, data))
        charts = collect_chart_data(label, params or {}, data, aggregate=True)
        for key, chart in charts.items():
            if key not in merged:
                merged[key] = ChartData(
                    key=key,
                    title=f"{chart.title} (comparison)",
                    x_label=chart.x_label,
                    y_label=chart.y_label,
                    series=list(chart.series),
                    description=chart.description,
                )
            else:
                merged[key].series.extend(chart.series)

    if session_stats:
        st.subheader("Comparison report")
        summary_columns = [
            "Session",
            "Samples",
            "Channels",
            "Duration (s)",
            "Median",
            "Mean",
            "Std",
            "P25",
            "P75",
            "Min",
            "Max",
        ]
        df = pd.DataFrame(session_stats)
        df = df[[col for col in summary_columns if col in df.columns]]
        st.dataframe(df)

    if channel_stats:
        with st.expander("Per-channel statistics"):
            channel_columns = [
                "Session",
                "Channel",
                "Median",
                "Mean",
                "Std",
                "P25",
                "P75",
                "Min",
                "Max",
            ]
            channel_df = pd.DataFrame(channel_stats)
            channel_df = channel_df[[col for col in channel_columns if col in channel_df.columns]]
            st.dataframe(channel_df)

    if not merged:
        st.info("No comparable charts for the selected sessions.")
        return

    st.subheader("Comparison overlays")
    for key in sorted(merged.keys()):
        render_chart(merged[key])
