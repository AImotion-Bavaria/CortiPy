"""Canonical Streamlit entrypoint for the CortiPy UI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cortipy.ui import SaveManager  # noqa: E402
from cortipy.ui_streamlit import session as ui  # noqa: E402
from cortipy.ui_streamlit.constants import DEVICE_DEFAULT_CHANNELS  # noqa: E402
from cortipy.ui_streamlit.device_settings import render_device_config  # noqa: E402
from cortipy.ui_streamlit.fields import default_values  # noqa: E402
from cortipy.ui_streamlit.live import (  # noqa: E402
    LiveViewService,
    _reset_plot_window_open_state,
    _selected_recording_seconds,
)
from cortipy.ui_streamlit.reports import (  # noqa: E402
    render_chart_section,
    render_comparison_charts,
)
from cortipy.ui_streamlit.styles import inject_global_styles  # noqa: E402
from cortipy.ui_streamlit.workflow import (  # noqa: E402
    render_workflow_page,
    render_workflow_progress,
)


def main() -> None:
    st.set_page_config(page_title="cortipy UI", layout="wide")
    inject_global_styles(st)
    st.title("cortipy - EEG Measurement UI")
    ui.ensure_state()

    flash = st.session_state.pop("_flash", None)
    if flash:
        st.success(flash)

    controls = ui.render_sidebar_controls()
    default_save = controls.default_save
    simulate = controls.simulate
    live_view_enabled = controls.live_view_enabled
    live_view_window = controls.live_view_window
    start_button = controls.start_button
    imported_data = st.session_state.get("imported_data")

    ui.ensure_navigation_state()

    page = st.segmented_control(
        "Section",
        ui.APP_VIEW_OPTIONS,
        key="active_view_selector",
        label_visibility="collapsed",
        width="stretch",
    )
    if page is None:
        page = st.session_state.get("active_view") or "Workflow"
    st.session_state["active_view"] = page

    run_region = st.container()
    progress_slot = st.container()

    general_values = dict(st.session_state["general_form"])
    device_values = dict(st.session_state.setdefault("device_forms", {}).get(general_values.get("Device"), {}))
    method_values = dict(
        st.session_state["method_forms"].get(
            general_values.get("Method"),
            default_values(ui.METHOD_SCHEMAS.get(general_values.get("Method"), [])),
        )
    )
    participant_values = dict(st.session_state["participant"])

    if page == "Session configuration":
        general_values = ui.render_general_form()
        method_col, device_col = st.columns(2)
        with method_col:
            method_values = ui.render_method_form(general_values["Method"])
        with device_col:
            device_values = render_device_config(general_values["Device"])
        participant_values = ui.render_participant_form()

    if page == "Electrodes":
        ui.render_channel_editor(general_values["Device"])

    assembled_params = ui.assemble_params(general_values, method_values, participant_values, device_values)
    validation_issues = ui.validate_params(assembled_params)
    if simulate:
        validation_issues = [
            issue
            for issue in validation_issues
            if issue != "UNICORN configuration requires a serial port / address."
        ]
    if page in {"Session configuration", "Preview"}:
        render_workflow_progress(progress_slot, assembled_params, validation_issues)

    if page == "Workflow":
        render_workflow_page(assembled_params, validation_issues, ui.set_active_view)

    if page == "Live preview":
        ui.render_live_preview_tab(assembled_params, validation_issues)

    if page == "Preview":
        if validation_issues:
            st.warning(" ; ".join(validation_issues))
        st.json(assembled_params)
        st.download_button(
            "Download params.json",
            data=ui.params_to_json(assembled_params),
            file_name=f"{assembled_params['Method']}_params.json",
            mime="application/json",
        )

    if page == "Charts":
        current_results = st.session_state.get("last_results")
        chart_label: Optional[str] = None
        chart_params: Optional[Dict[str, Any]] = None
        chart_data: Optional[np.ndarray] = None

        if current_results:
            chart_label = current_results.get("label", "Last run")
            chart_params = current_results.get("params") or assembled_params
            chart_data = current_results.get("data")
            st.caption("Showing charts from the last run or loaded session.")
        elif imported_data is not None:
            chart_label = "Imported data"
            chart_params = assembled_params
            chart_data = imported_data
            st.caption("Using imported data with current parameters.")

        if chart_label and chart_params is not None:
            try:
                render_chart_section(chart_label, chart_params, chart_data)
            except Exception as exc:  # pragma: no cover
                ui.LOGGER.exception("Charts tab failed")
                st.error(f"Charts failed: {exc}")
        else:
            st.info("Run a measurement or load a saved session to see charts.")

    if page == "Saved sessions":
        primary_payload, compare_payloads = ui.render_saved_sessions(Path(default_save).expanduser())
        if primary_payload:
            label, params_loaded, data_loaded = primary_payload
            try:
                render_chart_section(label, params_loaded, data_loaded)
            except Exception as exc:  # pragma: no cover
                ui.LOGGER.exception("Saved-session charts failed")
                st.error(f"Charts failed: {exc}")
        if compare_payloads:
            try:
                render_comparison_charts(compare_payloads)
            except Exception as exc:  # pragma: no cover
                ui.LOGGER.exception("Comparison charts failed")
                st.error(f"Comparison charts failed: {exc}")

    if start_button:
        if validation_issues:
            issues_md = " - " + "\n - ".join(validation_issues)
            st.error(f"Please fix these configuration issues before starting a run:\n{issues_md}")
            ui.render_footer()
            return
        channels_for_run = len(assembled_params.get("Channels", [])) or int(
            assembled_params.get("Parameters", {}).get("NumberEEGChannels") or 0
        )
        if channels_for_run <= 0:
            channels_for_run = DEVICE_DEFAULT_CHANNELS.get(assembled_params.get("Device"), 8)
        selection = st.session_state.get("live_preview_channel_selection") or ["All"]
        if "All" in selection or not selection:
            selected_indices = list(range(max(1, channels_for_run)))
        else:
            channel_labels = [
                row.get("Channel") or f"Ch {idx + 1}"
                for idx, row in enumerate(assembled_params.get("Channels", []))
            ] or [f"Ch {idx + 1}" for idx in range(max(1, channels_for_run))]
            selected_indices = [channel_labels.index(label) for label in selection if label in channel_labels]
        if not selected_indices:
            selected_indices = list(range(max(1, channels_for_run)))
        save_dir = Path(default_save).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        params_to_run = dict(assembled_params)
        params_to_run.pop("Evaluation", None)
        params_to_run.pop("data", None)
        imported_data = st.session_state.get("imported_data")
        use_imported = st.session_state.get("use_imported_data", False)
        connection_required = ui._requires_device_connection(
            params_to_run,
            simulate=simulate,
            use_imported_data=use_imported,
            imported_data=imported_data,
        )
        connected_device = None
        if connection_required:
            if not ui._session_device_ready(params_to_run):
                st.error("Connect the device first, then press Start measurement.")
                ui.render_footer()
                return
            connected_device = st.session_state.get("_connected_device")
            if connected_device is None:
                st.error("The device connection was lost. Connect the device again.")
                ui.render_footer()
                return
        _reset_plot_window_open_state(
            "live_preview_signal",
            "live_preview_fft",
            *[f"live_channel_{idx + 1}" for idx in selected_indices],
        )
        st.session_state["_live_view_placeholder"] = None
        with run_region, st.status("Measurement running...", expanded=True) as run_status:
            progress_placeholder = st.empty()
            live_view_service = LiveViewService(
                ui.get_live_view_placeholder() if live_view_enabled else None,
                window_seconds=float(live_view_window),
                channel_indices=selected_indices,
                fft_placeholder=st.session_state.get("_live_preview_fft_placeholder") if live_view_enabled else None,
                progress_placeholder=progress_placeholder,
                total_seconds=_selected_recording_seconds(params_to_run),
                max_update_seconds=0.5,
            )
            if live_view_enabled:
                st.session_state["_live_view_active"] = True
                st.session_state["_live_view_banner"] = "Measurement running: live EEG and FFT updating below."
            try:
                if simulate:
                    params_to_run["data"] = ui.simulated_recording_data(params_to_run)
                    saved_path = SaveManager(save_dir)(params_to_run)
                    st.session_state["last_results"] = {
                        "label": getattr(saved_path, "name", "Simulated run"),
                        "params": params_to_run,
                        "data": params_to_run.get("data"),
                    }
                    live_view_service.mark_complete()
                    run_status.update(label="Simulated data saved - open the Charts tab", state="complete")
                else:
                    if use_imported and imported_data is None:
                        run_status.update(label="No imported data attached", state="error")
                        st.error("Upload a data file or select a saved session first.")
                    else:
                        if use_imported and imported_data is not None:
                            params_to_run["Device"] = "Offline"
                            params_to_run["data"] = imported_data
                        if connected_device is not None:
                            connected_device.prepare_for_recording()
                        run_params, saved_path = ui.run_pipeline_once(
                            params_to_run,
                            save_dir,
                            live_view=live_view_service,
                            connected_device=connected_device,
                        )
                        st.session_state["last_results"] = {
                            "label": getattr(saved_path, "name", "Last run"),
                            "params": run_params,
                            "data": run_params.get("data"),
                        }
                        live_view_service.mark_complete()
                        run_status.update(label="Measurement finished and saved - open the Charts tab", state="complete")
            except Exception as exc:  # pragma: no cover
                ui.LOGGER.exception("Measurement failed")
                run_status.update(label="Measurement failed", state="error")
                st.error(f"Measurement failed: {exc}")
            finally:
                st.session_state["_live_view_active"] = False
                st.session_state["_live_view_banner"] = None
                if connected_device is not None:
                    try:
                        connected_device.disconnect()
                    except Exception:
                        ui.LOGGER.debug("Device disconnect after measurement raised", exc_info=True)
                    st.session_state.pop("_connected_device", None)
                    st.session_state.pop("_connected_device_signature", None)
                    st.session_state["_device_check"] = (
                        "warn",
                        "Measurement finished. Connect again before the next hardware run.",
                    )

    ui.render_footer()


if __name__ == "__main__":
    if hasattr(st, "runtime") and st.runtime.exists():
        main()
    else:
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
