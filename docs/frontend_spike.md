# Frontend spike: Streamlit vs. alternatives for the CortiPy UI

Status: spike / decision aid. The **cortipy pipeline (`cortipy/`) is frontend-agnostic** — any option
below reuses `MeasurementPipeline`, `DeviceFactory`, the evaluators, and the BIDS/SBIDS I/O unchanged.
Only `apps/streamlit_app.py` and `cortipy/ui_streamlit/` would be replaced in a migration.

## 1. Observed pain points (first-hand, this session)

These are concrete, reproduced issues — not hypotheticals:

| # | Pain point | Root cause (Streamlit model) |
|---|---|---|
| P1 | **Every click reruns the Streamlit script top-to-bottom** ("buttons not fluid"). | Streamlit's core execution model: any interaction re-executes the app entrypoint end-to-end. The implementation is now split into focused modules, but reruns are still the framework model. |
| P2 | **Electrode table "disappears and reappears"** on fill / impedance read / load. | `st.data_editor` is a *keyed* widget; to reflect programmatic data changes you must change its `key`, which **remounts** the component. There is no in-place "set value" API. |
| P3 | **Loaded/imported settings silently didn't apply.** | Keyed widgets ignore `value=`/`index=` once instantiated; you must mutate `st.session_state[key]` or clear it, then `st.rerun()`. Fixed this session, but it's a recurring footgun. |
| P4 | **Live plots during a blocking acquisition** need placeholder gymnastics. | A synchronous `run()` blocks the single script thread; live updates go through cached `st.empty()` placeholders and a manual service. |
| P5 | Headless/automated testing is fragile. | `streamlit.testing.v1.AppTest` can't serialize some widget states (hit a multiselect quirk repeatedly), so CI can only really assert "boots without exception." |

Streamlit's strengths remain real: fastest to build, great defaults, one language, rich widgets,
and the app already exists and works.

## 2. Options

| Framework | Model | Fixes P1/P2? | Effort to port | Notes |
|---|---|---|---|---|
| **Streamlit (stay)** | Full-script rerun | No (mitigate with `st.fragment`, `@st.cache_*`, splitting the file) | 0 (done) | `st.fragment` (1.37+) scopes reruns to a region — meaningfully helps P1/P4. Doesn't fix P2. |
| **NiceGUI** | Event-driven, per-component state (Vue/Quasar under the hood, pure-Python) | **Yes** — no full rerun; components update in place; `ui.aggrid`/`ui.table` edit without remount | Medium | Python-only, async-native (clean fit for P4 live streaming via `ui.timer`/websockets). Closest "Streamlit-like DX without the rerun tax." |
| **Dash (Plotly)** | Callback graph | Yes (targeted callbacks) | Medium-High | Verbose callback wiring; excellent plotting; heavier mental model. |
| **FastAPI + React/Svelte** | API + SPA | Yes (full control) | High | Best ceiling for fluidity/real-time; two languages; most build/maintenance. Overkill unless the UI becomes a product. |

## 3. Recommendation

**Short term: stay on Streamlit.** The fixes shipped this session (run-status `st.status`, widget-state
sync on load, diagnostics panel, quick-fill, device-aware `fs`) address most of the friction. Two cheap,
high-impact Streamlit-native follow-ups that don't need a migration:

- **Wrap heavy regions in `st.fragment`** (esp. the electrode editor and live-view) so interacting with
  them reruns *only that region*, not the whole app entrypoint - directly targets P1/P4.
- **Finish the modular split of `app.py`** (`state.py`, `forms.py`, `electrodes.py`, `charts.py`,
  `run_control.py`) so reruns import less and the code is maintainable. Pure refactor; do it as its own
  reviewed PR (see §5).

**Migrate only if** P1 (click fluidity) or P2 (editable-table remount) remain dealbreakers after the
above. In that case **NiceGUI is the target**: Python-only, event-driven (no full rerun), in-place
component updates (kills P2), and async live streaming (clean P4). The pipeline is reused as-is.

## 4. NiceGUI proof-of-concept (sketch — not yet run; NiceGUI not installed here)

Demonstrates the two things Streamlit can't do cleanly: instant button feedback with **no full rerun**,
and an **editable electrode grid that updates in place** (no remount/flicker).

```python
# apps/nicegui_spike.py  — run: python apps/nicegui_spike.py  (needs: pip install nicegui)
from nicegui import ui
from cortipy.core.pipeline import PipelineHooks   # pipeline reused unchanged
from cortipy import MeasurementPipeline

state = {"method": "SSVEP", "device": "Dummy", "fs": 250,
         "channels": [{"name": f"Ch{i+1}", "pos": p, "active": True}
                      for i, p in enumerate(["Fp1","Fp2","F3","F4","C3","C4","P3","P4"])]}

with ui.row():
    ui.select(["Alpha","SSVEP","ASSR","VEP","P300","BERA"], label="Method").bind_value(state, "method")
    ui.select(["ActiCHamp","UNICORN","Dummy"], label="Device").bind_value(state, "device")
    ui.select([250,500,1000,2000], label="fs (Hz)").bind_value(state, "fs")

# Editable grid: edits mutate rows in place — NO remount, NO flicker (fixes P2).
grid = ui.aggrid({
    "columnDefs": [{"field": "name", "editable": False},
                   {"field": "pos", "editable": True},
                   {"field": "active", "editable": True}],
    "rowData": state["channels"],
})
def fill_montage():
    for row, pos in zip(state["channels"], ["Fp1","Fpz","Fp2","F3","Fz","F4","C3","Cz"]):
        row["pos"] = pos; row["active"] = True
    grid.update()                      # in-place refresh, no remount
ui.button("Fill standard positions", on_click=fill_montage)

status = ui.label("Idle")              # updates instantly, no page rerun (fixes P1)
async def run_measurement():
    status.text = "🔴 Measurement running…"
    params = {"Method": state["method"], "Device": state["device"],
              "Parameters": {"fs": state["fs"], "RecordingTime": 5, "NumberEEGChannels": len(state["channels"])},
              "Channels": [{"Position": c["pos"]} for c in state["channels"] if c["active"]]}
    hooks = PipelineHooks(params_provider=lambda prev: params if prev is None else None,
                          should_continue=lambda _: False)
    await ui.run_javascript("void 0")  # yield to UI
    MeasurementPipeline(hooks=hooks).run()   # (wrap blocking call in run.io_bound in real code)
    status.text = "✅ Measurement finished"
ui.button("Start measurement", on_click=run_measurement, color="teal")

ui.run(title="cortipy (NiceGUI spike)", port=8502)
```

Live streaming during acquisition (P4) is a `ui.timer(0.1, push_latest_window)` reading the same
device buffer the Streamlit `LiveViewService` uses — no placeholder juggling.

## 5. Suggested next steps (each a small, reviewable PR)

1. `st.fragment` the electrode editor + live-view region (Streamlit, low risk, high fluidity win).
2. Modularize `app.py` into `cortipy/ui_streamlit/{state,forms,electrodes,charts,run_control}.py`
   (pure refactor; verify with `AppTest` boot + a smoke click per page).
3. Stand up `apps/nicegui_spike.py` from §4, click-test fluidity/electrode-edit head-to-head, then decide.
