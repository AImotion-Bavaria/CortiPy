GitHub Copilot Chat Assistant

# CortiPy Test Coverage Map

| Component | Description |
| :--- | :--- |
| **Core** |  |
| &nbsp;&nbsp;ModuleContext (`core/context.py`) | Buffer management (`append_data`, `reset_data_buffer`, `flush_data_to_params`), service registry, and device release flows including error swallowing. (Covered by: `tests/core/test_context.py`) |
| &nbsp;&nbsp;MeasurementPipeline (`core/pipeline.py`) | `run_once` and `run` orchestration with tracking modules, hooks interaction, and default module ordering. (Covered by: `tests/core/test_pipeline.py`) |
| **Devices** |  |
| &nbsp;&nbsp;DeviceFactory (`devices/factory.py`) | LSL, dummy, and offline creation paths plus unknown-device error handling. (Covered by: `tests/devices/test_factory.py`) |
| &nbsp;&nbsp;DummyDevice (`devices/dummy.py`) | Connect requirement enforcement and correct EEG+AUX sample generation. (Covered by: `tests/devices/test_devices.py`) |
| &nbsp;&nbsp;OfflineDevice (`devices/offline.py`) | `.npy` file playback, timed chunking, and sequential cursor updates. (Covered by: `tests/devices/test_devices.py`) |
| &nbsp;&nbsp;LSLDevice (`devices/lsl.py`) | Mocked pylsl resolution, inlet management, acquisition looping, and disconnect cleanup. (Covered by: `tests/devices/test_devices.py`) |
| **Shared Helpers** |  |
| &nbsp;&nbsp;Signal utilities (`shared/signal.py`) | FFT frequency range validation and `time_vector` second/millisecond conversions. (Covered by: `tests/shared/test_signal.py`) |
| **Module Base** |  |
| &nbsp;&nbsp;ModuleBase (`modules/base.py`) | Lifecycle gating (`should_run`, `connect`, `collect`, `execute`), evaluator hooks, and device management (`ensure_device`, `require_device`). (Covered by: `tests/modules/test_base_module.py`) |
| **Acquisition Modules** |  |
| &nbsp;&nbsp;AlphaModule | Live loop, FFT plotting triggers, `_ensure_array`, and multi-device referencing rules. (Covered by: `tests/modules/test_alpha_module.py`) |
| &nbsp;&nbsp;AssrModule | Recording-time chunking loop and AUX-channel handling for non-ActiCHamp devices. (Covered by: `tests/modules/test_assr_module.py`) |
| &nbsp;&nbsp;BeraModule | ActiCHamp guard, fs requirement, live metric pipeline, `_apply_reference`, and plotting metrics accumulation. (Covered by: `tests/modules/test_bera_module.py`) |
| &nbsp;&nbsp;BciModule | Parameter normalization, windowing, referencing, Bluetooth success/failure paths, spectrum plotting, and evaluation payloads. (Covered by: `tests/modules/test_bci_module.py`) |
| &nbsp;&nbsp;P300Module | Sampling-rate guard, data loop, referencing, and `_update_live_plot` segmentation/plotting logic. (Covered by: `tests/modules/test_p300_module.py`) |
| &nbsp;&nbsp;SsvepModule | Acquisition loop, referencing per device, and FFT plotting calls. (Covered by: `tests/modules/test_ssvep_module.py`) |
| &nbsp;&nbsp;VepModule | Sampling-rate guard, data loop, AUX resolution, referencing, and live plotting updates. (Covered by: `tests/modules/test_vep_module.py`) |
| **Evaluators** |  |
| &nbsp;&nbsp;AlphaEvaluator | Method gating, data/fs validation, and evaluation payload creation (alpha power, PSD, stats, SNR). (Covered by: `tests/evaluation/test_alpha_evaluator.py`) |
| &nbsp;&nbsp;AssrEvaluator | Method gating, data/fs validation, and ipsi/contra FFT/PSD/SNR/f-test metrics with plotting suppressed. (Covered by: `tests/evaluation/test_assr_evaluator.py`) |
| &nbsp;&nbsp;VepEvaluator | Method gating, data/fs validation, average-signal generation, peak stats, SNR/residual-noise metrics, and plotting hooks. (Covered by: `tests/evaluation/test_vep_evaluator.py`) |
| **Evaluation Helpers** |  |
| &nbsp;&nbsp;Shared plotting hooks | Covered implicitly via monkeypatch spies in module/evaluator tests; all live plot calls are patched/spied to assert they execute with expected parameters (e.g., alpha FFT, ASSR spectrum, BCI PSD, VEP charts). |
| **Bluetooth/Notifications** |  |
| &nbsp;&nbsp;BCI Bluetooth helper | Ensures Bluetooth connections fail gracefully when unavailable and succeed with mocked sockets, exercising cleanup hooks. (Covered by: `tests/modules/test_bci_module.py::test_try_open_bluetooth_handles_missing_dependency/success`) |
