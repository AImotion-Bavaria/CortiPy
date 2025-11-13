# CortiPy Test Coverage Map

| Component | Covered By | Description of Coverage |
| :--- | :--- | :--- |
| **Core** |  |  |
| &nbsp;&nbsp;ModuleContext (`core/context.py`) | `tests/core/test_context.py` | Buffer management (`append_data`, `reset_data_buffer`, `flush_data_to_params`), service registry, and device release flows including error swallowing. |
| &nbsp;&nbsp;MeasurementPipeline (`core/pipeline.py`) | `tests/core/test_pipeline.py` | `run_once` and `run` orchestration with tracking modules, hooks interaction, and default module ordering. |
| **Devices** |  |  |
| &nbsp;&nbsp;DeviceFactory (`devices/factory.py`) | `tests/devices/test_factory.py` | LSL, dummy, and offline creation paths plus unknown-device error handling. |
| &nbsp;&nbsp;DummyDevice (`devices/dummy.py`) | `tests/devices/test_devices.py` | Connect requirement enforcement and correct EEG+AUX sample generation. |
| &nbsp;&nbsp;OfflineDevice (`devices/offline.py`) | `tests/devices/test_devices.py` | `.npy` file playback, timed chunking, and sequential cursor updates. |
| &nbsp;&nbsp;LSLDevice (`devices/lsl.py`) | `tests/devices/test_devices.py` | Mocked pylsl resolution, inlet management, acquisition looping, and disconnect cleanup. |
| **Shared Helpers** |  |  |
| &nbsp;&nbsp;Signal utilities (`shared/signal.py`) | `tests/shared/test_signal.py` | FFT frequency range validation and `time_vector` second/millisecond conversions. |
| **Module Base** |  |  |
| &nbsp;&nbsp;ModuleBase (`modules/base.py`) | `tests/modules/test_base_module.py` | Lifecycle gating (`should_run`, `connect`, `collect`, `execute`), evaluator hooks, and device management (`ensure_device`, `require_device`). |
| **Acquisition Modules** |  |  |
| &nbsp;&nbsp;AlphaModule | `tests/modules/test_alpha_module.py` | Live loop, FFT plotting triggers, `_ensure_array`, and multi-device referencing rules. |
| &nbsp;&nbsp;AssrModule | `tests/modules/test_assr_module.py` | Recording-time chunking loop and AUX-channel handling for non-ActiCHamp devices. |
| &nbsp;&nbsp;BeraModule | `tests/modules/test_bera_module.py` | ActiCHamp guard, fs requirement, live metric pipeline, `_apply_reference`, and plotting metrics accumulation. |
| &nbsp;&nbsp;BciModule | `tests/modules/test_bci_module.py` | Parameter normalization, windowing, referencing, Bluetooth success/failure paths, spectrum plotting, and evaluation payloads. |
| &nbsp;&nbsp;P300Module | `tests/modules/test_p300_module.py` | Sampling-rate guard, data loop, referencing, and `_update_live_plot` segmentation/plotting logic. |
| &nbsp;&nbsp;SsvepModule | `tests/modules/test_ssvep_module.py` | Acquisition loop, referencing per device, and FFT plotting calls. |
| &nbsp;&nbsp;VepModule | `tests/modules/test_vep_module.py` | Sampling-rate guard, data loop, AUX resolution, referencing, and live plotting updates. |
| **Evaluators** |  |  |
| &nbsp;&nbsp;AlphaEvaluator | `tests/evaluation/test_alpha_evaluator.py` | Method gating, data/fs validation, and evaluation payload creation (alpha power, PSD, stats, SNR). |
| &nbsp;&nbsp;AssrEvaluator | `tests/evaluation/test_assr_evaluator.py` | Method gating, data/fs validation, and ipsi/contra FFT/PSD/SNR/f-test metrics with plotting suppressed. |
| &nbsp;&nbsp;VepEvaluator | `tests/evaluation/test_vep_evaluator.py` | Method gating, data/fs validation, average-signal generation, peak stats, SNR/residual-noise metrics, and plotting hooks. |
| **Evaluation Helpers** |  |  |
| &nbsp;&nbsp;Shared plotting hooks | Covered implicitly via monkeypatch spies in module/evaluator tests | All live plot calls are patched/spied to assert they execute with expected parameters (e.g., alpha FFT, ASSR spectrum, BCI PSD, VEP charts). |
| **Bluetooth/Notifications** |  |  |
| &nbsp;&nbsp;BCI Bluetooth helper | `tests/modules/test_bci_module.py::test_try_open_bluetooth_handles_missing_dependency/success` | Ensures Bluetooth connections fail gracefully when unavailable and succeed with mocked sockets, exercising cleanup hooks. |
