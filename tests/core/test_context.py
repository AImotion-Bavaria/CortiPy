"""Unit tests for ModuleContext."""

from __future__ import annotations

import numpy as np



def test_append_data_initializes_and_stacks(module_context_factory):
    """Confirms streaming EEG snippets are appended in-order so downstream analyzers see the exact timeline of the recording. This mirrors the expectation that each chunk from the amplifier extends the buffer rather than overwriting it."""
    context = module_context_factory()
    first_chunk = np.array([[1.0, 2.0], [3.0, 4.0]])
    second_chunk = np.array([[5.0, 6.0]])

    context.append_data(first_chunk)
    np.testing.assert_array_equal(context.data_buffer, first_chunk)

    context.append_data(second_chunk)
    expected = np.vstack([first_chunk, second_chunk])
    np.testing.assert_array_equal(context.data_buffer, expected)


def test_append_data_ignores_none(module_context_factory):
    """Protects the data buffer from spurious None values so accidental API calls never corrupt the live EEG trace. Clinicians can be confident that only true samples affect the record."""
    context = module_context_factory()
    context.append_data(None)
    assert context.data_buffer is None

    chunk = np.array([[1.0]])
    context.append_data(chunk)
    context.append_data(None)
    np.testing.assert_array_equal(context.data_buffer, chunk)


def test_reset_data_buffer_and_get_data(module_context_factory):
    """Verifies that flushing the acquisition buffer truly resets the signal pipeline, mimicking what an operator expects when restarting a run. After a reset, new data starts from a clean slate with no residual samples."""
    context = module_context_factory()
    assert context.get_data() is None
    context.append_data(np.ones((2, 1)))
    assert context.get_data() is not None
    context.reset_data_buffer()
    assert context.get_data() is None


def test_service_registry(module_context_factory):
    """Shows that helper services (stimulator handles, loggers, etc.) can be registered and retrieved reliably by later stages. This mirrors how the MATLAB workflow shares utilities between acquisition and evaluation steps."""
    context = module_context_factory()
    sentinel = object()
    context.attach_service("logger", sentinel)
    assert context.get_service("logger") is sentinel
    assert context.get_service("missing", "default") == "default"


def test_flush_data_to_params_updates_field(module_context_factory):
    """Demonstrates how raw EEG buffers end up in the Params block, just like clinicians expect when exporting runs from the MATLAB tool. The test guarantees that the collected samples are visible to evaluators and report generators."""
    context = module_context_factory()
    data = np.arange(6).reshape(3, 2)
    context.append_data(data)
    context.flush_data_to_params("CustomData")
    np.testing.assert_array_equal(context.params["CustomData"], data)


def test_flush_data_to_params_noop_when_buffer_empty(module_context_factory):
    """Ensures we never fabricate data: flushing with an empty buffer must leave Params untouched so reports stay honest. It proves that exporting without data will not pollute medical records."""
    context = module_context_factory()
    context.flush_data_to_params("data")
    assert "data" not in context.params


def test_release_device_disconnects_and_clears(module_context_factory):
    """Replicates the operator action of unplugging hardware, making sure the interface disconnects cleanly and the slot is freed. The context should end up hardware-free, ready for the next subject."""
    context = module_context_factory()

    class DummyDevice:
        def __init__(self) -> None:
            self.disconnected = False

        def disconnect(self) -> None:
            self.disconnected = True

    device = DummyDevice()
    context.device = device
    context.release_device()
    assert device.disconnected is True
    assert context.device is None


def test_release_device_swallows_disconnect_errors(module_context_factory):
    """Validates that a flaky device cable cannot crash the session manager; disconnect errors are absorbed and the workflow proceeds. This matches clinical reality where unplugging may fail yet the software must stay responsive."""
    context = module_context_factory()

    class FaultyDevice:
        def __init__(self) -> None:
            self.attempts = 0

        def disconnect(self) -> None:
            self.attempts += 1
            raise RuntimeError("boom")

    device = FaultyDevice()
    context.device = device
    context.release_device()
    assert device.attempts == 1
    assert context.device is None
