"""SBIDS loader/exporter integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cortipy.shared.sbids import SBIDSLoader, export_dataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_sbids_loader_reads_fixture_npz():
    """SBIDSLoader should resolve the fixture JSON-LD and load the NPZ data."""
    root = _repo_root()
    sbids_path = root / "sbids" / "sbids_meta_20251113-164440_SSVEP.jsonld"
    if not sbids_path.exists():
        pytest.skip("SBIDS fixture not available in this checkout.")
    loader = SBIDSLoader(root / "sbids", data_roots=[root / "cortipy_runs"])

    result = loader.read_sbids(meta_path=sbids_path)

    assert result.data.shape == (6500, 16)
    assert result.sampling_rate == pytest.approx(250.0)
    assert result.channels is not None
    assert len(result.channels) == result.data.shape[1]
    assert result.metadata.get("sbids", {}).get("dataset_id") == "20251113-164440_SSVEP"


def test_sbids_export_and_reload(tmp_path):
    """Export params/data to SBIDS JSON-LD and reload through the loader."""
    dataset_dir = tmp_path / "run"
    dataset_dir.mkdir()
    data = np.arange(0.0, 1.0, 0.01)[:, None]  # 100 samples, 1 channel
    np.savez(dataset_dir / "data.npz", data=data)
    params = {
        "Method": "Dummy",
        "Device": "Simulated",
        "Parameters": {
            "fs": 100.0,
            "RecordingTime": float(len(data)) / 100.0,
            "NumberEEGChannels": 1,
            "NumberAUXChannels": 0,
            "Filename": "data",
        },
        "Channels": [{"Channel": "Cz", "Position": "Cz", "Active": True}],
    }
    (dataset_dir / "params.json").write_text(json.dumps(params, indent=2))

    output = tmp_path / "sbids_meta_run.jsonld"
    written = export_dataset(
        dataset_dir=dataset_dir,
        dataset_id="RUN01",
        dataset_name="run",
        output=output,
        indent=2,
    )

    assert written == 1
    assert output.exists()
    loader = SBIDSLoader(tmp_path, data_roots=[dataset_dir.parent])
    result = loader.read_sbids(meta_path=output)

    assert result.data.shape == data.shape
    np.testing.assert_allclose(result.data, data)
    assert result.sampling_rate == pytest.approx(100.0)
