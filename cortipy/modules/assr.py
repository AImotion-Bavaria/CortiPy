"""ASSR module placeholder."""

from __future__ import annotations

import numpy as np

from cortipy.core.context import ModuleContext
from cortipy.evaluation.assr import AssrEvaluator
from cortipy.shared.notifications import info_end_live, info_start_live

from .base import ModuleBase


class AssrModule(ModuleBase):
    def __init__(self, evaluator: AssrEvaluator | None = None) -> None:
        super().__init__("ASSR", ["ASSR"], evaluator or AssrEvaluator())
        self.time_step = 5.0

    def collect_measurements(self, context: ModuleContext) -> None:
        params = context.params
        param_block = params.setdefault("Parameters", {})
        recording_time = float(param_block.get("RecordingTime", 60))
        device = self.require_device(context)

        info_start_live()
        aux_ch = param_block.get("NumberAUXChannels", 0) if params.get("Device") == "ActiCHamp" else 0
        data = []
        elapsed = 0.0
        while elapsed < recording_time:
            duration = min(self.time_step, recording_time - elapsed)
            chunk = np.asarray(device.acquire(duration, aux_ch), dtype=float)
            if chunk.ndim == 1:
                chunk = chunk[:, np.newaxis]
            data.append(chunk)
            elapsed += duration
        info_end_live()

        if data:
            buffer = np.vstack(data)
            context.data_buffer = buffer
            params["data"] = buffer
