"""Shared helpers exposed by cortipy."""

from .assr import assr_f_test, calc_snr as assr_calc_snr, compute_psd as assr_compute_psd
from .bids import BIDSLoader, BIDSLoadResult, ExperimentBinLoader
from .dataset import CortiDataset
from .bera import (
    avg_seg_avg,
    block_weighted,
    filter_bera,
    get_fsp_fmp,
    prepro,
    residual_noise,
    residual_noise_eclipse,
    wave_amplitude,
)
from .filtering import filter_vep
from .notifications import beep, info_end_live, info_start_live
from .plotting import (
    plot_fft_live,
    plot_live_avg_bera,
    plot_live_avg_vep,
    plot_bera_results,
    plot_cortipy_topomap,
    plot_live_erp,
    plot_p300_results,
    plot_assr_spectrum,
)
from .segmentation import seg_sig_fast, seg_sig_fast_p300
from .signal import calc_fft, hann_window, time_vector
from .ssvep import (
    cca_correlations,
    classify_fft,
    compute_t2circ,
    describe_prediction,
    get_frequency_indices,
    plot_psd_ssvep,
    send_prediction_bt,
    ssvep_f_test,
    ssvep_snr,
)
from .sbids import (
    SBIDSLoader,
    SbidsExporter,
    build_sbids_document_for_dataset,
    default_output_path as sbids_default_output_path,
    export_dataset as export_sbids_dataset,
    read_sbids,
    to_sbids,
)
from .triggers import trigger_adc

__all__ = [
    "BIDSLoader",
    "BIDSLoadResult",
    "ExperimentBinLoader",
    "CortiDataset",
    "SBIDSLoader",
    "SbidsExporter",
    "build_sbids_document_for_dataset",
    "sbids_default_output_path",
    "export_sbids_dataset",
    "read_sbids",
    "to_sbids",
    "calc_fft",
    "hann_window",
    "time_vector",
    "seg_sig_fast",
    "seg_sig_fast_p300",
    "plot_fft_live",
    "plot_live_avg_vep",
    "plot_live_avg_bera",
    "plot_bera_results",
    "plot_cortipy_topomap",
    "plot_live_erp",
    "plot_p300_results",
    "plot_assr_spectrum",
    "plot_psd_ssvep",
    "filter_vep",
    "filter_bera",
    "prepro",
    "block_weighted",
    "avg_seg_avg",
    "get_fsp_fmp",
    "residual_noise",
    "residual_noise_eclipse",
    "wave_amplitude",
    "assr_compute_psd",
    "assr_calc_snr",
    "assr_f_test",
    "get_frequency_indices",
    "classify_fft",
    "compute_t2circ",
    "cca_correlations",
    "send_prediction_bt",
    "describe_prediction",
    "ssvep_snr",
    "ssvep_f_test",
    "trigger_adc",
    "beep",
    "info_start_live",
    "info_end_live",
]
