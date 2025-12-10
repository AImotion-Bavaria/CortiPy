## Synthetic Data Gaps for Exact Reproduction (Experiment 1)

To make the CortiPy plots numerically match the EEGLAB reference PDFs, the following metadata/events need to be saved alongside each synthetic dataset. Right now the BIN/params.json files lack this information, so the loader must guess.

### Oddball (synData/Oddball)
- Save the randomized marker sequence (length 300, values 1=standard, 2=target) to disk (e.g., `markers.json`, `markers.mat`, or embed in `params.json`).
- Optionally add a stim channel with impulses at each trial onset carrying the condition code.
- Record trial length (600 ms), sample rate (1 kHz), and total trials (300) explicitly in `params.json`.

### VEP (synData/VEP)
- Save trial count (100), epoch length (500 ms), sample rate (1 kHz) in `params.json`.
- Provide event onsets (trial boundaries); optionally add a stim channel so epochs can be built deterministically (instead of guessing fixed windows).

### ABR (synData/ABR)
- Current data matches the script (200 trials × 15 ms, fs=25 kHz). No extra markers needed, but keep fs/trial count/epoch length in `params.json` for clarity.

### ASSR (synData/ASSR) and SSVEP (synData/SSVEP)
- Continuous (single 5 s epoch, fs=1 kHz). No events required, but note stimulus frequency/amplitude and noise settings in `params.json` for reference.

### How to export from MATLAB scripts
- After you build `markers` (e.g., in Oddball), write it to disk:  
  `save(fullfile(outputFolder,'markers.mat'),'markers');`  
  or JSON/CSV equivalent.
- To add a stim channel, create a vector of length nSamples with impulses at trial onsets carrying the event code, and append it as an extra channel before saving the BIN.
- Ensure `params.json` includes: `fs`, `epoch_len_ms`, `n_trials`, and (if applicable) `markers_path` or an inline `markers` array.

With real markers/events and explicit trial metadata available, the Experiment 1 loader can drop heuristics (e.g., peak-based target detection) and produce plots that overlay with the EEGLAB references.***
