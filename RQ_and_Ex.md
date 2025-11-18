# Research Question
- Does using the CortiPy pipeline with NPZ data reduce loading and preprocessing time compared to EEGLAB BIDS datasets?
- Can a large language model (LLM) provide meaningful, interactive answers about EEG results when fed structured evaluation outputs from CortiPy
  - Before starting the measurement. is Z good, alpha power good when eyes closed/open
  - After evaluation re metrics and plots as expected...
# Experiment
- We compare the time to load and preprocess the P300 EEG dataset (ds003061) using the CortiPy pipeline with NPZ files versus standard EEGLAB BIDS processing.
- Generating artifical datasets. Adding clear flaw like high impedance, wrong position of electrode and check output of LLM
