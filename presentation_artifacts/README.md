# CortiPy EMBC demo - generated artifacts

This folder contains `EMBC_CortiPy_public_demo.ipynb` and its generated demo artifacts.
Re-running the notebook recreates all data from randomly generated EEG.

## How to run
1. Install deps:  `pip install -e .[ui,bids]`  (numpy, pandas, matplotlib, mne, pyarrow, pyedflib)
2. Open `presentation_artifacts/EMBC_CortiPy_public_demo.ipynb` and **Run All**.
3. Figures render inline and export here to `figures/` as PDF + SVG + PNG@300 + PNG@600.

The notebook creates all data and directories itself. The final cell tidies this folder; set
`CLEAN_ARTIFACTS = False` in the setup cell (or env `CORTIPY_CLEAN=false`) to keep the exported figures.
