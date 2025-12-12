# Contributing to CortiPy

Thank you for improving CortiPy. This project supports EEG acquisition and analysis workflows for research and clinical settings. Please review the expectations below before opening an issue or pull request.

## Ground rules
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
- Do not share or commit identifiable participant data. Use synthetic or anonymized fixtures.
- Keep changes focused and well-described; prefer smaller PRs over large mixed updates.
- Include tests for new features and fixes when feasible; mark hardware-dependent tests as skipped by default.

## Getting started
1) Clone and create an environment:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[ui,bids,sbids]  # full stack including UI and BIDS/SBIDS extras
```
2) Run the suite without hardware:
```bash
pytest
```
3) For UI development: `streamlit run apps/streamlit_app.py`

## Coding guidelines
- Python 3.9–3.12; follow PEP8/PEP257; prefer type hints.
- Keep public APIs documented; add docstrings for new modules/functions.
- Add comments only where code is non-obvious (performance, clinical rationale, device quirks).
- Update docs/README when behavior or interfaces change.

## Commits and reviews
- Use clear messages (scope: summary). Reference issues when applicable.
- Include release notes or changelog entry for user-facing changes.
- Expect CI to run linting, tests, and packaging checks before merge.

## Versioning
- We follow semantic versioning (MAJOR.MINOR.PATCH).
- Single source of truth: `cortipy/__init__.py`.
- To bump versions and sync metadata, run: `python scripts/bump_version.py --part patch|minor|major` (updates `__version__` and `CITATION.cff`).

## Maintainer contacts
- General/project: joh1391@thi.de
- Academic collaborations: Rahul.Mondal@thi.de or Laurens.Kreilinger@thi.de
- Conduct/abuse reports: joh1391@thi.de

We appreciate your contributions—thank you for helping keep CortiPy reliable for the community.
