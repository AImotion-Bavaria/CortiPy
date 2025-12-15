# Security Policy

## Reporting a vulnerability
- Email joh1391@thi.de with a description, affected versions (if known), and a minimal reproduction. Please avoid filing public issues for new vulnerabilities.
- We will acknowledge reports within 5 business days and provide a remediation timeline after triage.
- For severe issues, coordinate a public disclosure date with the maintainers.

## Scope and data handling
- CortiPy is not a regulated medical device and is provided for research use; do not use it for diagnosis or patient care.
- Do not transmit participant-identifiable data in reports. Use synthetic or anonymized samples when demonstrating issues.

## Supported versions
- Latest main branch and the most recent tagged release receive fixes and advisories.

## Secure development tips
- Prefer pinned dependencies via `pyproject.toml`; avoid introducing new runtime services without review.
- Keep Streamlit UI endpoints non-public; run behind authenticated networks for clinical environments.
