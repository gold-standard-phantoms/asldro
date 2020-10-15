# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Project repository set up with Continuous integration,
  Continuous testing (pytest, tox), Repository structure,
  Versioning information, Setup configuration,
  Deployment configuration
- Autoformatting and linting rules set up
- Pipe-and-filter architecture
- Ground truth data based on the ICBM 2009a Nonlinear Symmetric brain atlas
- NiftiImageContainer and NumpyImageContainer
- Ground truth JSON validation schema
- Filters for fft and ifft
- Complex noise filter
- Filter blocks (chains of filters)
- Parameter validation module
- GKM filter (PASL, CASL, pCASL)
- MRI signal filter
- Sphinx documentation generation (and documentation)
- Command line interface
- Affine matrix transformation filter
- README generator
- PyPI deployment
- AffineMatrixFilter accepts an affine to apply last
- User input parameter validation
