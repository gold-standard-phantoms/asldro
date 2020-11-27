# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2020-11-27
### Added
- Allow using an input (external) HRGT file.
- Modulation of ground truth quantities by a scale and offset value

### Fixed
- Bug that caused ground truth not to be resampled before being output

### Changed
- JSON loader filter can take any JSON schema (or none)

## [2.0.0] - 2020-11-19
### Added
- Inversion recovery MRI model
- Image container metadata
- Acquire MRI Image Filter (combines MRI Signal, transform/resampling, noise)
- Append metadata filter
- Combine time-series images filter
- Phase-magnitude filter
- CLI tool to output the HRGT files
- BIDS output
- Output of structural and ground truth data
- 1.5T and 3T HRGT files
- Allow overriding the ground truth values in the input parameter file

### Changed
- New input parameter file format (allows multi-series output)
- GE MRI signal model now allows arbitrary flip-angle

## [1.0.0] - 2020-10-16
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
- Affine matrix transformation filter (output sampling)
- README generator
- PyPI deployment
- CLI tool to output example model parameter file
