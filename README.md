# Installation

## Python Version

We recommend using the latest version of Python. ASL DRO supports Python
3.7 and newer.

## Dependencies

These distributions will be installed automatically when installing ASL DRO.


* [nibabel](https://nipy.org/nibabel/) provides read / write access to some common neuroimaging file formats


* [numpy](https://numpy.org/) provides efficient calculations with arrays and matrices


* [jsonschema](https://python-jsonschema.readthedocs.io/en/stable/) provides an implementation of JSON Schema validation for Python


* [nilearn](https://nipy.org/packages/nilearn/index.html) provides image manipulation tools and statistical learning for neuroimaging data

## Virtual environments

Use a virtual environment to manage the dependencies for your project, both in
development and in production.

What problem does a virtual environment solve? The more Python projects you
have, the more likely it is that you need to work with different versions of
Python libraries, or even Python itself. Newer versions of libraries for one
project can break compatibility in another project.

Virtual environments are independent groups of Python libraries, one for each
project. Packages installed for one project will not affect other projects or
the operating system’s packages.

Python comes bundled with the `venv` module to create virtual
environments.

### Create an environment

Create a project folder and a `venv` folder within:

```
$ mkdir myproject
$ cd myproject
$ python3 -m venv venv
```

On Windows:

```
$ py -3 -m venv venv
```

### Activate the environment

Before you work on your project, activate the corresponding environment:

```
$ . venv/bin/activate
```

On Windows:

```
> venv\Scripts\activate
```

Your shell prompt will change to show the name of the activated
environment.

## Install ASL DRO

Within the activated environment, use the following command to install
ASL DRO:

```
$ pip install asldro
```

ASL DRO is now installed. Check out the Quickstart or go to the
Documentation Overview.

# Quickstart

## Overview

ASL DRO is software that can generate digital reference objects for Arterial Spin Labelling (ASL) MRI.
It creates synthetic raw ASL data according to set acquisition and data format parameters, based
on input ground truth maps for:


* Perfusion rate


* Transit time


* Intrinsic MRI parameters: M0, T1, T2, T2\*


* Tissue segmentation (defined as a single tissue type per voxel)

## Getting started

Eager to get started? This page gives a good introduction to ASL DRO.
Follow Installation to set up a project and install ASL DRO first.

After installation the command line tool `asldro` will be made available. You can run:

```
asldro generate path/to/output_file.zip
```

to run the DRO generation as-per the ASL White Paper specification. The output file may
be either .zip or .tar.gz.

Is it also possible to specify a parameter file, which will override any of the default values:

```
asldro generate --params path/to/input_params.json path/to/output_file.zip
```

It is possible to create an example parameters file containing the model defaults by running:

```
asldro output params /path/to/input_params.json
```

which will create the `/path/to/input_params.json` file. The parameters may be adjusted as
necessary and used with the ‘generate’ command. The input parameters will include, as default:

```
{
  "asl_context": "m0scan control label",
  "label_type": "pcasl",
  "label_duration": 1.8,
  "signal_time": 3.6,
  "label_efficiency": 0.85,
  "echo_time": [0.01, 0.01, 0.01],
  "repetition_time": [10.0, 5.0, 5.0],
  "rot_z": [0.0, 0.0, 0.0],
  "rot_y": [0.0, 0.0, 0.0],
  "rot_x": [0.0, 0.0, 0.0],
  "transl_x": [0.0, 0.0, 0.0],
  "transl_y": [0.0, 0.0, 0.0],
  "transl_z": [0.0, 0.0, 0.0],
  "acq_matrix": [64, 64, 12],
  "acq_contrast": "se",
  "desired_snr": 10.0,
  "random_seed": 0
}
```

The parameters may be adjusted as necessary. The parameter asl_context defines the number of
simulated acquisition volumes that should be generated.  The following array parameters need to
have the same number of entries as there are defined volumes:


* `echo_time`


* `repetition_time`


* `rot_z`


* `rot_y`


* `rot_x`


* `transl_x`


* `transl_y`


* `transl_z`

For more details on input parameters see Parameters

## Pipeline details

The DRO currently runs using the default ground truth.
Future releases will allow this to be configured.  The pipeline comprises of:


1. Loading in the ground truth volumes.


2. Producing $\Delta M$ using the General Kinetic Model for the specified ASL parameters.


3. Generating synthetic M0, Control and Label volumes.


4. Applying motion


5. Sampling at the acquisition resolution


6. Adding instrument and physiological pseudorandom noise.

Each volume described in `asl_context` has the motion, resampling and noise processes applied
independently. The rotation and translation arrays in the input parameters describe this motion, and
the the random number generator is initialised with the same seed each time the DRO is run, so each
volume will have noise that is unique, but statistically the same.

If `desired_snr` is set to `0`, the resultant images will not have any noise applied.

Once the pipeline is run, the following images are created:


* Timeseries of magnitude ASL volumes in accordance with `asl_context` (asl_source_magnitude.nii.gz)


* Ground truth perfusion rate, resampled to `acq_matrix` (gt_cbf_acq_res_nii.gz)


* Ground truth tissue segmentation mask, resampled to `acq_matrix` (gt_labelmask_acq_res.nii.gz)

The DRO pipeline is summarised in this schematic (click to view full-size):



![image](docs/images/asldro.png)

# Development

Development of this software project must comply with a few code styling/quality rules and processes:


* Pylint must be used as the linter for all source code. A linting configuration can be found in `.pylintrc`. There should be no linting errors when checking in code.


* Before committing any files, [black](https://black.readthedocs.io/en/stable/) must be run with the default settings in order perform autoformatting on the project.


* Before pushing any code, make sure the CHANGELOG.md is updated as per the instructions in the CHANGELOG.md file.


* The project’s software development processes must be used ([found here](https://confluence.goldstandardphantoms.com/display/AD/Software+development+processes)).
