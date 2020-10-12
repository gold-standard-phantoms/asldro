# Installation

## Python Version

We recommend using the latest version of Python. ASL DRO supports Python
3.7 and newer.

## Dependencies

These distributions will be installed automatically when installing Flask.


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

Eager to get started? This page gives a good introduction to ASL DRO.
Follow Installation to set up a project and install ASL DRO first.

After installation the command line tool asldro will be made available. You can run:

```
asldro path/to/input_params.json path/to/output_file.zip
```

The output file may be either .zip or .tar.gz. The input parameters file must currently include, at minimum.

```
{
  "asl_context_array": "m0scan m0scan control label",
  "label_type": "pCASL",
}
```

The parameters may be adjusted as necessary.
# Development

Development of this software project must comply with a few code styling/quality rules and processes:


* Pylint must be used as the linter for all source code. A linting configuration can be found in `.pylintrc`. There should be no linting errors when checking in code.


* Before committing any files, [black](https://black.readthedocs.io/en/stable/) must be run with the default settings in order perform autoformatting on the project.


* Before pushing any code, make sure the CHANGELOG.md is updated as per the instructions in the CHANGELOG.md file.


* The project’s software development processes must be used ([found here](https://confluence.goldstandardphantoms.com/display/AD/Software+development+processes)).
