# ASLDRO


## Master
[![Build status](https://git.goldstandardphantoms.com/GSP/software/asldro/badges/master/build.svg)](https://git.goldstandardphantoms.com/GSP/software/asldro/commits/master)
Coverage [![Overall test coverage](https://git.goldstandardphantoms.com/GSP/software/asldro/badges/master/coverage.svg)](https://git.goldstandardphantoms.com/GSP/software/asldro/pipelines)

## Develop
[![Build status](https://git.goldstandardphantoms.com/GSP/software/asldro/badges/develop/build.svg)](https://git.goldstandardphantoms.com/GSP/software/asldro/commits/develop)
Coverage [![Overall test coverage](https://git.goldstandardphantoms.com/GSP/software/asldro/badges/develop/coverage.svg)](https://git.goldstandardphantoms.com/GSP/software/asldro/pipelines)


## Description

ASL Digital Reference Object


## Set up

Set up the python `virtualenv` and install requirements

    virtualenv env -p python3 --clear
    source env/bin/activate
    pip install -r requirements/dev.txt --upgrade


## How to run

TODO...


## How to develop

- Pylint must be used as the linter for this project. A linting configuration can be found in .pylintrc
- Before committing any files, `black` must be run with the default settings in order perform autoformatting on the project
