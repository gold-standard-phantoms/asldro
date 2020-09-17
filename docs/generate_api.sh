#!/bin/bash

sphinx-apidoc -f -o _api/ ../src/ "../src/**/test_*.py"
