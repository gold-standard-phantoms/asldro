#!/bin/bash

sphinx-build -b markdown -a . _md
cat _md/usage/installation.md <(echo) _md/usage/quickstart.md <(echo) _md/usage/development.md > ../README.md
sed  -i 's/images\//docs\/images\//' ../README.md

