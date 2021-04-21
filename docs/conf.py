# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pprint import pformat


import sphinx.ext.apidoc

sys.path.append(os.path.join(os.path.dirname(__name__), "../src"))


# -- Project information -----------------------------------------------------

project = "ASL DRO"
copyright = "2020, 2021, Gold Standard Phantoms"
author = "Tom Hampshire, Aaron Oliver-Taylor"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc", "sphinx.ext.mathjax", "sphinx_rtd_theme"]

# This value contains a list of modules to be mocked up.
# This is useful when some external dependencies are not met at build time
# and break the building process. You may only specify the root package of
# the dependencies themselves and omit the sub-modules:
autodoc_mock_imports = ["numpy", "nibabel", "jsonschema", "nilearn", "scipy"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]


language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

todo_include_todos = True


def run_apidoc(_):
    """ A hook to run on documentation building which will
        first generate the API stubs for the Sphinx build """

    sys.path.append(os.path.join(os.path.dirname(__file__)))
    current_dir = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(current_dir, "_api")
    source_dir = os.path.join(current_dir, "..", "src")
    exclude_pattern = "../src/**/test_*.py"
    sphinx.ext.apidoc.main(["-f", "-o", output_dir, source_dir, exclude_pattern])


def setup(app):
    """ Hook the apidoc generation on build """
    app.connect("builder-inited", run_apidoc)
