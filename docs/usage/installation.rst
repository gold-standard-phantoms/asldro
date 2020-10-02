Installation
============


Python Version
--------------

We recommend using the latest version of Python. ASL DRO supports Python
3.7 and newer.

Dependencies
------------

These distributions will be installed automatically when installing Flask.

* `nibabel`_ provides read / write access to some common neuroimaging file formats
* `numpy`_ provides efficient calculations with arrays and matrices
* `jsonschema`_ provides an implementation of JSON Schema validation for Python
* `nilearn`_ provides image manipulation tools and statistical learning for neuroimaging data

.. _nibabel: https://nipy.org/nibabel/
.. _numpy: https://numpy.org/
.. _jsonschema: https://python-jsonschema.readthedocs.io/en/stable/
.. _nilearn: https://nipy.org/packages/nilearn/index.html

Virtual environments
--------------------

Use a virtual environment to manage the dependencies for your project, both in
development and in production.

What problem does a virtual environment solve? The more Python projects you
have, the more likely it is that you need to work with different versions of
Python libraries, or even Python itself. Newer versions of libraries for one
project can break compatibility in another project.

Virtual environments are independent groups of Python libraries, one for each
project. Packages installed for one project will not affect other projects or
the operating system's packages.

Python comes bundled with the :mod:`venv` module to create virtual
environments.


.. _install-create-env:

Create an environment
~~~~~~~~~~~~~~~~~~~~~

Create a project folder and a :file:`venv` folder within:

.. code-block:: sh

    $ mkdir myproject
    $ cd myproject
    $ python3 -m venv venv

On Windows:

.. code-block:: bat

    $ py -3 -m venv venv


.. _install-activate-env:

Activate the environment
~~~~~~~~~~~~~~~~~~~~~~~~

Before you work on your project, activate the corresponding environment:

.. code-block:: sh

    $ . venv/bin/activate

On Windows:

.. code-block:: bat

    > venv\Scripts\activate

Your shell prompt will change to show the name of the activated
environment.


Install ASL DRO
---------------

Within the activated environment, use the following command to install
ASL DRO:

.. code-block:: sh

    $ pip install asldro

ASL DRO is now installed. Check out the :doc:`quickstart` or go to the
:doc:`Documentation Overview </index>`.

