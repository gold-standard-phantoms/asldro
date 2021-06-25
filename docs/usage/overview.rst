Overview
--------

.. image:: images/asldro_logo.png

.. image:: https://notebooks.gesis.org/binder/badge_logo.svg
 :target: https://notebooks.gesis.org/binder/v2/gh/gold-standard-phantoms/asldro/develop?filepath=asldro_example.ipynb
.. image:: https://shields.io/pypi/l/asldro.svg
 :target: https://pypi.org/project/asldro/

ASL DRO is software that can generate digital reference objects for Arterial Spin Labelling (ASL) MRI.
It creates synthetic raw ASL data according to set acquisition and data format parameters, based
on input ground truth maps for:

* Perfusion rate
* Transit time
* Intrinsic MRI parameters: M0, T1, T2, T2*
* Tissue segmentation (defined as a single tissue type per voxel)

Synthetic data is generated in Brain Imaging Data Structure format, comprising of a NIFTI image file
and accompanying json sidecar containing parameters.

ASLDRO was developed to address the need to test ASL image processing pipelines with data that has
a known ground truth. A strong emphasis has been placed on ensuring traceability of the developed
code, in particular with respect to testing.  The DRO pipelines uses a 'pipe and filter' architecture
with 'filters' performing data processing, which provides a common interface between processing
blocks.

How To Cite
~~~~~~~~~~~~

If you use ASLDRO in your work, please include the following citation

.. bibliography::
    :filter: False
    :style: unsrt

    OliverTaylor2021


How To Contribute
~~~~~~~~~~~~~~~~~~

Got a great idea for something to implement in ASLDRO, or maybe you have just
found a bug? Create an issue at 
https://github.com/gold-standard-phantoms/asldro/issues to get in touch with
the development team and we'll take it from there.