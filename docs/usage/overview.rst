Overview
--------

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

