Ground Truth
============

The input ground truth, known as the High Resolution Ground Truth (or HRGT)
comprises of a 5-dimensional NIFTI image, and json parameter file that
describes the NIFTI and also supplies any non-image ground truth parameters. 3D Volumes for each
ground truth quantity are concatenated along the 5th NIFTI dimension.

.. image:: /images/ground_truth_concatenation.png

The following quantities are required:

* Perfusion rate: The rate of delivery of arterial blood to an organ.
* Tissue Transit Time: The time following labelling for the perfusion signal to reach a voxel.
* M0: The equilibrium magnetisation within a voxel.
* T1: Spin-lattice (longitudinal) relaxation time constant.
* T2: Spin-spin (transverse) relaxation time constant.
* T2*: Spin-spin (transverse) relaxation time constant, including time-invariant magnetic field
  inhomogeneities.
* Segmentation Mask: Voxels are assigned a number corresponding with the specific tissue type.

By definition the HRGT does not have any partial volume effects; only a single tissue type is in
each voxel. This is in contrast with the usual 'fuzzy' segmentation masks that are supplied as part
of templates.

The json parameter file describes the following:

* The quantities present in the HRGT NIFTI file, in the order they are concatenated along the 5th
  dimension.
* The units that correspond with the quantities (note that ASLDRO expects all units to be SI)
* The names of the tissues and the corresponding value in the segmentation mask.
* Parameters: the blood-brain partition coefficient, t1 of arterial blood and the magnetic field
  strength that the HRGT is for.  ASLDRO does use the magnetic field strength parameter, other 
  than to include in BIDS field "MagneticFieldStrength".


Different HRGT's are selected by modifiying the "ground_truth" entry in the input parameter file.
ASLDRO currently supports the following ground truths:


hrgt_icbm_2009a_nls_v4
~~~~~~~~~~~~~~~~~~~~~~

A 1x1x1mm ground truth based on the MNI ICBM 2009a Nonlinear
Symmetric template, obtained from http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009,
with quantitative parameters set based on supplied masks.  This ground truth has the following
values for each tissue and quantity (corresponding to 3T):

+--------------+----------------+--------------+----------+----------+----------+----------+
| Tissue       | Perfusion Rate | Transit Time | T1       | T2       | T2*      | label    |
|              | [ml/100g/min]  | [s]          | [s]      | [s]      | [s]      |          |
+==============+================+==============+==========+==========+==========+==========+
| Grey Matter  | 60.00          | 1.00         | 1.33     | 0.110    | 0.050    | 1        | 
+--------------+----------------+--------------+----------+----------+----------+----------+
| White Matter | 20.00          | 1.50         | 0.83     | 0.080    | 0.050    | 2        |
+--------------+----------------+--------------+----------+----------+----------+----------+
| CSF          | 0.00           | 1000.0       | 3.00     | 0.300    | 0.200    | 3        |
+--------------+----------------+--------------+----------+----------+----------+----------+

label is the integer value assigned to the tissue in the seg_label volume.