.. _custom-ground-truth:

Making an input ground truth
==============================

The input ground truth consists of two separate files:

* A NIFTI image which has each ground truth quantity concatenated across the 5th dimension.
* A json file which contains specified fields that describe the ground truth.

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

Below is the json parameter file for the built in `hrgt_icbm_2009a_nls_3t` ground truth

.. code-block:: json

    {
        "quantities": [
            "perfusion_rate", "transit_time", "t1", "t2", "t2_star", "m0", "seg_label"],
        "units": ["ml/100g/min", "s", "s", "s", "s", "", ""],
        "segmentation": {
            "grey_matter": 1,
            "white_matter": 2,
            "csf": 3
        },
        "parameters": {
            "lambda_blood_brain": 0.9,
            "t1_arterial_blood": 1.65,
            "magnetic_field_strength": 3
        }
    }

The background is assumed to have value 0.

How to construct a ground truth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The general procedure for constructing your own ground truth is as follows.

#. Obtain the segmentation masks for each of the tissues you wish to represent in the ground truth.
   For example this could be from an atlas image, or could simply be automated/manual segmentations
   from a single scan.
#. If these masks are fuzzy they will need to be thresholded to create a binary mask.
#. Combine these individual binary masks into a Label Map: where each tisue is represented by an
   integer value. This ensures that there is only one tissue type per voxel.
#. Using this label mask create volumes where values are assigned to each tissue. There should be
   one volume per quantity, all of the required quantities are necessary for the ASLDRO pipelines to
   run, but more can be added. All the voxels of a particular tissue could be set to the same value,
   or some variation could be introduced.
#. Concatenate these volumes along the 5th dimension and save as a NIFTI file.  The Label Map is 
   the `'seg_label'` volume, this needs to be included.
#. Construct the json parameter file:
    #. list the names of the quantities in the order they are concatenated in the 'quantities' array.
    #. list the corresponding units of the quantities in the 'units' array.  If the quantity is 
       unitless then use an empty string `""`.
    #. Add key/value pairs comprising of the tissue name and the integer value it is assigned in the
       Label Map image to the object 'segmentation'.  Valid tissue names are:

        * 'background': identifies background signal
        * 'grey_matter': identifies grey matter
        * 'white_matter': identifies white matter
        * 'csf': identifies cerebrospinal fluid
        * 'vascular': identifies blood vessels
        * 'lesion': identifies lesions

    #. Create the 'parameters' object, this must have the three entries for 't1_arterial_blood',
       'lambda_blood_brain', and 'magnetic_field_strength'.
#. Test your ground truth with ASLDRO. Once loaded it will be validated, so you will receive an
   error message if there are any issues.