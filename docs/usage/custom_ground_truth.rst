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

The blood-brain-partition coefficient can be supplied as an image or as a single whole-brain value 
in the accompanying JSON file.

By definition the HRGT does not have any partial volume effects; only a single tissue type is in
each voxel. This is in contrast with the usual 'fuzzy' segmentation masks that are supplied as part
of templates.

The json parameter file describes the following:

* The quantities present in the HRGT NIFTI file, in the order they are concatenated along the 5th
  dimension.
* The units that correspond with the quantities (note that ASLDRO expects all units to be SI)
* The names of the tissues and the corresponding value in the segmentation mask.
* Parameters: the blood-brain partition coefficient, T1 of arterial blood and the magnetic field
  strength that the HRGT is for.  ASLDRO does not use the magnetic field strength parameter, other 
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
--------------------------------

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

    #. Create the 'parameters' object, this must have the entries for 't1_arterial_blood',
       and 'magnetic_field_strength'. If 'lambda_blood_brain' is not an image quantity then
       it is also required here.
#. Test your ground truth with ASLDRO. Once loaded it will be validated, so you will receive an
   error message if there are any issues.


Command line tools to generate ground truths
---------------------------------------------

ASLDRO comes with two command line functions to assist with generating custom ground truths:

create-hrgt
~~~~~~~~~~~~

::

   asldro create-hrgt /path/to/hrgt_params.json /path/to/seg_mask.nii.gz /path/to/output_dir

This command creates a ground truth that is valid for ASLDRO using the following:

:``/path/to/hrgt_params.json``: Path to a JSON file (must have extension .json) that describes how
   to construct the ground truth. It has the following objects:

   :'label_values': An array of integers, which must be the same as the integer values in the
     accompanying seg_mask.nii.gz, including zero values. The order in this array defines
     the order in the objects 'label_names', and the the array for each quantity in 'quantities'.
   :'label_names': An array of strings, defining the names for the regions.
   :'quantities': Comprises multiple objects, one for each quantity to be represented in the
     HRGT. The value for each quantity is an array of floats, the same length as the number of
     regions, and corresponding to the order in 'label_values'.
   :'units': An array of strings, defining the units for each quantity. Must match the number of
     quantities, and corresponds with the order.
   :'properties': This requires two objects, 't1_arterial_blood' and 'magnetic_field_strength', both
     of which are floats. Additional properties are allowed and will propagate to the output
     hrgt JSON file. Note that to be a valid input HRGT to ASLDRO, either one of the quantities,
     or one of the parameters must be 'lambda_blood_brain'.

:``/path/to/seg_mask.nii.gz``: A NIFTI file (must have extension .nii or .nii.gz) which defines
  segmentation regions. It is recommended that this is an integer data type, however floating
  point data is accepted and will have a ceiling operation applied to it.
:``/path/to/output_dir``: The directory to output the HRGT files to. hrgt.nii.gz and hrgt.json will
  be created here, if the files already exist they will be overwritten.

An example of the hrgt_params.json is shown below, the quantity values are used in the
built-in hrgt_icbm_2009a_nls_3t ground truth:

.. code-block:: json

   {
      "label_values": [0, 1, 2, 3],
      "label_names": ["background", "grey_matter", "white_matter", "csf"],
      "quantities": {
         "perfusion_rate": [0.0, 60.0, 20.0, 0.0],
         "transit_time": [0.0, 0.8, 1.2, 1000.0],
         "t1": [0.0, 1.33, 0.83, 3.0],
         "t2": [0.0, 0.08, 0.11, 0.3],
         "t2_star": [0.0, 0.066, 0.053, 0.2],
         "m0": [0.0, 74.62, 64.73, 68.06]
      },
      "units": ["ml/100g/min", "s", "s", "s", "s", ""],
      "parameters": {
         "t1_arterial_blood": 1.80,
         "lambda_blood_brain": 0.9,
         "magnetic_field_strength": 3.0		
      }
   }


combine-masks
~~~~~~~~~~~~~~

::

   asldro combine-masks /path/to/combine_masks_params.json /path/to/output_image.nii.gz

This command combines multiple 'fuzzy' masks into a single segmentation mask. A fuzzy mask is
defined as having voxel values between 0 and 1 that define the fraction of that voxel that is
occupied by the particular region/tissue; they can represent partial volumes.
ASLDRO defines the HRGT to not have any partial volume effects, i.e. only a single tissue/region
type per voxel. The ``combine-masks`` function combines these fuzzy masks in a defined manner,
using a combination of thresholding and a priority order for each mask.

A voxel will be assigned a region value if the following conditions are all met:

   #. Its fuzzy_mask value is higher than a defined threshold (default is 0.05)
   #. Its fuzzy_mask value is highest out of any other masks that also have non-zero values
      in that voxel.
   #. If it jointly has the same highest value as one or more other masks, it must have a higher
      priority assigned than the competing regions.

The following arguments are required:

:``/path/to/combine_masks_params.json``: Path to a JSON file (must have extension .json) which
   describes how the masks are to be combined. This has the following objects:

   :'mask_files': An array of paths to the mask files to combine. Each file must be in NIFTI format
     with extension .nii or .nii.gz. They must all have the same dimensions, and have the same
     affine matrix (srow_x, srow_y, and srow_z) so that they correspond to the same location
     in world-space.
   :'region_values': An array of integers, to assign to each mask region. Their order corresponds
     with the order of the mask files.
   :'region_priority': An array of integers defining the priority order for the regions, 1 being the
     highest priority.
   :'threshold': (optional) A number between 0 and 1 that defines the threshold that voxel values
     must be greater than to be considered for array assignment. Default value is 0.05.

:``/path/to/output_image.nii.gz``: Path to a NIFTI (must have extension .nii or.nii.gz) image
   that will be created.

If only a single mask file is supplied, only the threshold condition will be applied. However
'region_values' and 'region_priority' must still be present.

An example of the combine_masks_params.json file is shown below:

.. code-block:: json

   {
      "mask_files": [
         "/path/to/mask_1.nii.gz",
         "/path/to/mask_2.nii.gz",
         "/path/to/mask_3.nii.gz"
      ],
      "region_values": [1, 2, 3],
      "region_priority": [2, 1, 3],
      "threshold": 0.13
   }




