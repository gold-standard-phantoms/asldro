Quickstart
==========

Getting started
---------------

Eager to get started? This page gives a good introduction to ASL DRO.
Follow :doc:`installation` to set up a project and install ASL DRO first.

After installation the command line tool ``asldro`` will be made available. You can run::

    asldro generate path/to/output_file.zip

to run the DRO generation as-per the ASL White Paper specification. The output file may
be either .zip or .tar.gz.

Is it also possible to specify a parameter file, which will override any of the default values::

    asldro generate --params path/to/input_params.json path/to/output_file.zip

It is possible to create an example parameters file containing the model defaults by running::

    asldro output params /path/to/input_params.json

which will create the ``/path/to/input_params.json`` file. The parameters may be adjusted as
necessary and used with the 'generate' command. 

For details on input parameters see :doc:`parameters`

It is also possible to output the high-resolution ground-truth (HRGT) files.
To get a list of the available data, type::

    asldro output hrgt -h

To output the HRGT, type::

    asldro output hrgt HRGT OUTPUT_DIR

where HRGT is the code of the files to download, and OUTPUT_DIR is the directory to output to.

There are also two ancillary command line features that assist with the creation of 
HRGT's. To create a valid HRGT::

    asldro create-hrgt /path/to/hrgt_params.json /path/to/seg_mask.nii.gz /path/to/output_dir

This takes a segmentation mask image, where each voxel is has an integer value that defines
a region/tissue type, and a parameter file that describes values for each quantity to assign
to each region, then concatenates volumes for each quantity into a 5D NIFTI, which is saved
along with a JSON file describing the HRGT to the output directory, and are valid as an input
HRGT for ASLDRO.

To create a segmentation mask from individual 'fuzzy' masks for each region type::

    asldro combine-masks /path/to/combine_masks_params.json /path/to/output_image.nii.gz

This combines multiple fuzzy masks, where the voxel values define the fraction of that
voxel that is occupied by the particular region/tissue, into a single segmentation mask image,
that is valid for the create-hrgt command. The combine_masks_params.json defines the values
to assign for each region, and an order of priority and thresholds.

For more information about how to use these features see :doc:`custom_ground_truth`.



Pipeline details
----------------

There are three pipelines available in ASLDRO

* The full ASL pipeline.
* A structural MRI pipeline (generates gradient echo, spin echo or inversion recovery signal).
* A ground truth pipeline that simply resamples the input ground truth to the specified resolution.

In a single instance of ASLDRO, the input parameter file can configure any number and configurations
of these pipelines to be run, much in the way that this can be done on an MRI scanner.

The full ASL pipeline comprises of:

#. Loading in the ground truth volumes.
#. Producing :math:`\Delta M` using the General Kinetic Model for the specified ASL parameters.
#. Generating synthetic M0, Control and Label volumes.
#. Applying motion
#. Sampling at the acquisition resolution
#. Adding instrument and physiological pseudorandom noise.

The structural pipeline excludes the General Kinetic Model, and just generates volumes with synthetic
MR contrast.  The ground truth pipeline only has the motion model and sampling.

Each volume described in ``asl_context`` has the motion, resampling and noise processes applied
independently. The rotation and translation arrays in the input parameters describe this motion, and
the the random number generator is initialised with the same seed each time the DRO is run, so each
volume will have noise that is unique, but statistically the same.

If ``desired_snr`` is set to ``0``, the resultant images will not have any noise applied. 

Each pipeline outputs files in BIDS (https://bids.neuroimaging.io/) format, consisting of a NIFTI
image file and accompanying json sidecar. In the case of an ASL image an 
additional '\*_aslcontext.tsv' file is also generated which describes the ASL volumes
present in the timeseries. 

The DRO pipeline is summarised in this schematic (click to view full-size):

.. image:: /images/asldro.png
  :scale: 50
