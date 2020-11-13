Quickstart
==========

Overview
--------

ASL DRO is software that can generate digital reference objects for Arterial Spin Labelling (ASL) MRI.
It creates synthetic raw ASL data according to set acquisition and data format parameters, based
on input ground truth maps for:

* Perfusion rate
* Transit time
* Intrinsic MRI parameters: M0, T1, T2, T2*
* Tissue segmentation (defined as a single tissue type per voxel)

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
necessary and used with the 'generate' command. The input parameters will include, as default:

.. code-block:: json

    {
      "asl_context": "m0scan control label",
      "label_type": "pcasl",
      "label_duration": 1.8,
      "signal_time": 3.6,
      "label_efficiency": 0.85,
      "echo_time": [0.01, 0.01, 0.01],
      "repetition_time": [10.0, 5.0, 5.0],
      "rot_z": [0.0, 0.0, 0.0],
      "rot_y": [0.0, 0.0, 0.0],
      "rot_x": [0.0, 0.0, 0.0],
      "transl_x": [0.0, 0.0, 0.0],
      "transl_y": [0.0, 0.0, 0.0],
      "transl_z": [0.0, 0.0, 0.0],
      "acq_matrix": [64, 64, 12],
      "acq_contrast": "se",
      "desired_snr": 10.0,
      "random_seed": 0
    }

The parameters may be adjusted as necessary. The parameter `asl_context` defines the number of 
simulated acquisition volumes that should be generated.  The following array parameters need to
have the same number of entries as there are defined volumes:

* ``echo_time``
* ``repetition_time``
* ``rot_z``
* ``rot_y``
* ``rot_x``
* ``transl_x``
* ``transl_y``
* ``transl_z``

For more details on input parameters see :doc:`parameters`

It is also possible to output the high-resolution ground-truth (HRGT) files.
To get a list of the available data, type::

    asldro output hrgt -h

To output the HRGT, type::

    asldro output hrgt HRGT OUTPUT_DIR

where HRGT is the code of the files to download, and OUTPUT_DIR is the directory to output to.
    

Pipeline details
----------------

The DRO currently runs using the default ground truth.
Future releases will allow this to be configured.  The pipeline comprises of:

#. Loading in the ground truth volumes.
#. Producing :math:`\Delta M` using the General Kinetic Model for the specified ASL parameters.
#. Generating synthetic M0, Control and Label volumes.
#. Applying motion
#. Sampling at the acquisition resolution
#. Adding instrument and physiological pseudorandom noise.

Each volume described in ``asl_context`` has the motion, resampling and noise processes applied
independently. The rotation and translation arrays in the input parameters describe this motion, and
the the random number generator is initialised with the same seed each time the DRO is run, so each
volume will have noise that is unique, but statistically the same.

If ``desired_snr`` is set to ``0``, the resultant images will not have any noise applied. 

Once the pipeline is run, the following images are created:

* Timeseries of magnitude ASL volumes in accordance with ``asl_context`` (asl_source_magnitude.nii.gz)
* Ground truth perfusion rate, resampled to ``acq_matrix`` (gt_cbf_acq_res_nii.gz)
* Ground truth tissue segmentation mask, resampled to ``acq_matrix`` (gt_labelmask_acq_res.nii.gz)

The DRO pipeline is summarised in this schematic (click to view full-size):

.. image:: /images/asldro.png
  :scale: 50
