Quickstart
==========

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
      "echo_time": [0.1, 0.2, 0.3],
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


