Quickstart
==========

Eager to get started? This page gives a good introduction to ASL DRO.
Follow :doc:`installation` to set up a project and install ASL DRO first.

After installation the command line tool `asldro` will be made available. You can run::

    asldro path/to/input_params.json path/to/output_file.zip

The output file may be either .zip or .tar.gz. The input parameters file must currently include, at minimum.

.. code-block:: json

    {
      "asl_context_array": "m0scan m0scan control label",
      "label_type": "pCASL",
      "lambda_blood_brain": 0.9,
      "t1_arterial_blood": 1.65
    }

The parameters may be adjusted as necessary.
