.. _asl_quantification:

ASL Quantification
===================

ASDRO has a command line function to perform quantification on valid
ASL BIDS data, comprising a NIFTI, JSON sidecar and *aslcontext.tsv file

::

    asldro asl-quantify --params /path/to/quant_params.json /path/to/asl.nii.gz /path/to/output_dir

where:

:``path/to/quant_params``: Path to a JSON file (must have extension .json) providing
  parameters for the quantifiction calculation. It has the following objects:

    :QuantificationModel: (string) defaults to "whitepaper" 
        (see :class:`.AslQuantificationFilter` for options)
    :ArterialSpinLabelingType: (string) "PCASL", "CASL" or "PASL"
    :PostLabelingDelay: (float) The post labeling delay in seconds.
    :LabelingDuration: (float) The label duration in seconds (pCASL/CASL only)
    :BolusCutOffDelayTime: (float) The bolus cutoff delay time (PASL only)
    :LabelingEfficiency: (float) The efficiency of the labeling pulse
    :T1ArterialBlood: (float) If not supplied the default value is based on
        the value of the BIDS field "MagneticFieldStrength":

        :1.5 Tesla: 1.35s 
        :3.0 Tesla: 1.65s
    
:``/path/to/asl.nii.gz``: The path to a valid ASL NIFTI file (must have extension
  .nii or .nii.gz). It is assumed that
  in the same location there are also corresponding ``/path/to/asl.json`` and
  ``/path/to/aslcontext.tsv`` files.
:``path/to/output_dir``: The directory to output the generated perfusion rate/CBF
  maps to, comprising a NIFTI image and JSON sidecar. The files will be given
  the same filename as the input NIFTI, with '_cbf' appended before the extension.

  For more details on implementation see the :func:`.asl_quantification` pipeline.

  An example of the quant_params.json file is given below:

  .. code-block:: json

    {
        "QuantificationModel": "whitepaper",
        "PostLabelingDelay": 1.8,
        "LabelingEfficiency": 0.85,
        "BloodBrainPartitionCoefficient": 0.9,
        "T1ArterialBlood": 1.65,
        "ArterialSpinLabelingType": "PCASL",
        "LabelingDuration": 1.8,
    }
  