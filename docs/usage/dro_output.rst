DRO Output Data
================

When ASLDRO's main pipeline is run using the ``generate`` command, for example::

    asldro generate --params path/to/input_params.json path/to/output_archive.zip

the resulting DRO image data is saved in the archive, in accordance with the
data structure and format as specified by BIDS (https://bids.neuroimaging.io/).

ASLDRO currently supports BIDS version 1.5.0, however because of the nature of 
DRO data there are some small deviations and non-official fields used. Deviations
are described in the section :ref:`bids-deviations`.

Each image series defined in the input parameters will have a NIFTI image and
corresponding JSON sidecar generated. For ASL image series an additional 
*_aslcontext.tsv is generated which indicates the type of ASL volumes in the
image.

These will be saved in the following subdirectories:

* 'perf' for series type 'asl'
* 'anat' for series type 'structural'
* 'ground_truth' for series type 'ground_truth'

Filenames are given by ``acq-<series_number>_<modality_label>.<ext>``,
where:

* ``<series_number>`` is the order the particular image series is in the ``"image_series"``
  array in the input parameters file (1 indexed, zero padded to 3 digits).
* ``<modality_label>`` is based on series type:
    :asl: determined by the input parameter ``asl_context``. If ``asl_context``
      only contains entries  with ``'m0scan'`` then it will be
      ``'m0scan'``, otherwise ``'asl'``.
    :structural: determined by the input parameter ``modality``, which can be
      "T1w", "T2w", "FLAIR", or "anat" (default).
    :ground_truth: the concatenation of 'ground_truth' and the name of 
      the 'quantity' for the ground truth image, separated by an underscore.

For example, running the DRO with parameters to generate the following image
series in order:

#. asl, ``asl_context = "m0scan, control, label"``
#. asl, ``asl_context = "control, label"``
#. asl, ``asl_context = "m0scan"``
#. structural, ``modality = "T1w"``
#. strutural, ``modality = "T2w"``
#. structural, modality entry missing.
#. ground truth

Will result in the following files output

::

    output_archive.zip
    |-- perf
    |   |-- acq-001_asl.nii.gz
    |   |-- acq-001_asl.json
    |   |-- acq-001_aslcontext.tsv
    |   |-- acq-002_asl.nii.gz
    |   |-- acq-002_asl.json
    |   |-- acq-002_aslcontext.tsv
    |   |-- acq-003_m0scan.nii.gz
    |   
    |-- anat
    |   |-- acq-004_T1w.nii.gz
    |   |-- acq-004_T1w.json
    |   |-- acq-005_T2w.nii.gz
    |   |-- acq-005_T2w.json
    |   |-- acq-006_anat.nii.gz
    |   |-- acq-006_anat.json
    |
    |---ground_truth
        |-- acq-007_ground_truth_perfusion_rate.nii.gz
        |-- acq-007_ground_truth_perfusion_rate.json
        |-- acq-007_ground_truth_transit_time.nii.gz
        |-- acq-007_ground_truth_transit_time.json
        |-- acq-007_ground_truth_t1.nii.gz
        |-- acq-007_ground_truth_t1.json
        |-- acq-007_ground_truth_t2.nii.gz
        |-- acq-007_ground_truth_t2.json
        |-- acq-007_ground_truth_t2_star.nii.gz
        |-- acq-007_ground_truth_t2_star.json
        |-- acq-007_ground_truth_m0.nii.gz
        |-- acq-007_ground_truth_m0.json
        |-- acq-007_ground_truth_seg_label.nii.gz
        |-- acq-007_ground_truth_seg_label.json


.. _bids-deviations:

Deviations from the BIDS Standard
-----------------------------------

Background Suppression
~~~~~~~~~~~~~~~~~~~~~~~

In the BIDS standard it is assumed that background suppression pulses comprise of:

#. A saturation pulse of duration 0 occuring at the start of the labelling pulse, 
    e.g. the time betweeen the saturation pulse and the imaging excitation pulse is
    equal to ``label_duration + post_label_delay``.
#. All inversion pulses occur after the start of the labelling pulse.

To allow for more possibilities for background timings, the following changes have
been implemented.

:BackgroundSuppressionPulseTime: (modified) Negative values are permitted. A
  negative value indicates that the inversion pulse occurs before the label
  pulse has started.
:BackgroundSuppressionSatPulseTime: (new) The time in seconds between the saturation
  pulse and the imaging excitation pulse.


