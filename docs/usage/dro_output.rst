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

These will be saved in a subdirectory ``sub-<subject_label``> and then in
the following subdirectories:

* 'perf' for series type 'asl'
* 'anat' for series type 'structural'
* 'ground_truth' for series type 'ground_truth'

Filenames are given by ``sub-<subject_label>_acq-<series_number>_<modality_label>.<ext>``,
where:

* ``<subject_label>`` is given by the global configuration parameter :ref:`subject_label<subject-label>`
  which defaults to "001" 
* ``<series_number>`` is the order the particular image series is in the ``"image_series"``
  array in the input parameters file (1 indexed, zero padded to 3 digits).
* ``<modality_label>`` is based on series type:
    :asl: determined by the input parameter ``asl_context``. If ``asl_context``
      only contains entries  with ``'m0scan'`` then it will be
      ``'m0scan'``, otherwise ``'asl'``.
    :structural: determined by the input parameter ``modality``, which can be
      "T1w" (default), "T2w", "FLAIR", "PDw", "T2starw", "inplaneT1", "PDT2", or
      "UNIT1".
    :ground_truth: the concatenation of 'ground-truth' and the name of 
      the 'quantity' for the ground truth image, separated by a hyphen. Any
      underscores in the quantity name will be converted to hyphens.

For example, running the DRO with parameters to generate the following image
series in order:

#. asl, ``asl_context = "m0scan, control, label"``
#. asl, ``asl_context = "control, label"``
#. asl, ``asl_context = "m0scan"``
#. structural, ``modality = "FLAIR"``
#. strutural, ``modality = "T2w"``
#. structural, modality entry missing.
#. ground truth

Will result in the following files output

::

  output_archive.zip
    |-- dataset_description.json
    |-- README
    |-- .bidsignore
    |-- sub-001
        |
        |-- perf
        |   |-- sub-001_acq-001_asl.nii.gz
        |   |-- sub-001_acq-001_asl.json
        |   |-- sub-001_acq-001_aslcontext.tsv
        |   |-- sub-001_acq-002_asl.nii.gz
        |   |-- sub-001_acq-002_asl.json
        |   |-- sub-001_acq-002_aslcontext.tsv
        |   |-- sub-001_acq-003_m0scan.nii.gz
        |   
        |-- anat
        |   |-- sub-001_acq-004_FLAIR.nii.gz
        |   |-- sub-001_acq-004_FLAIR.json
        |   |-- sub-001_acq-005_T2w.nii.gz
        |   |-- sub-001_acq-005_T2w.json
        |   |-- sub-001_acq-006_T1w.nii.gz
        |   |-- sub-001_acq-006_T1w.json
        |
        |---ground_truth
            |-- sub-001_acq-007_Perfmap.nii.gz
            |-- sub-001_acq-007_Perfmap.json
            |-- sub-001_acq-007_ATTmap.nii.gz
            |-- sub-001_acq-007_ATTmap.json
            |-- sub-001_acq-007_T1map.nii.gz
            |-- sub-001_acq-007_T1map.json
            |-- sub-001_acq-007_T2map.nii.gz
            |-- sub-001_acq-007_T2map.json
            |-- sub-001_acq-007_T2starmap.nii.gz
            |-- sub-001_acq-007_T2starmap.json
            |-- sub-001_acq-007_M0map.nii.gz
            |-- sub-001_acq-007_M0map.json
            |-- sub-001_acq-007_dseg.nii.gz
            |-- sub-001_acq-007_dseg.json


.. _bids-deviations:

Deviations from the BIDS Standard
-----------------------------------

Ground Truth Image Series
~~~~~~~~~~~~~~~~~~~~~~~~~~

BIDS specifies that parameter maps should be saved in the 'anat' folder, however
ground truth parameter maps generated using the ``ground_truth`` image series
are saved in 'ground_truth' folder.

Additional suffixes have been devised for non-supported parameter maps:

* Perfmap: Perfusion rate map.
* ATTmap: Transit time map.
* Lambdamap: Blood brain partition coefficient map.

The .bidsignore file has entries to ignore everything in the ground_truth folder
and in addition the above non-supported suffixes.


Background Suppression
~~~~~~~~~~~~~~~~~~~~~~~

In the BIDS standard it is assumed that background suppression pulses comprise of:

#. A saturation pulse of duration 0 occuring at the start of the labelling pulse, 
    e.g. the time betweeen the saturation pulse and the imaging excitation pulse is
    equal to ``label_duration + post_label_delay``.
#. All inversion pulses occur after the start of the labelling pulse.

To allow for more possibilities for background timings, the following changes have
been implemented:

:BackgroundSuppressionPulseTime: (modified) Negative values are permitted. A
  negative value indicates that the inversion pulse occurs before the label
  pulse has started.
:BackgroundSuppressionSatPulseTime: (new) The time in seconds between the saturation
  pulse and the imaging excitation pulse.


Multiphase ASL
~~~~~~~~~~~~~~~

Multiphase ASL data can be generated by supplying the parameter 
:ref:`signal_time<signal-time>` as an array. For each value in this array
the volumes defined in :ref:`asl_context<asl-context>` are generated. ASL BIDS
does not currently support multiphase data, so the following has been implemented:

:MultiphaseIndex: (new) An array of integers, with one entry per volume in the ASL
  timeseries, indicating the index of the multiphase loop when each volume was generated.
:PostLabelingDelay: (modified) For multiphase data this is an array of the
  corresponding post labelling delay times for each multiphase index. For single
  phase ASL (i.e. only one PLD) then this is a single number.



