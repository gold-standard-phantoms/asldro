""" BidsOutputFilter """
import os
import logging
from typing import Union, List
from datetime import datetime, timezone
import json
from jsonschema import validate

import numpy as np
import nibabel as nib

from asldro.containers.image import (
    BaseImageContainer,
    IMAGINARY_IMAGE_TYPE,
    COMPLEX_IMAGE_TYPE,
    MAGNITUDE_IMAGE_TYPE,
    PHASE_IMAGE_TYPE,
    REAL_IMAGE_TYPE,
)
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.validators.parameters import (
    Parameter,
    ParameterValidator,
    isinstance_validator,
    from_list_validator,
    greater_than_equal_to_validator,
    for_each_validator,
    regex_validator,
    or_validator,
    non_empty_list_validator,
)
from asldro.filters.gkm_filter import GkmFilter
from asldro.filters.mri_signal_filter import MriSignalFilter
from asldro.filters.ground_truth_loader import GroundTruthLoaderFilter
from asldro.filters.transform_resample_image_filter import TransformResampleImageFilter
from asldro.utils.general import map_dict
from asldro.validators.user_parameter_input import (
    ASL,
    STRUCTURAL,
    GROUND_TRUTH,
    MODALITY,
    ASL_CONTEXT,
    SUPPORTED_IMAGE_TYPES,
    SUPPORTED_ASL_CONTEXTS,
    SUPPORTED_STRUCT_MODALITY_LABELS,
)
from asldro.filters.background_suppression_filter import BackgroundSuppressionFilter

from asldro.data.filepaths import ASL_BIDS_SCHEMA, M0SCAN_BIDS_SCHEMA

from asldro import __version__ as asldro_version

logger = logging.getLogger(__name__)


class BidsOutputFilter(BaseFilter):
    """ A filter that will output an input image container in Brain Imaging Data Structure
    (BIDS) format, in accordance with the version 1.5.0 specification.

    BIDS comprises of a NIFTI image file and accompanying .json sidecar that contains additional
    parameters.  More information on BIDS can be found at https://bids.neuroimaging.io/

    Multiple instances of the BidsOutputFilter can be used to output multiple
    images to the same directory, building a valid BIDS dataset.

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`BidsOutputFilter.add_inputs()`
    member function. They are also accessible via class constants,
    for example :class:`BidsOutputFilter.KEY_IMAGE`

    :param 'image': the image to save in BIDS format
    :type 'image': BaseImageContainer
    :param 'output_directory': The root directory to save to
    :type 'output_directory': str
    :param 'subject_label': The subject label, can only contain
      alphanumeric characters and a hyphen (-). Defaults to "001".
    :type 'subject_label': str, optional
    :param 'filename_prefix': string to prefix the filename with.
    :type 'filename_prefix': str, optional

    **Outputs**

    Once run, the filter will populate the dictionary :class:`BidsOutputFilter.outputs` with the
    following entries

    :param 'filename': the filename of the saved file
    :type 'filename': str
    :param 'sidecar': the fields that make up the output *.json file.
    :type 'sidecar': dict

    Files will be saved in subdirectories corresponding to the metadata entry ``series_type``:

    * 'structural' will be saved in the subdirectory 'anat'
    * 'asl' will be saved in the subdirectory 'perf'
    * 'ground_truth' will be saved in the subdirectory 'ground_truth'

    Filenames will be given by:
    ``sub-<subject_label>_<filename_prefix>_acq-<series_number>_<modality_label>.<ext>``, where

    * ``<subject_label>`` is the string supplied by the input ``subject_label``
    * ``<series_number>`` is given by metadata field ``series_number``, which is an integer number
      and will be prefixed by zeros so that it is 3 characterslong, for example 003, 010, 243
    * ``<filename_prefix>`` is the string supplied by the input ``filename_prefix``
    * ``<modality_label>`` is determined based on ``series_type``:

        * 'structural': it is given by the metadata field ``modality``.
        * 'asl': it is determined by asl_context.  If asl_context only contains entries that match
          with 'm0scan' then it will be set to 'm0scan', otherwise 'asl'.
        * 'ground_truth': the metadata field ``quantity`` will be mapped to the according BIDS 
          parameter map names, as given by :class:`BidsOutputFilter.QUANTITY_MAPPING` 

    **Image Metadata**

    The input ``image`` must have certain metadata fields present, these being dependent on the
    ``series_type``.

    :param 'series_type': Describes the type of series.  Either 'asl', 'structural' or
        'ground_truth'.
    :type 'series_type': str
    :param 'modality': modality of the image series, only required by 'structural'.
    :type 'modality': string
    :param 'series_number': number to identify the image series by, if multiple image series are
        being saved with similar parameters so that their filenames and BIDS fields would be
        identical, providing a unique series number will address this.
    :type 'series_number': int
    :param 'quantity': ('ground_truth' only) name of the quantity that the image is a map of.
    :type 'quantity': str
    :param 'units': ('ground_truth' only) units the quantity is in.
    :type 'units': str

    If ``series_type`` and ``modality_label`` are both 'asl' then the following metadata entries are
    required:

    :param 'label_type': describes the type of ASL labelling.
    :type 'str':
    :param 'label_duration': duration of the labelling pulse in seconds.
    :type 'label_duration': float
    :param 'post_label_delay: delay time following the labelling pulse before the acquisition in
        seconds. For multiphase ASL these values are supplied as a list, where entries are either
        numeric or equal to ``None``. 
    :type 'post_label_delay': float or List containg either float or ``None``.
    :param 'label_efficiency': the degree of inversion of the magnetisation (between 0 and 1)
    :type 'label_efficiency': float
    :param 'image_flavour': a string that is used as the third entry in the BIDS field ``ImageType``
        (corresponding with the dicom tag (0008,0008).  For ASL images this should be 'PERFUSION'.
    :type 'image_flavour': str
    :param 'background_suppression': A boolean denoting whether background suppression has been
      performed. Can be omitted, in which case it will be assumed there
      is no background suppression.
    :type 'background_suppression': bool
    :param 'background_suppression_inv_pulse_timing': A list of inversion pulse timings for the
      background suppression pulses. Required if ``'background_suppression'`` is True.
    :type 'background_suppression_inv_pulse_timing': list[float]
    :param 'multiphase_index': Array of the index in the multiphase loop when the volume was
      acquired. Only required if ``post_label_delay`` is a list and has length > 1
    :type 'multiphase_index': int or List[int]

    In a multiphase ASL image is supplied (more than one PLD), then there are additional validation
    checks:

    * The length of ``multiphase_index`` and ``post_label_delay`` must the the same.
    * All values of ``post_label_delay`` corresponding to a given ``multiphase_index``
      that are not equal to ``None`` must be the same.

    Input image metadata will be mapped to corresponding BIDS fields. See
    :class:`BidsOutputFilter.BIDS_MAPPING` for this mapping.

    Note that for multiphase ASL with background suppression, the BIDS timings are calculated based
    on the longest post labelling delay.

    **dataset_description.json**

    If it does not exist already, when run the filter will create the file
    ``dataset_description.json`` at the root of the output directory.
    This contains the information contained in 
    :class:`BidsOutputFilter.DATASET_DESCRIPTION`.

    **README**
    
    If it does not already exist, when run the filter will create the text file ``README`` at the
    root of the output directory. This contains some information about the dataset, including
    a list of all the images output.

    **.bidsignore**

    A .bidsignore file is created at the root of the output directory, which includes entries
    indicating non-standard files to ignore.
    """

    # Key Constants
    KEY_IMAGE = "image"
    KEY_OUTPUT_DIRECTORY = "output_directory"
    KEY_FILENAME_PREFIX = "filename_prefix"
    KEY_SUBJECT_LABEL = "subject_label"
    KEY_FILENAME = "filename"
    KEY_SIDECAR = "sidecar"

    SERIES_DESCRIPTION = "series_description"
    SERIES_NUMBER = "series_number"
    SERIES_TYPE = "series_type"
    DRO_SOFTWARE = "DROSoftware"
    DRO_SOFTWARE_VERSION = "DROSoftwareVersion"
    DRO_SOFTWARE_URL = "DROSoftwareUrl"
    ACQ_DATE_TIME = "AcquisitionDateTime"
    IMAGE_TYPE = "ImageType"
    M0_TYPE = "M0Type"
    M0_ESTIMATE = "M0Estimate"

    ASL_SUBDIR = "perf"
    STRUCT_SUBDIR = "anat"
    GT_SUBDIR = "ground_truth"

    # metadata parameters to BIDS fields mapping dictionary
    BIDS_MAPPING = {
        GkmFilter.KEY_LABEL_TYPE: "ArterialSpinLabelingType",
        GkmFilter.KEY_LABEL_DURATION: "LabelingDuration",
        GkmFilter.KEY_LABEL_EFFICIENCY: "LabelingEfficiency",
        GkmFilter.KEY_POST_LABEL_DELAY: "PostLabelingDelay",
        MriSignalFilter.KEY_ECHO_TIME: "EchoTime",
        MriSignalFilter.KEY_REPETITION_TIME: "RepetitionTimePreparation",
        MriSignalFilter.KEY_EXCITATION_FLIP_ANGLE: "FlipAngle",
        MriSignalFilter.KEY_INVERSION_TIME: "InversionTime",
        MriSignalFilter.KEY_ACQ_TYPE: "MRAcquisitionType",
        MriSignalFilter.KEY_ACQ_CONTRAST: "ScanningSequence",
        SERIES_DESCRIPTION: "SeriesDescription",
        SERIES_NUMBER: "SeriesNumber",
        TransformResampleImageFilter.VOXEL_SIZE: "AcquisitionVoxelSize",
        GroundTruthLoaderFilter.KEY_UNITS: "Units",
        GroundTruthLoaderFilter.KEY_MAG_STRENGTH: "MagneticFieldStrength",
        GroundTruthLoaderFilter.KEY_SEGMENTATION: "LabelMap",
        GroundTruthLoaderFilter.KEY_QUANTITY: "Quantity",
        GkmFilter.KEY_BOLUS_CUT_OFF_FLAG: "BolusCutOffFlag",
        GkmFilter.KEY_BOLUS_CUT_OFF_DELAY_TIME: "BolusCutOffDelayTime",
        BackgroundSuppressionFilter.M_BACKGROUND_SUPPRESSION: "BackgroundSuppression",
        BackgroundSuppressionFilter.M_BSUP_NUM_PULSES: "BackgroundSuppressionNumberPulses",
        BackgroundSuppressionFilter.M_BSUP_SAT_PULSE_TIMING: "BackgroundSuppressionSatPulseTime",
        "multiphase_index": "MultiphaseIndex",
    }

    # maps ASLDRO MRI contrast to BIDS contrast names
    ACQ_CONTRAST_MAPPING = {
        MriSignalFilter.CONTRAST_GE: "GR",
        MriSignalFilter.CONTRAST_SE: "SE",
        MriSignalFilter.CONTRAST_IR: "IR",
    }

    # maps ASLDRO image type names to complex components used in BIDS
    COMPLEX_IMAGE_COMPONENT_MAPPING = {
        REAL_IMAGE_TYPE: "REAL",
        IMAGINARY_IMAGE_TYPE: "IMAGINARY",
        COMPLEX_IMAGE_TYPE: "COMPLEX",
        PHASE_IMAGE_TYPE: "PHASE",
        MAGNITUDE_IMAGE_TYPE: "MAGNITUDE",
    }

    # Maps ASLDRO tissue types to BIDS standard naming
    LABEL_MAP_MAPPING = {
        "background": "BG",
        "grey_matter": "GM",
        "white_matter": "WM",
        "csf": "CSF",
        "vascular": "VS",
        "lesion": "L",
    }

    # Dataset description dictionary
    DATASET_DESCRIPTION = {
        "Name": "DRO Data generated by ASLDRO",
        "BIDSVersion": "1.5.0",
        "DatasetType": "raw",
        "License": "PD",
        "HowToAcknowledge": "Please cite this abstract: ASLDRO: "
        "Digital reference object software for Arterial Spin Labelling, "
        "A Oliver-Taylor et al., Abstract #2731, Proc. ISMRM 2021",
        "DROSoftware": "ASLDRO",
        "DROSoftwareVersion": asldro_version,
        "ReferencesAndLinks": [
            "code: https://github.com/gold-standard-phantoms/asldro",
            "pypi: https://pypi.org/project/asldro/",
            "docs: https://asldro.readthedocs.io/",
        ],
    }

    QUANTITY_MAPPING = {
        "t1": "T1map",
        "t2": "T2map",
        "t2_star": "T2starmap",
        "m0": "M0map",
        "perfusion_rate": "Perfmap",
        "transit_time": "ATTmap",
        "lambda_blood_brain": "Lambdamap",
        "seg_label": "dseg",
    }

    def __init__(self):
        super().__init__(name="BIDS Output")

    def _run(self):
        """ Writes the input image to disk in BIDS format """
        image: BaseImageContainer = self.inputs[self.KEY_IMAGE]
        output_directory = self.inputs[self.KEY_OUTPUT_DIRECTORY]
        subject_string = "sub-" + self.inputs[self.KEY_SUBJECT_LABEL]
        output_directory = os.path.join(output_directory, subject_string)
        # map the image metadata to the json sidecar
        json_sidecar = map_dict(image.metadata, self.BIDS_MAPPING, io_map_optional=True)
        series_number_string = f"acq-{image.metadata[self.SERIES_NUMBER]:03d}"
        # if the `filename_prefix` is not empty add an underscore after it
        if self.inputs[self.KEY_FILENAME_PREFIX] == "":
            filename_prefix = ""
        else:
            filename_prefix = self.inputs[self.KEY_FILENAME_PREFIX] + "_"

        # amend json sidecar
        # add ASLDRO information
        json_sidecar["Manufacturer"] = "Gold Standard Phantoms"
        json_sidecar[
            "PulseSequenceDetails"
        ] = f"Digital Reference Object Data generated by ASLDRO, version {asldro_version}"
        # set the acquisition date time to the current time in UTC
        json_sidecar[self.ACQ_DATE_TIME] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )

        # set the ScanningSequence value according to the BIDS spec
        if json_sidecar.get("ScanningSequence") is not None:
            json_sidecar["ScanningSequence"] = self.ACQ_CONTRAST_MAPPING[
                json_sidecar["ScanningSequence"]
            ]

        # set the ComplexImageType
        json_sidecar["ComplexImageComponent"] = self.COMPLEX_IMAGE_COMPONENT_MAPPING[
            image.image_type
        ]

        # default modality_label
        modality_label = ""
        # series_type specific things
        ## Series type 'asl'
        if image.metadata[self.SERIES_TYPE] == ASL:
            # ASL series, create aslcontext.tsv string
            sub_directory = self.ASL_SUBDIR

            modality_label = self.determine_asl_modality_label(
                image.metadata[ASL_CONTEXT]
            )

            if modality_label == ASL:
                # create _aslcontext_tsv
                asl_context_tsv = "volume_type\n" + "\n".join(
                    image.metadata[ASL_CONTEXT]
                )
                asl_context_filename = os.path.join(
                    output_directory,
                    sub_directory,
                    subject_string
                    + "_"
                    + filename_prefix
                    + series_number_string
                    + "_aslcontext.tsv",
                )
                # BIDS spec states ArterialSpinLabelingType should be uppercase
                json_sidecar["ArterialSpinLabelingType"] = json_sidecar[
                    "ArterialSpinLabelingType"
                ].upper()

                # set the BIDS field M0 correctly
                if any("m0scan" in s for s in image.metadata[ASL_CONTEXT]):
                    # if aslcontext contains one or more "m0scan" volumes set to
                    # "Included" to indicate "WithinASL"
                    json_sidecar[self.M0_TYPE] = "Included"
                elif isinstance(image.metadata.get("m0"), (float, int)):
                    # numerical value of m0 supplied so use this.
                    json_sidecar[self.M0_TYPE] = "Estimate"
                    json_sidecar[self.M0_ESTIMATE] = image.metadata["m0"]
                else:
                    # no numeric value or m0scan, so set to "Absent"
                    json_sidecar[self.M0_TYPE] = "Absent"

                # set the ImageType field
                json_sidecar["ImageType"] = [
                    "ORIGINAL",
                    "PRIMARY",
                    image.metadata["image_flavour"],
                    "NONE",
                ]

                # if the data is multiphase then the sidecar needs some editing to ensure
                # post label delay is correctly dealt with
                if isinstance(json_sidecar["PostLabelingDelay"], list):
                    if len(json_sidecar["PostLabelingDelay"]) > 1:
                        unique_mpindex = np.unique(json_sidecar["MultiphaseIndex"])
                        new_pld_array = []
                        for index in unique_mpindex:
                            mp_index_pld_vals = [
                                val
                                for idx, val in enumerate(
                                    json_sidecar["PostLabelingDelay"]
                                )
                                if json_sidecar["MultiphaseIndex"][idx] == index
                                and val is not None
                            ]
                            new_pld_array.append(mp_index_pld_vals[0])
                        json_sidecar["PostLabelingDelay"] = new_pld_array

                        # indexes = np.unique(
                        #     np.asarray(
                        #         json_sidecar["PostLabelingDelay"], dtype=np.float
                        #     ),
                        #     return_index=True,
                        # )[1]
                        # # np.unique returns sorted values, we want them in the order they
                        # # come in
                        # json_sidecar["PostLabelingDelay"] = [
                        #     json_sidecar["PostLabelingDelay"][index]
                        #     for index in sorted(indexes)
                        #     if json_sidecar["PostLabelingDelay"][index] is not None
                        # ]
                else:
                    # data is single phase, so if MultiphaseIndex is present, remove
                    json_sidecar.pop("MultiphaseIndex", None)

                # do some things for background suppression
                if json_sidecar.get("BackgroundSuppression", False):
                    # calculate the inversion pulse timings with respect to the start of
                    # the labelling pulse (which occurs
                    # LabelingDuration + PostLabelingDelay before the excitation pulse)
                    # use the maximum PLD to cover case where data is multiphase
                    max_pld = np.max(json_sidecar["PostLabelingDelay"])
                    label_and_pld_dur = json_sidecar["LabelingDuration"] + max_pld
                    inv_pulse_times = label_and_pld_dur - np.asarray(
                        image.metadata[
                            BackgroundSuppressionFilter.M_BSUP_INV_PULSE_TIMING
                        ]
                    )

                    json_sidecar[
                        "BackgroundSuppressionPulseTime"
                    ] = inv_pulse_times.tolist()
                    json_sidecar[
                        "BackgroundSuppressionNumberPulses"
                    ] = inv_pulse_times.size
                else:
                    json_sidecar["BackgroundSuppression"] = False

                # set vascular crushing to false
                json_sidecar["VascularCrushing"] = False

                # validate the sidecar against the ASL BIDS schema
                # load in the ASL BIDS schema
                with open(ASL_BIDS_SCHEMA) as file:
                    asl_bids_schema = json.load(file)

                validate(instance=json_sidecar, schema=asl_bids_schema)

            elif modality_label == "m0scan":
                # set the ImageType field
                json_sidecar["ImageType"] = [
                    "ORIGINAL",
                    "PRIMARY",
                    "PROTON_DENSITY",
                    "NONE",
                ]

                # validate the sidecar against the ASL BIDS schema
                # load in the ASL BIDS schema
                with open(M0SCAN_BIDS_SCHEMA) as file:
                    m0scan_bids_schema = json.load(file)

                validate(instance=json_sidecar, schema=m0scan_bids_schema)

        ## Series type 'structural'
        elif image.metadata[self.SERIES_TYPE] == STRUCTURAL:
            sub_directory = self.STRUCT_SUBDIR
            modality_label = image.metadata[MODALITY]
            json_sidecar["ImageType"] = [
                "ORIGINAL",
                "PRIMARY",
                modality_label.upper(),
                "NONE",
            ]

        ## Series type 'ground_truth'
        elif image.metadata[self.SERIES_TYPE] == GROUND_TRUTH:
            sub_directory = self.GT_SUBDIR
            # set the modality label using QUANTITY_MAPPING
            modality_label = self.QUANTITY_MAPPING[
                image.metadata[GroundTruthLoaderFilter.KEY_QUANTITY]
            ]
            json_sidecar["Quantity"] = modality_label

            # if there is a LabelMap field, use LABEL_MAP_MAPPING to change the subfield names to
            # the BIDS standard
            if json_sidecar.get("LabelMap") is not None:
                json_sidecar["LabelMap"] = map_dict(
                    json_sidecar["LabelMap"],
                    io_map=self.LABEL_MAP_MAPPING,
                    io_map_optional=True,
                )
            json_sidecar["ImageType"] = [
                "ORIGINAL",
                "PRIMARY",
                modality_label.upper(),
                "NONE",
            ]

        # if it doesn't exist make the sub-directory
        if not os.path.exists(os.path.join(output_directory, sub_directory)):
            os.makedirs(os.path.join(output_directory, sub_directory))

        # construct filenames
        base_filename = (
            subject_string
            + "_"
            + filename_prefix
            + series_number_string
            + "_"
            + modality_label
        )
        nifti_filename = base_filename + ".nii.gz"
        json_filename = base_filename + ".json"

        # turn the nifti and json filenames into full paths
        nifti_filename = os.path.join(output_directory, sub_directory, nifti_filename)
        # write the nifti file
        logger.info("saving %s", nifti_filename)
        nib.save(image.nifti_image, nifti_filename)

        json_filename = os.path.join(output_directory, sub_directory, json_filename)
        # write the json sidecar
        logger.info("saving %s", json_filename)
        with open(json_filename, "w") as json_file:
            json.dump(json_sidecar, json_file, indent=4)

        # add filenames to outputs
        self.outputs[self.KEY_FILENAME] = [nifti_filename, json_filename]
        if "asl_context_filename" in locals():
            self.outputs[self.KEY_FILENAME].append(asl_context_filename)
            logger.info("saving %s", asl_context_filename)
            with open(asl_context_filename, "w") as tsv_file:
                tsv_file.write(asl_context_tsv)
                tsv_file.close()

        self.outputs[self.KEY_SIDECAR] = json_sidecar

        # if it doesn't exist, save the dataset_description.json file in the root output folder
        dataset_description_filename = os.path.join(
            self.inputs[self.KEY_OUTPUT_DIRECTORY], "dataset_description.json"
        )
        if not os.path.isfile(dataset_description_filename):
            # write the json sidecar
            logger.info("saving %s", dataset_description_filename)
            with open(dataset_description_filename, "w") as json_file:
                json.dump(self.DATASET_DESCRIPTION, json_file, indent=4)

        # if it doesn't exist, save the canned text into README in the root output folder
        readme_filename = os.path.join(self.inputs[self.KEY_OUTPUT_DIRECTORY], "README")
        if not os.path.isfile(readme_filename):
            logger.info("saving %s", readme_filename)
            with open(readme_filename, mode="w") as readme_file:
                readme_file.write(self.readme_canned_text())
                readme_file.close()

        # now append to readme some information about the image series
        with open(readme_filename, mode="a") as readme_file:
            readme_file.write(
                f"{image.metadata[self.SERIES_NUMBER]}. {modality_label}: {image.metadata.get(self.SERIES_DESCRIPTION)}\n"
            )
            readme_file.close()

        # Finally, create a .bidsignore file to deal with anything non-canon
        bidsignore_filename = os.path.join(
            self.inputs[self.KEY_OUTPUT_DIRECTORY], ".bidsignore"
        )
        if not os.path.isfile(bidsignore_filename):
            with open(bidsignore_filename, mode="w") as bidsignore_file:
                bidsignore_file.writelines(
                    "\n".join(
                        [
                            "ground_truth/",
                            "**/*Perfmap*",
                            "**/*ATTmap*",
                            "**/*Lambdamap*",
                        ]
                    )
                )
                bidsignore_file.close()

    def _validate_inputs(self):
        """ Checks that the inputs meet their validation critera
        'image' must be a derived from BaseImageContainer
        'output_directory' must be a string and a path
        'filename_prefix' must be a string and is optional

        Also checks the input image's metadata
        """

        input_validator = ParameterValidator(
            parameters={
                self.KEY_IMAGE: Parameter(
                    validators=isinstance_validator(BaseImageContainer)
                ),
                self.KEY_OUTPUT_DIRECTORY: Parameter(
                    validators=isinstance_validator(str)
                ),
                self.KEY_FILENAME_PREFIX: Parameter(
                    validators=isinstance_validator(str),
                    optional=True,
                    default_value="",
                ),
                self.KEY_SUBJECT_LABEL: Parameter(
                    validators=[
                        isinstance_validator(str),
                        regex_validator("^[A-Za-z0-9\\-]+$"),
                    ],
                    optional=True,
                    default_value="001",
                ),
            }
        )
        # validate the inputs
        new_params = input_validator.validate(
            self.inputs, error_type=FilterInputValidationError
        )

        metdata_validator = ParameterValidator(
            parameters={
                self.SERIES_TYPE: Parameter(
                    validators=from_list_validator(SUPPORTED_IMAGE_TYPES)
                ),
                MODALITY: Parameter(
                    validators=from_list_validator(SUPPORTED_STRUCT_MODALITY_LABELS),
                    optional=True,
                ),
                self.SERIES_NUMBER: Parameter(
                    validators=[
                        isinstance_validator(int),
                        greater_than_equal_to_validator(0),
                    ]
                ),
                ASL_CONTEXT: Parameter(
                    validators=isinstance_validator((str, list)), optional=True,
                ),
                GkmFilter.KEY_LABEL_TYPE: Parameter(
                    validators=isinstance_validator(str), optional=True,
                ),
                GkmFilter.KEY_LABEL_DURATION: Parameter(
                    validators=isinstance_validator(float), optional=True
                ),
                GkmFilter.KEY_BOLUS_CUT_OFF_DELAY_TIME: Parameter(
                    validators=or_validator(
                        [isinstance_validator(float), non_empty_list_validator()]
                    ),
                    optional=True,
                ),
                GkmFilter.KEY_BOLUS_CUT_OFF_FLAG: Parameter(
                    validators=isinstance_validator(bool), optional=True
                ),
                GkmFilter.KEY_POST_LABEL_DELAY: Parameter(
                    validators=or_validator(
                        [isinstance_validator(float), non_empty_list_validator()]
                    ),
                    optional=True,
                ),
                GkmFilter.KEY_LABEL_EFFICIENCY: Parameter(
                    validators=isinstance_validator(float), optional=True
                ),
                GroundTruthLoaderFilter.KEY_QUANTITY: Parameter(
                    validators=isinstance_validator(str), optional=True,
                ),
                GroundTruthLoaderFilter.KEY_UNITS: Parameter(
                    validators=isinstance_validator(str), optional=True,
                ),
                "image_flavour": Parameter(
                    validators=isinstance_validator(str), optional=True,
                ),
                BackgroundSuppressionFilter.M_BSUP_INV_PULSE_TIMING: Parameter(
                    validators=isinstance_validator(bool), optional=True
                ),
                BackgroundSuppressionFilter.M_BSUP_INV_PULSE_TIMING: Parameter(
                    validators=for_each_validator(greater_than_equal_to_validator(0)),
                    optional=True,
                ),
                "multiphase_index": Parameter(
                    validators=or_validator(
                        [
                            for_each_validator(isinstance_validator(int)),
                            isinstance_validator(int),
                        ]
                    ),
                    optional=True,
                ),
            }
        )
        # validate the metadata
        metadata = self.inputs[self.KEY_IMAGE].metadata
        metdata_validator.validate(metadata, error_type=FilterInputValidationError)

        # Specific validation for series_type == "structural"
        if metadata[self.SERIES_TYPE] == STRUCTURAL:
            if metadata.get(MODALITY) is None:
                raise FilterInputValidationError(
                    "metadata field 'modality' is required when `series_type` is 'structural'"
                )

        # specific validation when series_type is "ground_truth"
        if metadata[self.SERIES_TYPE] == GROUND_TRUTH:
            if metadata.get(GroundTruthLoaderFilter.KEY_QUANTITY) is None:
                raise FilterInputValidationError(
                    "metadata field 'quantity' is required when `series_type` is 'ground_truth'"
                )
        if metadata[self.SERIES_TYPE] == GROUND_TRUTH:
            if metadata.get(GroundTruthLoaderFilter.KEY_UNITS) is None:
                raise FilterInputValidationError(
                    "metadata field 'units' is required when `series_type` is 'ground_truth'"
                )

        # Specific validation for series_type == "asl"
        if metadata[self.SERIES_TYPE] == ASL:
            # asl_context needs some further validating
            asl_context = metadata.get(ASL_CONTEXT)
            if asl_context is None:
                raise FilterInputValidationError(
                    "metadata field 'asl_context' is required when `series_type` is 'asl'"
                )
            if isinstance(asl_context, str):
                asl_context_validator = ParameterValidator(
                    parameters={
                        ASL_CONTEXT: Parameter(
                            validators=from_list_validator(SUPPORTED_ASL_CONTEXTS),
                        ),
                    }
                )

            elif isinstance(asl_context, list):
                asl_context_validator = ParameterValidator(
                    parameters={
                        ASL_CONTEXT: Parameter(
                            validators=for_each_validator(
                                from_list_validator(SUPPORTED_ASL_CONTEXTS)
                            ),
                        ),
                    }
                )
            asl_context_validator.validate(
                {"asl_context": asl_context}, error_type=FilterInputValidationError
            )

            # determine the modality_label based on asl_context
            modality_label = self.determine_asl_modality_label(asl_context)

            if modality_label == ASL:
                # do some checking for when the `modality` is 'asl'
                if metadata.get(GkmFilter.KEY_LABEL_TYPE) is None:
                    raise FilterInputValidationError(
                        "metadata field 'label_type' is required for 'series_type'"
                        + " and 'modality' is 'asl'"
                    )
                if metadata.get(GkmFilter.KEY_POST_LABEL_DELAY) is None:
                    raise FilterInputValidationError(
                        "metadata field 'post_label_delay' is required for 'series_type'"
                        + " and 'modality' is 'asl'"
                    )
                if metadata.get("image_flavour") is None:
                    raise FilterInputValidationError(
                        "metadata field 'image_flavour' is required for 'series_type'"
                        + " and 'modality' is 'asl'"
                    )
                # if "background_suppression" is True then additional parameters are required
                if metadata.get(BackgroundSuppressionFilter.M_BACKGROUND_SUPPRESSION):
                    # check that 'background_suppression' actually is a bool and not an int
                    # pdb.set_trace()
                    if not isinstance(
                        metadata.get(
                            BackgroundSuppressionFilter.M_BACKGROUND_SUPPRESSION
                        ),
                        bool,
                    ):
                        raise FilterInputValidationError(
                            "'BackgroundSuppression should be a bool"
                        )
                    if (
                        metadata.get(
                            BackgroundSuppressionFilter.M_BSUP_INV_PULSE_TIMING
                        )
                        is None
                    ):
                        raise FilterInputValidationError(
                            "metadata field 'background_suppression_inv_pulse_timing' is required "
                            "if 'background_suppression' is True"
                        )
                elif (
                    # Catch the case where it is 0 because False is a subclass of int
                    (
                        metadata.get(
                            BackgroundSuppressionFilter.M_BACKGROUND_SUPPRESSION
                        )
                        == 0
                    )
                    and not isinstance(
                        metadata.get(
                            BackgroundSuppressionFilter.M_BACKGROUND_SUPPRESSION
                        ),
                        bool,
                    )
                ):
                    raise FilterInputValidationError(
                        "'BackgroundSuppression should be a bool"
                    )

                # validation specific for multiphase ASL
                if isinstance(metadata.get(GkmFilter.KEY_POST_LABEL_DELAY), list):
                    if len(metadata.get(GkmFilter.KEY_POST_LABEL_DELAY)) > 1:
                        if metadata.get("multiphase_index") is None:
                            raise FilterInputValidationError(
                                "metadata field 'multiphase_index' is required for 'series_type'"
                                + "and 'modality' is 'asl', and 'post_label_delay' has length > 1"
                            )
                        if not len(metadata.get("multiphase_index")) == len(
                            metadata.get(GkmFilter.KEY_POST_LABEL_DELAY)
                        ):
                            raise FilterInputValidationError(
                                "For 'series_type' and 'modality' 'asl', and if 'post_label_delay' has length > 1"
                                "'multiphase_index' must have the same length as 'post_label_delay'"
                            )
                        # for each multiphase index, the values of post_label_delay should all be the
                        # same or none
                        unique_mpindex = np.unique(metadata["multiphase_index"])

                        for index in unique_mpindex:
                            mp_index_pld_vals = [
                                val
                                for idx, val in enumerate(metadata["post_label_delay"])
                                if metadata["multiphase_index"][idx] == index
                                and val is not None
                            ]
                            if not np.all(
                                np.asarray(mp_index_pld_vals) == mp_index_pld_vals[0]
                            ):
                                raise FilterInputValidationError(
                                    "For multiphase ASL data, at each multiphase index"
                                    "all non-None values of 'post_label_delay' must be the"
                                    "same "
                                )

                if metadata.get(GkmFilter.KEY_LABEL_TYPE) in (
                    GkmFilter.CASL,
                    GkmFilter.PCASL,
                ):
                    # validation specific to (p)casl

                    if metadata.get(GkmFilter.KEY_LABEL_DURATION) is None:
                        raise FilterInputValidationError(
                            "metadata field 'label_duration' is required for 'series_type'"
                            + "and 'modality' is 'asl', and 'label_type' is 'pcasl' or 'casl'"
                        )
                elif metadata.get(GkmFilter.KEY_LABEL_TYPE) == GkmFilter.PASL:
                    # validation specific to pasl
                    if metadata.get(GkmFilter.KEY_BOLUS_CUT_OFF_FLAG) is None:
                        raise FilterInputValidationError(
                            "metadata field 'bolus_cut_off_flag' is required for"
                            + " 'series_type and 'modality' is 'asl', "
                            + "and 'label_type' is 'pasl'"
                        )
                    if metadata.get(GkmFilter.KEY_BOLUS_CUT_OFF_FLAG):
                        if metadata.get(GkmFilter.KEY_BOLUS_CUT_OFF_DELAY_TIME) is None:
                            raise FilterInputValidationError(
                                "metadata field 'bolus_cut_off_delay_time' is required for"
                                + " 'series_type and 'modality' is 'asl', "
                                + "'label_type' is 'pasl', and 'bolus_cut_off_flag' is True"
                            )

        # Check that self.inputs[self.KEY_OUTPUT_DIRECTORY] is a valid path.
        if not os.path.exists(self.inputs[self.KEY_OUTPUT_DIRECTORY]):
            raise FilterInputValidationError(
                f"'output_directory' {self.inputs[self.KEY_OUTPUT_DIRECTORY]} does not exist"
            )

        # merge the updated parameters from the output with the input parameters
        self.inputs = {**self._i, **new_params}

    @staticmethod
    def determine_asl_modality_label(asl_context: Union[str, List[str]]) -> str:
        """Function that determines the modality_label for asl image types based
        on an input asl_context list

        :param asl_context: either a single string or list of asl context strings
            , e.g. ["m0scan", "control", "label"]
        :type asl_context: Union[str, List[str]]
        :return: a string determining the asl context, either "asl" or "m0scan"
        :rtype: str
        """
        # by default the modality label should be "asl"
        modality_label = ASL
        if isinstance(asl_context, str):
            if asl_context == "m0scan":
                modality_label = "m0scan"
        elif isinstance(asl_context, list):
            if all("m0scan" in s for s in asl_context):
                modality_label = "m0scan"
        return modality_label

    @staticmethod
    def readme_canned_text() -> str:
        """Outputs canned text for the dataset readme"""

        readme = """
This dataset has been generated by ASLDRO, software that can generate digital reference
objects for Arterial Spin Labelling (ASL) MRI.

It creates synthetic raw ASL data according to set acquisition and data format parameters, based
on input ground truth maps for:

* Perfusion rate
* Transit time
* Intrinsic MRI parameters: M0, T1, T2, T2\\*
* Tissue segmentation (defined as a single tissue type per voxel)

For more information on ASLDRO please visit:

* github: https://github.com/gold-standard-phantoms/asldro
* pypi: https://pypi.org/project/asldro/
* documentation: https://asldro.readthedocs.io/

This dataset comprises of the following image series:
"""
        return readme
