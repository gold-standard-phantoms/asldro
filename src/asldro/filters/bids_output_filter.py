""" BidsOutputFilter """

import os
import logging
from typing import Union, List
from datetime import datetime, timezone
import json
from jsonschema import validate

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
)
from asldro.data.filepaths import ASL_BIDS_SCHEMA, M0SCAN_BIDS_SCHEMA

from asldro import __version__ as asldro_version

logger = logging.getLogger(__name__)


class BidsOutputFilter(BaseFilter):
    """ A filter that will output an input image container in Brain Imaging Data Structure
    (BIDS) format.

    BIDS comprises of a NIFTI image file and accompanying .json sidecar that contains additional
    parameters.  More information on BIDS can be found at https://bids.neuroimaging.io/

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`BidsOutputFilter.add_inputs()`
    member function. They are also accessible via class constants,
    for example :class:`BidsOutputFilter.KEY_IMAGE`

    :param 'image': the image to save in BIDS format
    :type 'image': BaseImageContainer
    :param 'output_directory': The root directory to save to
    :type 'output_directory': str
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
    * 'asl' will be saved in the subdirectory 'asl'
    * 'ground_truth' will be saved in the subdirectory 'ground_truth'

    Filenames will be given by: <series_number>_<filename_prefix>_<modality_label>.<ext>, where

    * <series_number> is given by metadata field ``series_number``, which is an integer number
      and will be prefixed by zeros so that it is 3 characterslong, for example 003, 010, 243
    * <filename_prefix> is the string supplied by the input ``filename_prefix``
    * <modality_label> is determined based on ``series_type``:

        * 'structural': it is given by the metadata field ``modality``.
        * 'asl': it is determined by asl_context.  If asl_context only contains entries that match
          with 'm0scan' then it will be set to 'm0scan', otherwise 'asl'.
        * 'ground_truth': it will be a concatenation of 'ground_truth_' + the metadata field
          ``quantity``, e.g. 'ground_truth_t1'.

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
        seconds.
    :type 'post_label_delay': float
    :param 'label_efficiency': the degree of inversion of the magnetisation (between 0 and 1)
    :type 'label_efficiency': float
    :param 'image_flavour': a string that is used as the third entry in the BIDS field ``ImageType``
        (corresponding with the dicom tag (0008,0008).  For ASL images this should be 'PERFUSION'.
    "type 'image_flavour': str
    """

    # Key Constants
    KEY_IMAGE = "image"
    KEY_OUTPUT_DIRECTORY = "output_directory"
    KEY_FILENAME_PREFIX = "filename_prefix"
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

    # metadata parameters to BIDS fields mapping dictionary
    BIDS_MAPPING = {
        GkmFilter.KEY_LABEL_TYPE: "LabelingType",
        GkmFilter.KEY_LABEL_DURATION: "LabelingDuration",
        GkmFilter.KEY_LABEL_EFFICIENCY: "LabelingEfficiency",
        GkmFilter.KEY_POST_LABEL_DELAY: "PostLabelingDelay",
        MriSignalFilter.KEY_ECHO_TIME: "EchoTime",
        MriSignalFilter.KEY_REPETITION_TIME: "RepetitionTime",
        MriSignalFilter.KEY_EXCITATION_FLIP_ANGLE: "FlipAngle",
        MriSignalFilter.KEY_INVERSION_TIME: "InversionTime",
        MriSignalFilter.KEY_ACQ_TYPE: "MrAcquisitionType",
        MriSignalFilter.KEY_ACQ_CONTRAST: "ScanningSequence",
        SERIES_DESCRIPTION: "SeriesDescription",
        SERIES_NUMBER: "SeriesNumber",
        TransformResampleImageFilter.VOXEL_SIZE: "AcquisitionVoxelSize",
        GroundTruthLoaderFilter.KEY_UNITS: "Units",
        GroundTruthLoaderFilter.KEY_MAG_STRENGTH: "MagneticFieldStrength",
        GroundTruthLoaderFilter.KEY_SEGMENTATION: "LabelMap",
        GroundTruthLoaderFilter.KEY_QUANTITY: "Quantity",
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

    def __init__(self):
        super().__init__(name="BIDS Output")

    def _run(self):
        """ Writes the input image to disk in BIDS format """
        image: BaseImageContainer = self.inputs[self.KEY_IMAGE]
        output_directory = self.inputs[self.KEY_OUTPUT_DIRECTORY]
        # map the image metadata to the json sidecar
        json_sidecar = map_dict(image.metadata, self.BIDS_MAPPING, io_map_optional=True)
        series_number_string = f"{image.metadata[self.SERIES_NUMBER]:03d}"
        # if the `filename_prefix` is not empty add an underscore after it
        if self.inputs[self.KEY_FILENAME_PREFIX] == "":
            filename_prefix = ""
        else:
            filename_prefix = self.inputs[self.KEY_FILENAME_PREFIX] + "_"
        # amend json sidecar
        # add ASLDRO information
        json_sidecar[self.DRO_SOFTWARE] = "ASLDRO"
        json_sidecar[self.DRO_SOFTWARE_VERSION] = asldro_version
        json_sidecar[self.DRO_SOFTWARE_URL] = [
            "code: https://github.com/gold-standard-phantoms/asldro",
            "pypi: https://pypi.org/project/asldro/",
            "docs: https://asldro.readthedocs.io/",
        ]
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
            sub_directory = "asl"

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
                    f"{filename_prefix}" + series_number_string + "_aslcontext.tsv",
                )
                # BIDS spec states LabelingType should be uppercase
                json_sidecar["LabelingType"] = json_sidecar["LabelingType"].upper()

                # set the BIDS field M0 correctly
                if any("m0scan" in s for s in image.metadata[ASL_CONTEXT]):
                    # if aslcontext contains one or more "m0scan" volumes set to True to indicate
                    # "WithinASL"
                    json_sidecar["M0"] = True
                elif isinstance(image.metadata["m0"], float):
                    # numerical value of m0 supplied so use this.
                    json_sidecar["M0"] = image.metadata["m0"]
                else:
                    # no numeric value or m0scan, so set to False
                    json_sidecar["M0"] = False

                # set the ImageType field
                json_sidecar["ImageType"] = [
                    "ORIGINAL",
                    "PRIMARY",
                    image.metadata["image_flavour"],
                    "NONE",
                ]

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
            sub_directory = "anat"
            modality_label = image.metadata[MODALITY]
            json_sidecar["ImageType"] = [
                "ORIGINAL",
                "PRIMARY",
                modality_label.upper(),
                "NONE",
            ]

        ## Series type 'ground_truth'
        elif image.metadata[self.SERIES_TYPE] == GROUND_TRUTH:
            sub_directory = "ground_truth"
            # set the modality label
            modality_label = (
                f"ground_truth_{image.metadata[GroundTruthLoaderFilter.KEY_QUANTITY]}"
            )
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
                image.metadata[GroundTruthLoaderFilter.KEY_QUANTITY].upper(),
                "NONE",
            ]

        # if it doesn't exist make the sub-directory
        if not os.path.exists(os.path.join(output_directory, sub_directory)):
            os.makedirs(os.path.join(output_directory, sub_directory))

        # construct filenames
        nifti_filename = (
            f"{filename_prefix}" + series_number_string + f"_{modality_label}.nii.gz"
        )
        json_filename = (
            f"{filename_prefix}" + series_number_string + f"_{modality_label}.json"
        )

        # turn the nifti and json filenames into full paths

        nifti_filename = os.path.join(output_directory, sub_directory, nifti_filename)
        # write the nifti file
        logger.info(f"saving {nifti_filename}")
        nib.save(image.nifti_image, nifti_filename)

        json_filename = os.path.join(output_directory, sub_directory, json_filename)
        # write the json sidecar
        logger.info(f"saving {json_filename}")
        with open(json_filename, "w") as json_file:
            json.dump(json_sidecar, json_file, indent=4)

        # add filenames to outputs
        self.outputs[self.KEY_FILENAME] = [nifti_filename, json_filename]
        if "asl_context_filename" in locals():
            self.outputs[self.KEY_FILENAME].append(asl_context_filename)
            logger.info(f"saving {asl_context_filename}")
            with open(asl_context_filename, "w") as tsv_file:
                tsv_file.write(asl_context_tsv)
                tsv_file.close()

        self.outputs[self.KEY_SIDECAR] = json_sidecar

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
                    validators=isinstance_validator(str), optional=True
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
                GkmFilter.KEY_POST_LABEL_DELAY: Parameter(
                    validators=isinstance_validator(float), optional=True
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
                if metadata.get(GkmFilter.KEY_LABEL_DURATION) is None:
                    raise FilterInputValidationError(
                        "metadata field 'label_duration' is required for 'series_type'"
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
