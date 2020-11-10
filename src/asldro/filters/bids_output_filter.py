""" BidsOutputFilter """

import os
import logging

import nibabel as nib
from datetime import datetime, timezone
from io import StringIO
import json

from asldro.containers.image import BaseImageContainer
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

from asldro import __version__ as asldro_version


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

    Files will be saved in subdirectories corresponding to the metadata entry 'series_type':
    * 'structural' will be saved in the subdirectory 'anat'
    * 'asl' will be saved in the subdirectory 'asl'
    * 'ground_truth' will be saved in the subdirectory 'ground_truth'
    
    Filenames will be given by: <series_number>_<filename_prefix>_<modality_label>.<ext>, where
    
    * <series_number> is an integer number and will be prefixed by zeros so that it is 3 characters
      long, for example 003, 010, 243
    * <filename_prefix> is the string supplied by the input `filename_prefix`
    * <modality_label> is an entry in the input image's metadata.  If it is not present this will be
      left blank

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
        MriSignalFilter.KEY_ACQ_CONTRAST: "PulseSequenceType",
        SERIES_DESCRIPTION: "SeriesDescription",
        SERIES_NUMBER: "SeriesNumber",
        TransformResampleImageFilter.VOXEL_SIZE: "AcquisitionVoxelSize",
    }

    ACQ_CONTRAST_MAPPING = {
        MriSignalFilter.CONTRAST_GE: "Gradient Echo",
        MriSignalFilter.CONTRAST_SE: "Spin Echo",
        MriSignalFilter.CONTRAST_IR: "Inversion Recovery",
    }

    def __init__(self):
        super().__init__(name="BIDS Output")

    def _run(self):
        """ Writes the input image to disk in BIDS format """
        image: BaseImageContainer = self.inputs[self.KEY_IMAGE]
        output_directory = self.inputs[self.KEY_OUTPUT_DIRECTORY]
        # map the image metadata to the json sidecar
        json_sidecar = map_dict(image.metadata, self.BIDS_MAPPING, io_map_optional=True)

        # construct filenames
        series_number_string = f"_{image.metadata[self.SERIES_NUMBER]:03d}"
        nifti_filename = (
            f"{self.inputs[self.KEY_FILENAME_PREFIX]}"
            + series_number_string
            + f"_{image.metadata[MODALITY]}.nii.gz"
        )
        json_filename = (
            f"{self.inputs[self.KEY_FILENAME_PREFIX]}"
            + series_number_string
            + f"_{image.metadata[MODALITY]}.json"
        )

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
        # set the PulseSequenceType value according to the BIDS spec
        if json_sidecar.get("PulseSequenceType") is not None:
            json_sidecar["PulseSequenceType"] = self.ACQ_CONTRAST_MAPPING[
                json_sidecar["PulseSequenceType"]
            ]

        # series_type specific things
        if image.metadata[self.SERIES_TYPE] == ASL:
            # ASL series, create aslcontext.tsv string
            sub_directory = "asl"
            if image.metadata[MODALITY] == ASL:
                # create _aslcontext_tsv
                asl_context_tsv = "volume_type\n" + "\n".join(
                    image.metadata[ASL_CONTEXT]
                )
                asl_context_filename = os.path.join(
                    output_directory,
                    sub_directory,
                    f"{self.inputs[self.KEY_FILENAME_PREFIX]}"
                    + series_number_string
                    + "_aslcontext.tsv",
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

        elif image.metadata[self.SERIES_TYPE] == STRUCTURAL:
            sub_directory = "anat"

        elif image.metadata[self.SERIES_TYPE] == GROUND_TRUTH:
            sub_directory = "ground_truth"

        # make the sub-directory
        os.makedirs(os.path.join(output_directory, sub_directory))
        # turn the nifti and json filenames into full paths
        nifti_filename = os.path.join(output_directory, sub_directory, nifti_filename)
        json_filename = os.path.join(output_directory, sub_directory, json_filename)

        # write the nifti file
        nib.save(image.nifti_image, nifti_filename)
        # write the json sidecar
        with open(json_filename, "w") as json_file:
            json.dump(json_sidecar, json_file, indent=4)

        # add filenames to outputs
        self.outputs[self.KEY_FILENAME] = [nifti_filename, json_filename]
        if "asl_context_filename" in locals():
            self.outputs[self.KEY_FILENAME].append(asl_context_filename)
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
                MODALITY: Parameter(validators=isinstance_validator(str)),
                self.SERIES_NUMBER: Parameter(
                    validators=[
                        isinstance_validator(int),
                        greater_than_equal_to_validator(0),
                    ]
                ),
                ASL_CONTEXT: Parameter(
                    validators=for_each_validator(
                        from_list_validator(SUPPORTED_ASL_CONTEXTS)
                    ),
                    optional=True,
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
            }
        )
        # validate the metadata
        metdata_validator.validate(
            self.inputs[self.KEY_IMAGE].metadata, error_type=FilterInputValidationError
        )

        # Specific validation for series_type == "asl"
        if self.inputs[self.KEY_IMAGE].metadata[self.SERIES_TYPE] == ASL:
            if self.inputs[self.KEY_IMAGE].metadata[MODALITY] == ASL:
                # do some checking for when the modality is asl
                if self.inputs[self.KEY_IMAGE].metadata.get(ASL_CONTEXT) == None:
                    raise FilterInputValidationError(
                        "metadata field 'asl_context' is required for 'series_type'"
                        + " and 'modality' == 'asl'"
                    )
                if (
                    self.inputs[self.KEY_IMAGE].metadata.get(GkmFilter.KEY_LABEL_TYPE)
                    == None
                ):
                    raise FilterInputValidationError(
                        "metadata field 'label_type' is required for 'series_type'"
                        + " and 'modality' == 'asl'"
                    )
                if (
                    self.inputs[self.KEY_IMAGE].metadata.get(
                        GkmFilter.KEY_LABEL_DURATION
                    )
                    == None
                ):
                    raise FilterInputValidationError(
                        "metadata field 'label_duration' is required for 'series_type'"
                        + " and 'modality' == 'asl'"
                    )
                if (
                    self.inputs[self.KEY_IMAGE].metadata.get(
                        GkmFilter.KEY_POST_LABEL_DELAY
                    )
                    == None
                ):
                    raise FilterInputValidationError(
                        "metadata field 'post_label_delay' is required for 'series_type'"
                        + " and 'modality' == 'asl'"
                    )

        # Check that self.inputs[self.KEY_OUTPUT_DIRECTORY] is a valid path.
        if os.path.exists(self.inputs[self.KEY_OUTPUT_DIRECTORY]) == False:
            raise FilterInputValidationError(
                f"'output_directory' {self.inputs[self.KEY_OUTPUT_DIRECTORY]} does not exist"
            )

        # merge the updated parameters from the output with the input parameters
        self.inputs = {**self._i, **new_params}

