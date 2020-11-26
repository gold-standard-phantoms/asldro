""" Examples of filter chains """
import pprint
import logging
import shutil
from tempfile import TemporaryDirectory

import numpy as np

from asldro.containers.image import BaseImageContainer, NiftiImageContainer
from asldro.filters.ground_truth_loader import GroundTruthLoaderFilter
from asldro.filters.json_loader import JsonLoaderFilter
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.filters.gkm_filter import GkmFilter
from asldro.filters.invert_image_filter import InvertImageFilter
from asldro.filters.phase_magnitude_filter import PhaseMagnitudeFilter
from asldro.filters.transform_resample_image_filter import TransformResampleImageFilter
from asldro.filters.acquire_mri_image_filter import AcquireMriImageFilter
from asldro.filters.combine_time_series_filter import CombineTimeSeriesFilter
from asldro.filters.append_metadata_filter import AppendMetadataFilter
from asldro.filters.bids_output_filter import BidsOutputFilter
from asldro.data.filepaths import GROUND_TRUTH_DATA
from asldro.utils.general import splitext
from asldro.validators.schemas.index import SCHEMAS

from asldro.validators.user_parameter_input import (
    validate_input_params,
    get_example_input_params,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".zip", ".tar.gz"]
# Used in shutil.make_archive
EXTENSION_MAPPING = {".zip": "zip", ".tar.gz": "gztar"}


def run_full_pipeline(input_params: dict = None, output_filename: str = None):
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    # TODO: fix the above disabled linting errors
    """A function that runs the entire DRO pipeline. This
    can be extended as more functionality is included.
    This function is deliberately verbose to explain the
    operation, inputs and outputs of individual filters.
    :param input_params: The input parameter dictionary. If None, the defaults will be
    used
    :param output_filename: The output filename. Must be an zip/tar.gz archive. If None,
    no files will be generated.
    """

    if input_params is None:
        input_params = get_example_input_params()

    if output_filename is not None:
        _, output_filename_extension = splitext(output_filename)
        if output_filename_extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"File output type {output_filename_extension} not supported"
            )

    # Validate parameter and update defaults
    input_params = validate_input_params(input_params)

    json_filter = JsonLoaderFilter()
    json_filter.add_input(
        "filename", input_params["global_configuration"]["ground_truth"]["json"]
    )
    json_filter.add_input("schema", SCHEMAS["ground_truth"])
    nifti_filter = NiftiLoaderFilter()
    nifti_filter.add_input(
        "filename", input_params["global_configuration"]["ground_truth"]["nii"]
    )

    ground_truth_filter = GroundTruthLoaderFilter()
    ground_truth_filter.add_parent_filter(nifti_filter)
    ground_truth_filter.add_parent_filter(json_filter)
    # Pull out the parameters that might override values in the ground_truth
    ground_truth_filter.add_inputs(
        {
            "image_override": input_params["global_configuration"]["image_override"]
            if "image_override" in input_params["global_configuration"]
            else {},
            "parameter_override": input_params["global_configuration"][
                "parameter_override"
            ]
            if "parameter_override" in input_params["global_configuration"]
            else {},
            "ground_truth_modulate": input_params["global_configuration"][
                "ground_truth_modulate"
            ]
            if "ground_truth_modulate" in input_params["global_configuration"]
            else {},
        }
    )
    ground_truth_filter.run()

    logger.info("JsonLoaderFilter outputs:\n%s", pprint.pformat(json_filter.outputs))
    logger.debug("NiftiLoaderFilter outputs:\n%s", pprint.pformat(nifti_filter.outputs))
    logger.debug(
        "GroundTruthLoaderFilter outputs:\n%s",
        pprint.pformat(ground_truth_filter.outputs),
    )

    # create output lists to be populated in the "image_series" loop
    output_image_list = []
    # Loop over "image_series" in input_params
    # Take the asl image series and pass it to the remainder of the pipeline
    # update the input_params variable so it contains the asl series parameters
    for series_index, image_series in enumerate(input_params["image_series"]):
        series_number = series_index + 1

        ############################################################################################
        # ASL pipeline
        # Comprises GKM, then MRI signal model, transform and resampling,
        # and noise for each dynamic.
        # After the 'acquisition loop' the dynamics are concatenated into a single 4D file
        if image_series["series_type"] == "asl":
            asl_params = image_series["series_parameters"]
            # initialise the random number generator for the image series
            np.random.seed(image_series["series_parameters"]["random_seed"])
            logger.info(
                "Running DRO generation with the following parameters:\n%s",
                pprint.pformat(asl_params),
            )

            # Run the GkmFilter on the ground_truth data
            gkm_filter = GkmFilter()
            # Add ground truth parameters from the ground_truth_filter: perfusion_rate, transit_time
            # m0,lambda_blood_brain, t1_arterial_blood all have the same keys; t1 maps
            # to t1_tissue
            gkm_filter.add_parent_filter(
                parent=ground_truth_filter,
                io_map={
                    "perfusion_rate": gkm_filter.KEY_PERFUSION_RATE,
                    "transit_time": gkm_filter.KEY_TRANSIT_TIME,
                    "m0": gkm_filter.KEY_M0,
                    "t1": gkm_filter.KEY_T1_TISSUE,
                    "lambda_blood_brain": gkm_filter.KEY_LAMBDA_BLOOD_BRAIN,
                    "t1_arterial_blood": gkm_filter.KEY_T1_ARTERIAL_BLOOD,
                },
            )
            # Add parameters from the input_params: label_type, signal_time, label_duration and
            # label_efficiency all have the same keys
            gkm_filter.add_inputs(asl_params)
            # reverse the polarity of delta_m.image for encoding it into the label signal
            invert_delta_m_filter = InvertImageFilter()
            invert_delta_m_filter.add_parent_filter(
                parent=gkm_filter, io_map={gkm_filter.KEY_DELTA_M: "image"}
            )
            # Create one-time data that is required by the Acquisition Loop
            # 1. m0 resampled at the acquisition resolution
            m0_resample_filter = TransformResampleImageFilter()
            m0_resample_filter.add_parent_filter(
                ground_truth_filter,
                io_map={"m0": TransformResampleImageFilter.KEY_IMAGE},
            )
            m0_resample_filter.add_input(
                TransformResampleImageFilter.KEY_TARGET_SHAPE,
                tuple(asl_params["acq_matrix"]),
            )

            # Acquisition Loop: loop over ASL context, run the AcquireMriImageFilter and put the
            # output image into a list
            # acquired_images_list: List[nib.Nifti2Image] = []
            combine_time_series_filter = CombineTimeSeriesFilter()
            for idx, asl_context in enumerate(asl_params["asl_context"].split()):
                acquire_mri_image_filter = AcquireMriImageFilter()
                # map inputs from the ground truth: t1, t2, t2_star, m0 all share the same name
                # so no explicit mapping is necessary.
                acquire_mri_image_filter.add_parent_filter(parent=ground_truth_filter)

                # map inputs from asl_params. acq_contrast, excitation_flip_angle, desired_snr,
                # inversion_time, inversion_flip_angle (last 2 are optional)
                acquire_mri_image_filter.add_inputs(
                    asl_params,
                    io_map={
                        "acq_contrast": AcquireMriImageFilter.KEY_ACQ_CONTRAST,
                        "excitation_flip_angle": AcquireMriImageFilter.KEY_EXCITATION_FLIP_ANGLE,
                        "desired_snr": AcquireMriImageFilter.KEY_SNR,
                        "inversion_time": AcquireMriImageFilter.KEY_INVERSION_TIME,
                        "inversion_flip_angle": AcquireMriImageFilter.KEY_INVERSION_FLIP_ANGLE,
                    },
                    io_map_optional=True,
                )

                # if asl_context == "label" use the inverted delta_m as
                # the input MriSignalFilter.KEY_MAG_ENC
                if asl_context.lower() == "label":
                    acquire_mri_image_filter.add_parent_filter(
                        parent=invert_delta_m_filter,
                        io_map={"image": AcquireMriImageFilter.KEY_MAG_ENC},
                    )
                # set the image flavour to "PERFUSION"
                acquire_mri_image_filter.add_input(
                    AcquireMriImageFilter.KEY_IMAGE_FLAVOUR, "PERFUSION"
                )
                # build acquisition loop parameter dictionary - parameters that cannot be directly
                # mapped
                acq_loop_params = {
                    AcquireMriImageFilter.KEY_ECHO_TIME: asl_params["echo_time"][idx],
                    AcquireMriImageFilter.KEY_REPETITION_TIME: asl_params[
                        "repetition_time"
                    ][idx],
                    AcquireMriImageFilter.KEY_ROTATION: (
                        asl_params["rot_x"][idx],
                        asl_params["rot_y"][idx],
                        asl_params["rot_z"][idx],
                    ),
                    AcquireMriImageFilter.KEY_TRANSLATION: (
                        asl_params["transl_x"][idx],
                        asl_params["transl_y"][idx],
                        asl_params["transl_z"][idx],
                    ),
                    AcquireMriImageFilter.KEY_TARGET_SHAPE: tuple(
                        asl_params["acq_matrix"]
                    ),
                }
                # add these inputs to the filter
                acquire_mri_image_filter.add_inputs(acq_loop_params)
                # map the reference_image for the noise generation to the m0 ground truth.
                acquire_mri_image_filter.add_parent_filter(
                    m0_resample_filter,
                    io_map={
                        m0_resample_filter.KEY_IMAGE: AcquireMriImageFilter.KEY_REF_IMAGE
                    },
                )
                phase_magnitude_filter = PhaseMagnitudeFilter()
                phase_magnitude_filter.add_parent_filter(
                    parent=acquire_mri_image_filter
                )

                append_metadata_filter = AppendMetadataFilter()
                append_metadata_filter.add_parent_filter(
                    phase_magnitude_filter, io_map={"magnitude": "image"}
                )
                append_metadata_filter.add_input(
                    AppendMetadataFilter.KEY_METADATA,
                    {
                        "series_description": image_series["series_description"],
                        "series_type": image_series["series_type"],
                        "series_number": series_number,
                        "asl_context": asl_context,
                    },
                )

                # Add the acqusition pipeline to the combine time series filter after
                # calculating the magnitude component of the time series data
                combine_time_series_filter.add_parent_filter(
                    parent=append_metadata_filter, io_map={"image": f"image_{idx}"}
                )

            combine_time_series_filter.run()
            acquired_timeseries_nifti_container: NiftiImageContainer = (
                combine_time_series_filter.outputs["image"].as_nifti()
            )
            acquired_timeseries_nifti_container.header["descrip"] = image_series[
                "series_description"
            ]

            # place in output_nifti list
            output_image_list.append(acquired_timeseries_nifti_container)
            # logging
            logger.debug("GkmFilter outputs: \n %s", pprint.pformat(gkm_filter.outputs))
            logger.debug(
                "combine_time_series_filter outputs: \n %s",
                pprint.pformat(combine_time_series_filter.outputs),
            )

        ############################################################################################
        # Structural pipeline
        # Comprises MRI signal,transform and resampling and noise models
        if image_series["series_type"] == "structural":
            struct_params = image_series["series_parameters"]
            # initialise the random number generator for the image series
            np.random.seed(image_series["series_parameters"]["random_seed"])

            logger.info(
                "Running DRO generation with the following parameters:\n%s",
                pprint.pformat(struct_params),
            )

            # Simulate acquisition
            acquire_mri_image_filter = AcquireMriImageFilter()
            # map inputs from the ground truth: t1, t2, t2_star, m0 all share the same name
            # so no explicit mapping is necessary.
            acquire_mri_image_filter.add_parent_filter(parent=ground_truth_filter)

            # append struct_params with additional parameters that need to be built/modified
            struct_params = {
                **struct_params,
                **{
                    AcquireMriImageFilter.KEY_ROTATION: (
                        struct_params["rot_x"],
                        struct_params["rot_y"],
                        struct_params["rot_z"],
                    ),
                    AcquireMriImageFilter.KEY_TRANSLATION: (
                        struct_params["transl_x"],
                        struct_params["transl_y"],
                        struct_params["transl_z"],
                    ),
                    AcquireMriImageFilter.KEY_TARGET_SHAPE: tuple(
                        struct_params["acq_matrix"]
                    ),
                    AcquireMriImageFilter.KEY_SNR: struct_params["desired_snr"],
                },
            }

            # map inputs from struct_params. acq_contrast, excitation_flip_angle, desired_snr,
            # inversion_time, inversion_flip_angle (last 2 are optional)
            acquire_mri_image_filter.add_inputs(
                struct_params,
                io_map_optional=True,
            )

            append_metadata_filter = AppendMetadataFilter()
            append_metadata_filter.add_parent_filter(acquire_mri_image_filter)
            append_metadata_filter.add_input(
                AppendMetadataFilter.KEY_METADATA,
                {
                    "series_description": image_series["series_description"],
                    "modality": struct_params["modality"],
                    "series_type": image_series["series_type"],
                    "series_number": series_number,
                },
            )

            if struct_params["output_image_type"] == "magnitude":
                phase_magnitude_filter = PhaseMagnitudeFilter()
                phase_magnitude_filter.add_parent_filter(append_metadata_filter)
                phase_magnitude_filter.run()
                struct_image_container = phase_magnitude_filter.outputs[
                    PhaseMagnitudeFilter.KEY_MAGNITUDE
                ]
            else:
                append_metadata_filter.run()
                struct_image_container = append_metadata_filter.outputs[
                    AcquireMriImageFilter.KEY_IMAGE
                ]

            struct_image_container.header["descrip"] = image_series[
                "series_description"
            ]
            # Append list of the output images
            output_image_list.append(struct_image_container)

        ############################################################################################
        # Ground truth pipeline
        # Comprises resampling all of the ground truth images with the specified resampling
        # parameters
        if image_series["series_type"] == "ground_truth":
            ground_truth_params = image_series["series_parameters"]
            logger.info(
                "Running DRO generation with the following parameters:\n%s",
                pprint.pformat(ground_truth_params),
            )
            # Loop over all the ground truth images and resample as specified
            ground_truth_keys = ground_truth_filter.outputs.keys()
            ground_truth_image_keys = [
                key
                for key in ground_truth_keys
                if isinstance(ground_truth_filter.outputs[key], BaseImageContainer)
            ]
            for quantity in ground_truth_image_keys:
                resample_filter = TransformResampleImageFilter()
                # map the ground_truth_filter to the resample filter
                resample_filter.add_parent_filter(
                    ground_truth_filter, io_map={quantity: "image"}
                )

                ground_truth_params = {
                    **ground_truth_params,
                    **{
                        TransformResampleImageFilter.KEY_ROTATION: (
                            ground_truth_params["rot_x"],
                            ground_truth_params["rot_y"],
                            ground_truth_params["rot_z"],
                        ),
                        TransformResampleImageFilter.KEY_TRANSLATION: (
                            ground_truth_params["transl_x"],
                            ground_truth_params["transl_y"],
                            ground_truth_params["transl_z"],
                        ),
                        AcquireMriImageFilter.KEY_TARGET_SHAPE: tuple(
                            ground_truth_params["acq_matrix"]
                        ),
                    },
                }
                resample_filter.add_inputs(ground_truth_params)
                resample_filter.run()
                append_metadata_filter = AppendMetadataFilter()
                append_metadata_filter.add_parent_filter(resample_filter)
                append_metadata_filter.add_input(
                    AppendMetadataFilter.KEY_METADATA,
                    {
                        "series_description": image_series["series_description"],
                        "series_type": image_series["series_type"],
                        "series_number": series_number,
                    },
                )
                # Run the append_metadata_filter to generate an acquired volume
                append_metadata_filter.run()
                # append to output image list
                output_image_list.append(
                    append_metadata_filter.outputs[AppendMetadataFilter.KEY_IMAGE]
                )

    # Output everything to a temporary directory
    with TemporaryDirectory() as temp_dir:
        for idx, image_to_output in enumerate(output_image_list):
            bids_output_filter = BidsOutputFilter()
            bids_output_filter.add_input(
                BidsOutputFilter.KEY_OUTPUT_DIRECTORY, temp_dir
            )
            bids_output_filter.add_input(BidsOutputFilter.KEY_IMAGE, image_to_output)
            # run the filter to write the BIDS files to disk
            bids_output_filter.run()

        if output_filename is not None:
            filename, file_extension = splitext(output_filename)
            # output the file archive
            logger.info("Creating output archive: %s", output_filename)
            shutil.make_archive(
                filename, EXTENSION_MAPPING[file_extension], root_dir=temp_dir
            )


if __name__ == "__main__":
    run_full_pipeline()
