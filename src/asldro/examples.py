""" The main ASLDRO pipeline """
import pdb
import pprint
import logging
import shutil
from copy import deepcopy
from tempfile import TemporaryDirectory

import numpy as np

from asldro.containers.image import BaseImageContainer, NiftiImageContainer
from asldro.filters.background_suppression_filter import BackgroundSuppressionFilter
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
from asldro.utils.general import splitext, map_dict
from asldro.validators.schemas.index import SCHEMAS

from asldro.validators.user_parameter_input import (
    validate_input_params,
    get_example_input_params,
    BACKGROUND_SUPPRESSION,
    BS_SAT_PULSE_TIME,
    BS_INV_PULSE_TIMES,
    BS_PULSE_EFFICIENCY,
    BS_T1_OPT,
    BS_SAT_PULSE_TIME_OPT,
    BS_NUM_INV_PULSES,
    BS_APPLY_TO_ASL_CONTEXT,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".zip", ".tar.gz"]
# Used in shutil.make_archive
EXTENSION_MAPPING = {".zip": "zip", ".tar.gz": "gztar"}


def run_full_pipeline(input_params: dict = None, output_filename: str = None) -> dict:
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    # TODO: fix the above disabled linting errors
    """A function that runs the entire DRO pipeline. This
    can be extended as more functionality is included.
    This function is deliberately verbose to explain the
    operation, inputs and outputs of individual filters.

    :param input_params: The input parameter dictionary. If None, the defaults will be
      used.
    :param output_filename: The output filename. Must be an zip/tar.gz archive. If None,
      no files will be generated.

    :returns: A dictionary containing

      :'hrgt': the ground truth after modifications (outputs from the GroundTruthLoaderFilter)
      :'asldro_output': list of the image containers generated by the pipeline which would normally
        be saved in BIDS format

    :rtype: dict
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
    subject_label = input_params["global_configuration"]["subject_label"]

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
            # 2. determine if background suppression is to be performed
            # if so then generate the suppressed static magnetisation
            do_bs = False
            if asl_params[BACKGROUND_SUPPRESSION]:
                do_bs = True
                bs_params = map_dict(
                    asl_params[BACKGROUND_SUPPRESSION],
                    {
                        BS_SAT_PULSE_TIME: BackgroundSuppressionFilter.KEY_SAT_PULSE_TIME,
                        BS_INV_PULSE_TIMES: BackgroundSuppressionFilter.KEY_INV_PULSE_TIMES,
                        BS_PULSE_EFFICIENCY: BackgroundSuppressionFilter.KEY_PULSE_EFFICIENCY,
                        BS_NUM_INV_PULSES: BackgroundSuppressionFilter.KEY_NUM_INV_PULSES,
                        BS_T1_OPT: BackgroundSuppressionFilter.KEY_T1_OPT,
                    },
                    io_map_optional=True,
                )
                # if "sat_pulse_time_opt" is provided then some values need switching
                # round
                if (
                    asl_params[BACKGROUND_SUPPRESSION].get(BS_SAT_PULSE_TIME_OPT)
                    is not None
                ):
                    # "sat_pulse_time" goes to "mag_time", as this is the time we
                    # want to generate mangetisation at.
                    bs_params[BackgroundSuppressionFilter.KEY_MAG_TIME] = bs_params[
                        BackgroundSuppressionFilter.KEY_SAT_PULSE_TIME
                    ]
                    # "sat_pulse_time_opt" goes to "sat_pulse_time", because this is what
                    # we want to generate optimised times for
                    bs_params[
                        BackgroundSuppressionFilter.KEY_SAT_PULSE_TIME
                    ] = asl_params[BACKGROUND_SUPPRESSION][BS_SAT_PULSE_TIME_OPT]

                bs_filter = BackgroundSuppressionFilter()
                bs_filter.add_inputs(bs_params)
                bs_filter.add_parent_filter(
                    parent=ground_truth_filter,
                    io_map={
                        "m0": BackgroundSuppressionFilter.KEY_MAG_Z,
                        "t1": BackgroundSuppressionFilter.KEY_T1,
                    },
                )

            # initialise combine time series filter outside of both loops
            combine_time_series_filter = CombineTimeSeriesFilter()
            # if "signal_time" is a singleton, copy and place in a list

            if isinstance(asl_params["signal_time"], (float, int)):
                signal_time_list = [deepcopy(asl_params["signal_time"])]
            else:
                # otherwise it is a list so copy the entire list
                signal_time_list = deepcopy(asl_params["signal_time"])

            vol_index = 0

            # Multiphase ASL Loop: loop over signal_time_list
            for multiphase_index, t in enumerate(signal_time_list):
                # place the loop value for signal_time into the asl_params dict
                asl_params["signal_time"] = t
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

                # ASL Context Loop: loop over ASL context, run the AcquireMriImageFilter and put the
                # output image into the CombineTimeSeriesFilter

                for asl_context_index, asl_context in enumerate(
                    asl_params["asl_context"].split()
                ):
                    acquire_mri_image_filter = AcquireMriImageFilter()
                    # check that background suppression is enabled, and that it should be run
                    # for the current ``asl_context```
                    if do_bs and (
                        asl_context
                        in asl_params[BACKGROUND_SUPPRESSION].get(
                            BS_APPLY_TO_ASL_CONTEXT
                        )
                    ):
                        # map all inputs except for m0
                        acquire_mri_image_filter.add_parent_filter(
                            parent=ground_truth_filter,
                            io_map={
                                key: key
                                for key in ground_truth_filter.outputs.keys()
                                if key != "m0"
                            },
                        )
                        # get m0 from bs
                        acquire_mri_image_filter.add_parent_filter(
                            parent=bs_filter,
                            io_map={
                                BackgroundSuppressionFilter.KEY_MAG_Z: AcquireMriImageFilter.KEY_M0,
                            },
                        )
                    else:
                        # map all inputs from the ground truth
                        acquire_mri_image_filter.add_parent_filter(
                            parent=ground_truth_filter
                        )

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
                        AcquireMriImageFilter.KEY_ECHO_TIME: asl_params["echo_time"][
                            asl_context_index
                        ],
                        AcquireMriImageFilter.KEY_REPETITION_TIME: asl_params[
                            "repetition_time"
                        ][asl_context_index],
                        AcquireMriImageFilter.KEY_ROTATION: (
                            asl_params["rot_x"][asl_context_index],
                            asl_params["rot_y"][asl_context_index],
                            asl_params["rot_z"][asl_context_index],
                        ),
                        AcquireMriImageFilter.KEY_TRANSLATION: (
                            asl_params["transl_x"][asl_context_index],
                            asl_params["transl_y"][asl_context_index],
                            asl_params["transl_z"][asl_context_index],
                        ),
                        AcquireMriImageFilter.KEY_TARGET_SHAPE: tuple(
                            asl_params["acq_matrix"]
                        ),
                        AcquireMriImageFilter.KEY_INTERPOLATION: asl_params[
                            "interpolation"
                        ],
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
                    append_metadata_filter = AppendMetadataFilter()

                    if asl_params["output_image_type"] == "magnitude":
                        # if 'output_image_type' is 'magnitude' use the phase_magnitude filter
                        phase_magnitude_filter = PhaseMagnitudeFilter()
                        phase_magnitude_filter.add_parent_filter(
                            acquire_mri_image_filter
                        )
                        append_metadata_filter.add_parent_filter(
                            phase_magnitude_filter, io_map={"magnitude": "image"}
                        )
                    else:
                        # otherwise just pass on the complex data
                        append_metadata_filter.add_parent_filter(
                            acquire_mri_image_filter
                        )

                    append_metadata_filter.add_input(
                        AppendMetadataFilter.KEY_METADATA,
                        {
                            "series_description": image_series["series_description"],
                            "series_type": image_series["series_type"],
                            "series_number": series_number,
                            "asl_context": asl_context,
                            "multiphase_index": multiphase_index,
                        },
                    )

                    # Add the acqusition pipeline to the combine time series filter
                    combine_time_series_filter.add_parent_filter(
                        parent=append_metadata_filter,
                        io_map={"image": f"image_{vol_index}"},
                    )
                    # increment the volume index
                    vol_index += 1

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
                    AcquireMriImageFilter.KEY_INTERPOLATION: struct_params[
                        "interpolation"
                    ],
                },
            }

            # map inputs from struct_params. acq_contrast, excitation_flip_angle, desired_snr,
            # inversion_time, inversion_flip_angle (last 2 are optional)
            acquire_mri_image_filter.add_inputs(
                struct_params, io_map_optional=True,
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
            interpolation_list = ground_truth_params.pop("interpolation")
            for quantity in ground_truth_image_keys:
                resample_filter = TransformResampleImageFilter()
                # map the ground_truth_filter to the resample filter
                resample_filter.add_parent_filter(
                    ground_truth_filter, io_map={quantity: "image"}
                )
                # there are two interpolation parameters in an array, the first is for all
                # quantities except for "seg_label", the second is for "seg_label". This is
                # because "seg_label" is a mask nearest neighbour interpolation is usually
                # more appropriate.
                interp_idx = 0
                if quantity == "seg_label":
                    interp_idx = 1

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
                        AcquireMriImageFilter.KEY_INTERPOLATION: interpolation_list[
                            interp_idx
                        ],
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
        for asl_context_index, image_to_output in enumerate(output_image_list):
            bids_output_filter = BidsOutputFilter()
            bids_output_filter.add_input(
                BidsOutputFilter.KEY_OUTPUT_DIRECTORY, temp_dir
            )
            bids_output_filter.add_input(BidsOutputFilter.KEY_IMAGE, image_to_output)
            bids_output_filter.add_input(
                BidsOutputFilter.KEY_SUBJECT_LABEL, subject_label
            )
            # run the filter to write the BIDS files to disk
            bids_output_filter.run()

        if output_filename is not None:
            filename, file_extension = splitext(output_filename)
            # output the file archive
            logger.info("Creating output archive: %s", output_filename)
            shutil.make_archive(
                filename, EXTENSION_MAPPING[file_extension], root_dir=temp_dir
            )

    return {"hrgt": ground_truth_filter.outputs, "asldro_output": output_image_list}


if __name__ == "__main__":
    run_full_pipeline()
