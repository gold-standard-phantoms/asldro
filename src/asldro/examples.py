""" Examples of filter chains """
import pprint
import os
import logging
import shutil
import pdb

from typing import List
from tempfile import TemporaryDirectory

import numpy as np
import nibabel as nib
from asldro.containers.image import BaseImageContainer
from asldro.filters.ground_truth_loader import GroundTruthLoaderFilter
from asldro.filters.json_loader import JsonLoaderFilter
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.filters.gkm_filter import GkmFilter
from asldro.filters.invert_image_filter import InvertImageFilter
from asldro.filters.transform_resample_image_filter import TransformResampleImageFilter
from asldro.filters.add_complex_noise_filter import AddComplexNoiseFilter
from asldro.filters.acquire_mri_image_filter import AcquireMriImageFilter
from asldro.data.filepaths import (
    HRGT_ICBM_2009A_NLS_V3_JSON,
    HRGT_ICBM_2009A_NLS_V3_NIFTI,
    HRGT_ICBM_2009A_NLS_V4_JSON,
    HRGT_ICBM_2009A_NLS_V4_NIFTI,
)
from asldro.validators.user_parameter_input import (
    IMAGE_TYPE_VALIDATOR,
    ASL,
    validate_input_params,
    get_example_input_params,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".zip", ".tar.gz"]
# Used in shutil.make_archive
EXTENSION_MAPPING = {".zip": "zip", ".tar.gz": "gztar"}


def splitext(path):
    """
    The normal os.path.splitext treats path/example.tar.gz
    as having a filepath of path/example.tar with a .gz
    extension - this fixes it """
    for ext in [".tar.gz", ".tar.bz2"]:
        if path.lower().endswith(ext.lower()):
            return path[: -len(ext)], path[-len(ext) :]
    return os.path.splitext(path)


def run_full_pipeline(input_params: dict = None, output_filename: str = None):
    """ A function that runs the entire DRO pipeline. This
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

    # Load in the ground truth
    if input_params["global_configuration"]["ground_truth"] == "hrgt_icbm_2009a_nls_v3":
        ground_truth_nifti = HRGT_ICBM_2009A_NLS_V3_NIFTI
        ground_truth_json = HRGT_ICBM_2009A_NLS_V3_JSON
    elif (
        input_params["global_configuration"]["ground_truth"] == "hrgt_icbm_2009a_nls_v4"
    ):
        ground_truth_nifti = HRGT_ICBM_2009A_NLS_V4_NIFTI
        ground_truth_json = HRGT_ICBM_2009A_NLS_V4_JSON
    else:
        raise ValueError(
            "Global configuration "
            f"{input_params['global_configuration']['ground_truth']} "
            "does not correspond with accepted ground truths"
        )

    json_filter = JsonLoaderFilter()
    json_filter.add_input("filename", ground_truth_json)
    nifti_filter = NiftiLoaderFilter()
    nifti_filter.add_input("filename", ground_truth_nifti)
    ground_truth_filter = GroundTruthLoaderFilter()
    ground_truth_filter.add_parent_filter(nifti_filter)
    ground_truth_filter.add_parent_filter(json_filter)
    ground_truth_filter.run()

    logger.info("JsonLoaderFilter outputs:\n%s", pprint.pformat(json_filter.outputs))
    logger.debug("NiftiLoaderFilter outputs:\n%s", pprint.pformat(nifti_filter.outputs))
    logger.debug(
        "GroundTruthLoaderFilter outputs:\n%s",
        pprint.pformat(ground_truth_filter.outputs),
    )

    # create output lists to be populated in the "image_series" loop
    nifti_filename = []
    output_nifti = []
    # Loop over "image_series" in input_params
    # Take the asl image series and pass it to the remainder of the pipeline
    # update the input_params variable so it contains the asl series parameters
    for series_index, image_series in enumerate(input_params["image_series"]):
        series_number = series_index + 1

        ############################################################################################
        # ASL pipeline
        # Comprises GKM, then MRI signal model, transform and resampling, and noise for each dynamic.
        # After the 'acquisition loop' the dynamics are concatenated into a single 4D file
        if image_series["series_type"] == "asl":
            asl_params = image_series["series_parameters"]
            # initialise the random number generator for the image series
            np.random.seed(image_series["series_parameters"]["random_seed"])
            logger.info(
                "Running DRO generation with the following parameters:\n%s",
                pprint.pformat(asl_params),
            )

            # Generate the output filename
            # If asl_context only consists of M0 then the filename should be M0, otherwise ASL4D
            if asl_params["asl_context"].lower() == "m0":
                nifti_filename.append(f"{series_number}_M0.nii.gz")
            else:
                nifti_filename.append(f"{series_number}_ASL4D.nii.gz")

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
            gkm_filter.add_inputs(asl_params,)
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
            acquired_images_list: List[nib.Nifti2Image] = []
            for idx, asl_context in enumerate(asl_params["asl_context"].split()):
                acquire_mri_image_filter = AcquireMriImageFilter()
                # map inputs from the ground truth: t1, t2, t2_star, m0 all share the same name
                # so no explicit mapping is necessary.
                acquire_mri_image_filter.add_parent_filter(parent=ground_truth_filter,)

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
                # Run the acquire_mri_image_filter to generate an acquired volume
                acquire_mri_image_filter.run()
                image = acquire_mri_image_filter.outputs["image"]
                # Append list of the output images
                acquired_images_list.append(image.nifti_image)

            # Create a 4D ASL image with this timeseries
            # concatenate along the time axis (4th)
            # acquired_timeseries = nil.image.concat_imgs(acquired_images_list)
            image_shape = acquired_images_list[0].dataobj.shape
            acquired_timeseries_dataobj = np.ndarray(
                (
                    image_shape[0],
                    image_shape[1],
                    image_shape[2],
                    len(acquired_images_list),
                )
            )
            for idx, im in enumerate(acquired_images_list):
                acquired_timeseries_dataobj[:, :, :, idx]: np.ndarray = np.absolute(
                    np.asanyarray(im.dataobj)
                )
            # do not use the header during construction
            acquired_timeseries = type(acquired_images_list[0])(
                dataobj=acquired_timeseries_dataobj,
                affine=acquired_images_list[0].affine,
            )
            acquired_timeseries.update_header()
            acquired_timeseries.header["descrip"] = image_series["series_description"]
            # place in output_nifti list
            output_nifti.append(acquired_timeseries)
            # logging
            logger.debug("GkmFilter outputs: \n %s", pprint.pformat(gkm_filter.outputs))
            logger.debug(
                "acquire_mri_image_filter outputs: \n %s",
                pprint.pformat(acquire_mri_image_filter.outputs),
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

            # Generate the output subdirectory name and
            nifti_filename.append(f"{series_number}_structural.nii.gz")

            # Simulate acquisition
            acquire_mri_image_filter = AcquireMriImageFilter()
            # map inputs from the ground truth: t1, t2, t2_star, m0 all share the same name
            # so no explicit mapping is necessary.
            acquire_mri_image_filter.add_parent_filter(parent=ground_truth_filter,)

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
                struct_params, io_map_optional=True,
            )
            # Run the acquire_mri_image_filter to generate an acquired volume
            acquire_mri_image_filter.run()

            struct_image_container = acquire_mri_image_filter.outputs[
                AcquireMriImageFilter.KEY_IMAGE
            ]

            if struct_params["output_image_type"] == "magnitude":
                struct_image_container.image = np.absolute(struct_image_container.image)

            struct_image_container.header["descrip"] = image_series[
                "series_description"
            ]
            # Append list of the output images
            output_nifti.append(struct_image_container.nifti_image)

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
            ground_truth_niftis = []
            ground_truth_filenames = []
            for quantity in ground_truth_image_keys:
                ground_truth_filenames.append(
                    f"{series_number}_ground_truth_{quantity}.nii.gz"
                )

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
                resample_filter.run()
                # Append list of the output images
                ground_truth_niftis.append(
                    resample_filter.outputs[
                        TransformResampleImageFilter.KEY_IMAGE
                    ].nifti_image
                )
            output_nifti.append(ground_truth_niftis)
            # append the output filenames
            nifti_filename.append(ground_truth_filenames)

    # Output everything to a temporary directory
    with TemporaryDirectory() as temp_dir:
        for idx, nifti in enumerate(output_nifti):
            if isinstance(nifti, list):
                for n, im in enumerate(nifti):
                    nib.save(
                        im, os.path.join(temp_dir, nifti_filename[idx][n]),
                    )
            else:
                nib.save(
                    nifti, os.path.join(temp_dir, nifti_filename[idx]),
                )

        if output_filename is not None:
            filename, file_extension = splitext(output_filename)
            # output the file archive
            logger.info("Creating output archive: %s", output_filename)
            shutil.make_archive(
                filename, EXTENSION_MAPPING[file_extension], root_dir=temp_dir
            )


if __name__ == "__main__":
    run_full_pipeline()
