""" Examples of filter chains """
import pprint
import os
import logging
import shutil
import pdb
from copy import deepcopy
from typing import List
from tempfile import TemporaryDirectory

import numpy as np
import nibabel as nib
import nilearn as nil

from asldro.filters.ground_truth_loader import GroundTruthLoaderFilter
from asldro.filters.json_loader import JsonLoaderFilter
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.containers.image import NumpyImageContainer, INVERSE_DOMAIN
from asldro.filters.gkm_filter import GkmFilter
from asldro.filters.mri_signal_filter import MriSignalFilter
from asldro.filters.invert_image_filter import InvertImageFilter
from asldro.filters.transform_resample_image_filter import TransformResampleImageFilter
from asldro.filters.add_complex_noise_filter import AddComplexNoiseFilter
from asldro.data.filepaths import (
    HRGT_ICBM_2009A_NLS_V3_JSON,
    HRGT_ICBM_2009A_NLS_V3_NIFTI,
)
from asldro.validators.user_parameter_input import USER_INPUT_VALIDATOR

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
    operation, inputs and outputs of individual filters. """

    if input_params is None:
        input_params = {}  # Empty dictionary - will get populated with defaults

    if output_filename is not None:
        _, output_filename_extension = splitext(output_filename)
        if output_filename_extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"File output type {output_filename_extension} not supported"
            )

    # Validate parameter and update defaults
    input_params = USER_INPUT_VALIDATOR.validate(input_params)

    logger.info(
        "Running DRO generation with the following parameters:\n%s",
        pprint.pformat(input_params),
    )
    json_filter = JsonLoaderFilter()
    json_filter.add_input("filename", HRGT_ICBM_2009A_NLS_V3_JSON)

    nifti_filter = NiftiLoaderFilter()
    nifti_filter.add_input("filename", HRGT_ICBM_2009A_NLS_V3_NIFTI)

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

    # Run the GkmFilter on the ground_truth data
    gkm_filter = GkmFilter()
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
    gkm_filter.add_input(gkm_filter.KEY_LABEL_TYPE, input_params["label_type"])
    gkm_filter.add_input(gkm_filter.KEY_SIGNAL_TIME, input_params["signal_time"])
    gkm_filter.add_input(gkm_filter.KEY_LABEL_DURATION, input_params["label_duration"])
    gkm_filter.add_input(
        gkm_filter.KEY_LABEL_EFFICIENCY, input_params["label_efficiency"]
    )

    # reverse the polarity of delta_m.image for encoding it into the label signal
    invert_delta_m_filter = InvertImageFilter()
    invert_delta_m_filter.add_parent_filter(
        parent=gkm_filter, io_map={gkm_filter.KEY_DELTA_M: "image"}
    )

    # Create one-time data that is required by the Acquisition Loop
    # 1. m0 resampled at the acquisition resolution
    m0_resample_filter = TransformResampleImageFilter()
    m0_resample_filter.add_parent_filter(
        ground_truth_filter, io_map={"m0": TransformResampleImageFilter.KEY_IMAGE}
    )
    m0_resample_filter.add_input(
        TransformResampleImageFilter.KEY_TARGET_SHAPE, tuple(input_params["acq_matrix"])
    )

    # 2. CBF resampled at the acquisition resolution
    cbf_resample_filter = TransformResampleImageFilter()
    cbf_resample_filter.add_parent_filter(
        ground_truth_filter,
        io_map={"perfusion_rate": TransformResampleImageFilter.KEY_IMAGE},
    )
    cbf_resample_filter.add_input(
        TransformResampleImageFilter.KEY_TARGET_SHAPE, tuple(input_params["acq_matrix"])
    )

    # 3. tissue label masks resampled at the acquisition resolution
    labelmask_resample_filter = TransformResampleImageFilter()
    labelmask_resample_filter.add_parent_filter(
        ground_truth_filter,
        io_map={"seg_label": TransformResampleImageFilter.KEY_IMAGE},
    )
    labelmask_resample_filter.add_input(
        TransformResampleImageFilter.KEY_TARGET_SHAPE, tuple(input_params["acq_matrix"])
    )

    # Acquisition Loop
    acquired_images_list: List[nib.Nifti2Image] = []
    for idx, asl_context in enumerate(input_params["asl_context"].split()):

        # Calculate MRI signal based on asl_context
        mri_signal_filter = MriSignalFilter()
        mri_signal_filter.add_parent_filter(
            parent=ground_truth_filter,
            io_map={
                "t1": MriSignalFilter.KEY_T1,
                "t2": MriSignalFilter.KEY_T2,
                "t2_star": MriSignalFilter.KEY_T2_STAR,
                "m0": MriSignalFilter.KEY_M0,
            },
        )
        mri_signal_filter.add_input(
            MriSignalFilter.KEY_ACQ_CONTRAST, input_params["acq_contrast"]
        )
        mri_signal_filter.add_input(
            MriSignalFilter.KEY_ACQ_TE, input_params["echo_time"][idx]
        )
        mri_signal_filter.add_input(
            MriSignalFilter.KEY_ACQ_TR, input_params["repetition_time"][idx]
        )

        # for ASL context == "label" use the inverted delta_m as
        # the input MriSignalFilter.KEY_MAG_ENC
        if asl_context.lower() == "label":
            mri_signal_filter.add_parent_filter(
                parent=invert_delta_m_filter,
                io_map={"image": MriSignalFilter.KEY_MAG_ENC},
            )

        # Transform and resample
        rotation = (
            input_params["rot_x"][idx],
            input_params["rot_y"][idx],
            input_params["rot_z"][idx],
        )
        # use default for rotation origin (0.0, 0.0, 0.0)
        translation = (
            input_params["transl_x"][idx],
            input_params["transl_y"][idx],
            input_params["transl_z"][idx],
        )
        motion_resample_filter = TransformResampleImageFilter()
        motion_resample_filter.add_parent_filter(mri_signal_filter)
        motion_resample_filter.add_input(
            TransformResampleImageFilter.KEY_ROTATION, rotation
        )
        motion_resample_filter.add_input(
            TransformResampleImageFilter.KEY_TRANSLATION, translation
        )
        motion_resample_filter.add_input(
            TransformResampleImageFilter.KEY_TARGET_SHAPE,
            tuple(input_params["acq_matrix"]),
        )

        # Add noise based on SNR
        add_complex_noise_filter = AddComplexNoiseFilter()
        add_complex_noise_filter.add_parent_filter(
            m0_resample_filter,
            io_map={
                TransformResampleImageFilter.KEY_IMAGE: AddComplexNoiseFilter.KEY_REF_IMAGE
            },
        )
        add_complex_noise_filter.add_parent_filter(
            motion_resample_filter,
            io_map={
                TransformResampleImageFilter.KEY_IMAGE: AddComplexNoiseFilter.KEY_IMAGE
            },
        )
        add_complex_noise_filter.add_input(
            AddComplexNoiseFilter.KEY_SNR, input_params["desired_snr"]
        )

        # Run the add_complex_noise_filter
        add_complex_noise_filter.run()

        # Append list of the output images

        acquired_images_list.append(
            add_complex_noise_filter.outputs[
                AddComplexNoiseFilter.KEY_IMAGE
            ]._nifti_image
        )

    # concatenate along the time axis (4th)
    # acquired_timeseries = nil.image.concat_imgs(acquired_images_list)
    image_shape = acquired_images_list[0].dataobj.shape
    acquired_timeseries_dataobj = np.ndarray(
        (image_shape[0], image_shape[1], image_shape[2], len(acquired_images_list))
    )
    for idx, im in enumerate(acquired_images_list):
        acquired_timeseries_dataobj[:, :, :, idx]: np.ndarray = np.absolute(
            np.asanyarray(im.dataobj)
        )

    # construct timeseries header
    # header: nib.Nifti2Header=acquired_images_list[0].header
    # header.set_data_shape(4)
    # header.set
    acquired_timeseries = type(acquired_images_list[0])(
        dataobj=acquired_timeseries_dataobj, affine=acquired_images_list[0].affine
    )
    acquired_timeseries.update_header()
    acquired_timeseries.header["descrip"] = "ASLDRO generated magnitude source data"

    cbf_resample_filter.run()
    labelmask_resample_filter.run()

    # logging
    logger.debug("GkmFilter outputs: \n %s", pprint.pformat(gkm_filter.outputs))
    logger.debug(
        "add_complex_noise_filter outputs: \n %s",
        pprint.pformat(add_complex_noise_filter.outputs),
    )
    logger.debug(
        "motion_resample_filter outputs: \n %s",
        pprint.pformat(motion_resample_filter.outputs),
    )
    logger.debug(
        "mri_signal_filter outputs: \n %s", pprint.pformat(mri_signal_filter.outputs)
    )

    # Output everything to a temporary directory
    with TemporaryDirectory() as temp_dir:

        nib.save(
            acquired_timeseries, os.path.join(temp_dir, "asl_source_magnitude.nii.gz")
        )
        nib.save(
            cbf_resample_filter.outputs[cbf_resample_filter.KEY_IMAGE]._nifti_image,
            os.path.join(temp_dir, "gt_cbf_acq_res.nii.gz"),
        )
        nib.save(
            labelmask_resample_filter.outputs[
                labelmask_resample_filter.KEY_IMAGE
            ]._nifti_image,
            os.path.join(temp_dir, "gt_labelmask_acq_res.nii.gz"),
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
