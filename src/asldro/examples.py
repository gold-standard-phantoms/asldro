""" Examples of filter chains """
import pprint
import os
import logging
import shutil
from copy import deepcopy
from tempfile import TemporaryDirectory

import numpy as np
import nibabel as nib

from asldro.filters.ground_truth_loader import GroundTruthLoaderFilter
from asldro.filters.json_loader import JsonLoaderFilter
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.containers.image import NumpyImageContainer, INVERSE_DOMAIN
from asldro.filters.gkm_filter import GkmFilter
from asldro.filters.mri_signal_filter import MriSignalFilter
from asldro.filters.invert_image_filter import InvertImageFilter
from asldro.data.filepaths import (
    HRGT_ICBM_2009A_NLS_V3_JSON,
    HRGT_ICBM_2009A_NLS_V3_NIFTI,
)
from asldro.validators.user_parameter_input import USER_INPUT_VALIDATOR

logger = logging.getLogger(__name__)

EXAMPLE_INPUT_PARAMS = {
    "asl_context_array": "m0scan m0scan control label",
    "label_type": "pCASL",
    "lambda_blood_brain": 0.9,  # TODO: move into the ground truth loader
    "t1_arterial_blood": 1.65,  # TODO: move into the ground truth loader
}

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
        input_params = deepcopy(EXAMPLE_INPUT_PARAMS)
    if output_filename is not None:
        _, output_filename_extension = splitext(output_filename)
        if output_filename_extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"File output type {output_filename_extension} not supported"
            )

    # Validate parameter and update defaults
    input_params = USER_INPUT_VALIDATOR.validate(input_params)

    json_filter = JsonLoaderFilter()
    json_filter.add_input("filename", HRGT_ICBM_2009A_NLS_V3_JSON)

    nifti_filter = NiftiLoaderFilter()
    nifti_filter.add_input("filename", HRGT_ICBM_2009A_NLS_V3_NIFTI)

    ground_truth_filter = GroundTruthLoaderFilter()
    ground_truth_filter.add_parent_filter(nifti_filter)
    ground_truth_filter.add_parent_filter(json_filter)

    ground_truth_filter.run()

    logger.info(f"JsonLoaderFilter outputs:\n{pprint.pformat(json_filter.outputs)}")
    logger.info(f"NiftiLoaderFilter outputs:\n{pprint.pformat(nifti_filter.outputs)}")
    logger.info(
        f"GroundTruthLoaderFilter outputs:\n{pprint.pformat(ground_truth_filter.outputs)}"
    )

    # Create an image container in the INVERSE_DOMAIN
    image_container = NumpyImageContainer(
        image=np.zeros((3, 3, 3)), data_domain=INVERSE_DOMAIN
    )
    logger.info(f"NumpyImageContainer:\n{pprint.pformat(image_container)}")

    # Run the GkmFilter on the ground_truth data
    gkm_filter = GkmFilter()
    gkm_filter.add_parent_filter(
        parent=ground_truth_filter,
        io_map={
            "perfusion_rate": gkm_filter.KEY_PERFUSION_RATE,
            "transit_time": gkm_filter.KEY_TRANSIT_TIME,
            "m0": gkm_filter.KEY_M0,
            "t1": gkm_filter.KEY_T1_TISSUE,
        },
    )
    gkm_filter.add_input(gkm_filter.KEY_LABEL_TYPE, gkm_filter.PCASL)
    gkm_filter.add_input(gkm_filter.KEY_SIGNAL_TIME, input_params["signal_time"])
    gkm_filter.add_input(gkm_filter.KEY_LABEL_DURATION, input_params["label_duration"])
    gkm_filter.add_input(
        gkm_filter.KEY_LABEL_EFFICIENCY, input_params["label_efficiency"]
    )
    gkm_filter.add_input(
        gkm_filter.KEY_LAMBDA_BLOOD_BRAIN, input_params["lambda_blood_brain"]
    )
    gkm_filter.add_input(
        gkm_filter.KEY_T1_ARTERIAL_BLOOD, input_params["t1_arterial_blood"]
    )

    # Run the MriSignalFilter to obtain control, label and m0scan
    # control: gradient echo, TE=10ms, TR = 5000ms
    control_filter = MriSignalFilter()
    control_filter.add_parent_filter(
        parent=ground_truth_filter,
        io_map={
            "t1": control_filter.KEY_T1,
            "t2": control_filter.KEY_T2,
            "t2_star": control_filter.KEY_T2_STAR,
            "m0": control_filter.KEY_M0,
        },
    )
    control_filter.add_input(control_filter.KEY_ACQ_CONTRAST, "ge")
    control_filter.add_input(control_filter.KEY_ACQ_TE, 10e-3)
    control_filter.add_input(control_filter.KEY_ACQ_TR, 5.0)

    # label: gradient echo, TE=10ms, TR = 5000ms
    # reverse the polarity of delta_m.image for encoding it into the label signal
    invert_delta_m_filter = InvertImageFilter()
    invert_delta_m_filter.add_parent_filter(
        parent=gkm_filter, io_map={gkm_filter.KEY_DELTA_M: "image"}
    )

    label_filter = MriSignalFilter()
    label_filter.add_parent_filter(
        parent=ground_truth_filter,
        io_map={
            "t1": label_filter.KEY_T1,
            "t2": label_filter.KEY_T2,
            "t2_star": label_filter.KEY_T2_STAR,
            "m0": label_filter.KEY_M0,
        },
    )
    label_filter.add_parent_filter(
        parent=invert_delta_m_filter, io_map={"image": label_filter.KEY_MAG_ENC}
    )
    label_filter.add_input(label_filter.KEY_ACQ_CONTRAST, "ge")
    label_filter.add_input(label_filter.KEY_ACQ_TE, 10e-3)
    label_filter.add_input(label_filter.KEY_ACQ_TR, 5.0)

    # m0scan: gradient echo, TE=10ms, TR=10000ms
    m0scan_filter = MriSignalFilter()
    m0scan_filter.add_parent_filter(
        parent=ground_truth_filter,
        io_map={
            "t1": m0scan_filter.KEY_T1,
            "t2": m0scan_filter.KEY_T2,
            "t2_star": m0scan_filter.KEY_T2_STAR,
            "m0": m0scan_filter.KEY_M0,
        },
    )
    m0scan_filter.add_input(m0scan_filter.KEY_ACQ_CONTRAST, "ge")
    m0scan_filter.add_input(m0scan_filter.KEY_ACQ_TE, 10e-3)
    m0scan_filter.add_input(m0scan_filter.KEY_ACQ_TR, 10.0)

    # logging
    logger.info(f"GkmFilter outputs: \n {pprint.pformat(gkm_filter.outputs)}")
    logger.info(f"control_filter outputs: \n {pprint.pformat(control_filter.outputs)}")
    logger.info(f"label_filter outputs: \n {pprint.pformat(label_filter.outputs)}")
    logger.info(f"m0scan_filter outputs: \n {pprint.pformat(m0scan_filter.outputs)}")

    control_filter.run()
    label_filter.run()
    m0scan_filter.run()
    control_label_difference = (
        control_filter.outputs[control_filter.KEY_IMAGE].image
        - label_filter.outputs[label_filter.KEY_IMAGE].image
    )
    delta_m_array: np.ndarray = gkm_filter.outputs[gkm_filter.KEY_DELTA_M].image

    # Compare control - label with delta m from the GkmFilter.  Note that delta m must be multiplied
    # by exp(-TE/T2*) because control - label is transverse mangetisation and subject to T2* decay
    # and the output of the GkmFilter is longitudinal magnetisation
    t2_star_array: np.ndarray = ground_truth_filter.outputs["t2_star"].image
    comparison = np.allclose(
        control_label_difference,
        delta_m_array
        * np.exp(
            -np.divide(
                control_filter.inputs[control_filter.KEY_ACQ_TE],
                t2_star_array,
                out=np.zeros_like(t2_star_array),
                where=t2_star_array != 0,
            )
        ),
    )

    logger.info(f"control - label == delta_m? {comparison}")
    logger.info(
        f"residual = {np.sqrt(np.mean((control_label_difference - delta_m_array)**2))}"
    )

    # Output everything to a temporary directory
    with TemporaryDirectory() as temp_dir:

        nib.save(
            control_filter.outputs[control_filter.KEY_IMAGE]._nifti_image,
            os.path.join(temp_dir, "control.nii.gz"),
        )
        nib.save(
            label_filter.outputs[label_filter.KEY_IMAGE]._nifti_image,
            os.path.join(temp_dir, "label.nii.gz"),
        )
        nib.save(
            m0scan_filter.outputs[m0scan_filter.KEY_IMAGE]._nifti_image,
            os.path.join(temp_dir, "m0scan.nii.gz"),
        )
        nib.save(
            ground_truth_filter.outputs["m0"]._nifti_image,
            os.path.join(temp_dir, "m0scan_ground_truth.nii.gz"),
        )
        nib.save(
            gkm_filter.outputs[gkm_filter.KEY_DELTA_M]._nifti_image,
            os.path.join(temp_dir, "delta_m.nii.gz"),
        )
        if output_filename is not None:
            filename, file_extension = splitext(output_filename)
            # output the file archive
            shutil.make_archive(
                filename, EXTENSION_MAPPING[file_extension], root_dir=temp_dir
            )


if __name__ == "__main__":
    run_full_pipeline()
