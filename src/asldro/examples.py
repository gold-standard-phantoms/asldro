""" Examples of filter chains """
import pprint
import os
import numpy as np
import nibabel as nib

from asldro.filters.ground_truth_loader import GroundTruthLoaderFilter
from asldro.filters.json_loader import JsonLoaderFilter
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.containers.image import NumpyImageContainer, INVERSE_DOMAIN
from asldro.filters.gkm_filter import (
    GkmFilter,
    PCASL,
    KEY_PERFUSION_RATE,
    KEY_TRANSIT_TIME,
    KEY_M0,
    KEY_LABEL_TYPE,
    KEY_LABEL_DURATION,
    KEY_SIGNAL_TIME,
    KEY_LABEL_EFFICIENCY,
    KEY_LAMBDA_BLOOD_BRAIN,
    KEY_T1_ARTERIAL_BLOOD,
    KEY_T1_TISSUE,
    KEY_DELTA_M,
)
from asldro.filters.mri_signal_filter import (
    MriSignalFilter,
    KEY_T1,
    KEY_T2,
    KEY_T2_STAR,
    KEY_MAG_ENC,
    KEY_ACQ_CONTRAST,
    KEY_ACQ_TE,
    KEY_ACQ_TR,
    KEY_IMAGE,
)
from asldro.data.filepaths import (
    HRGT_ICBM_2009A_NLS_V3_JSON,
    HRGT_ICBM_2009A_NLS_V3_NIFTI,
)


def run_full_pipeline():
    """ A function that runs the entire DRO pipeline. This
    can be extended as more functionality is included.
    This function is deliberately verbose to explain the
    operation, inputs and outputs of individual filters. """

    json_filter = JsonLoaderFilter()
    json_filter.add_input("filename", HRGT_ICBM_2009A_NLS_V3_JSON)

    nifti_filter = NiftiLoaderFilter()
    nifti_filter.add_input("filename", HRGT_ICBM_2009A_NLS_V3_NIFTI)

    ground_truth_filter = GroundTruthLoaderFilter()
    ground_truth_filter.add_parent_filter(nifti_filter)
    ground_truth_filter.add_parent_filter(json_filter)

    ground_truth_filter.run()

    print(f"JsonLoaderFilter outputs:\n{pprint.pformat(json_filter.outputs)}")
    print(f"NiftiLoaderFilter outputs:\n{pprint.pformat(nifti_filter.outputs)}")
    print(
        f"GroundTruthLoaderFilter outputs:\n{pprint.pformat(ground_truth_filter.outputs)}"
    )

    # Create an image container in the INVERSE_DOMAIN
    image_container = NumpyImageContainer(
        image=np.zeros((3, 3, 3)), data_domain=INVERSE_DOMAIN
    )
    print(f"NumpyImageContainer:\n{pprint.pformat(image_container)}")

    # Run the GkmFilter on the ground_truth data
    label_duration = 1.8
    post_label_delay = 1.8
    signal_time = label_duration + post_label_delay
    gkm_filter = GkmFilter()
    gkm_filter.add_input(
        KEY_PERFUSION_RATE, ground_truth_filter.outputs["perfusion_rate"]
    )
    gkm_filter.add_input(KEY_TRANSIT_TIME, ground_truth_filter.outputs["transit_time"])
    gkm_filter.add_input(KEY_M0, ground_truth_filter.outputs["m0"])
    gkm_filter.add_input(KEY_T1_TISSUE, ground_truth_filter.outputs["t1"])
    gkm_filter.add_input(KEY_LABEL_TYPE, PCASL)
    gkm_filter.add_input(KEY_SIGNAL_TIME, signal_time)
    gkm_filter.add_input(KEY_LABEL_DURATION, label_duration)
    gkm_filter.add_input(KEY_LABEL_EFFICIENCY, 0.85)
    gkm_filter.add_input(KEY_LAMBDA_BLOOD_BRAIN, 0.9)
    gkm_filter.add_input(KEY_T1_ARTERIAL_BLOOD, 1.65)
    gkm_filter.run()

    print(f"GkmFilter outputs: \n {pprint.pformat(gkm_filter.outputs)}")

    # Run the MriSignalFilter to obtain control, label and m0scan
    # control: gradient echo, TE=10ms, TR = 5000ms
    control_filter = MriSignalFilter()
    control_filter.add_input(KEY_T1, ground_truth_filter.outputs["t1"])
    control_filter.add_input(KEY_T2, ground_truth_filter.outputs["t2"])
    control_filter.add_input(KEY_T2_STAR, ground_truth_filter.outputs["t2_star"])
    control_filter.add_input(KEY_M0, ground_truth_filter.outputs["m0"])
    control_filter.add_input(KEY_ACQ_CONTRAST, "ge")
    control_filter.add_input(KEY_ACQ_TE, 10e-3)
    control_filter.add_input(KEY_ACQ_TR, 5.0)
    control_filter.run()
    print(f"control_filter outputs: \n {pprint.pformat(control_filter.outputs)}")

    # label: gradient echo, TE=10ms, TR = 5000ms
    delta_m = gkm_filter.outputs[KEY_DELTA_M].clone()
    # reverse the polarity of delta_m.image for encoding it into the label signal
    t2_star = ground_truth_filter.outputs["t2_star"].image
    delta_m.image = -delta_m.image
    label_filter = MriSignalFilter()
    label_filter.add_input(KEY_T1, ground_truth_filter.outputs["t1"])
    label_filter.add_input(KEY_T2, ground_truth_filter.outputs["t2"])
    label_filter.add_input(KEY_T2_STAR, ground_truth_filter.outputs["t2_star"])
    label_filter.add_input(KEY_M0, ground_truth_filter.outputs["m0"])
    label_filter.add_input(KEY_MAG_ENC, delta_m)
    label_filter.add_input(KEY_ACQ_CONTRAST, "ge")
    label_filter.add_input(KEY_ACQ_TE, 10e-3)
    label_filter.add_input(KEY_ACQ_TR, 5.0)
    label_filter.run()
    print(f"label_filter outputs: \n {pprint.pformat(label_filter.outputs)}")

    # m0scan: gradient echo, TE=10ms, TR=10000ms
    m0scan_filter = MriSignalFilter()
    m0scan_filter.add_input(KEY_T1, ground_truth_filter.outputs["t1"])
    m0scan_filter.add_input(KEY_T2, ground_truth_filter.outputs["t2"])
    m0scan_filter.add_input(KEY_T2_STAR, ground_truth_filter.outputs["t2_star"])
    m0scan_filter.add_input(KEY_M0, ground_truth_filter.outputs["m0"])
    m0scan_filter.add_input(KEY_ACQ_CONTRAST, "ge")
    m0scan_filter.add_input(KEY_ACQ_TE, 10e-3)
    m0scan_filter.add_input(KEY_ACQ_TR, 10.0)
    m0scan_filter.run()
    print(f"m0scan_filter outputs: \n {pprint.pformat(m0scan_filter.outputs)}")

    control_label_difference = (
        control_filter.outputs[KEY_IMAGE].image - label_filter.outputs[KEY_IMAGE].image
    )
    delta_m_array: np.ndarray = gkm_filter.outputs[KEY_DELTA_M].image

    comparison = np.allclose(control_label_difference, delta_m_array,)

    print(f"control - label == delta_m? {comparison}")
    print(
        f"residual = {np.sqrt(np.mean((control_label_difference - delta_m_array)**2))}"
    )
    if not os.path.exists("output"):
        os.mkdir("output")

    nib.save(control_filter.outputs[KEY_IMAGE]._nifti_image, "output/control.nii.gz")
    nib.save(label_filter.outputs[KEY_IMAGE]._nifti_image, "output/label.nii.gz")
    nib.save(m0scan_filter.outputs[KEY_IMAGE]._nifti_image, "output/m0scan.nii.gz")
    nib.save(
        ground_truth_filter.outputs["m0"]._nifti_image,
        "output/m0scan_ground_truth.nii.gz",
    )
    nib.save(gkm_filter.outputs[KEY_DELTA_M]._nifti_image, "output/delta_m.nii.gz")


if __name__ == "__main__":
    run_full_pipeline()
