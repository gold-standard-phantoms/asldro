"""Pipeline to create a ground truth for the QASPER phantom"""

import logging
import os
import json
import argparse
from tempfile import TemporaryDirectory
import nibabel as nib
import numpy as np
import nilearn as nil

from asldro.pipelines.generate_ground_truth import generate_hrgt
from asldro.pipelines.combine_masks import combine_fuzzy_masks
from asldro.data.filepaths import QASPER_DATA
from asldro.utils.generate_numeric_function import (
    generate_circular_function_array,
    generate_gaussian_function,
)
from asldro.cli import DirType

logger = logging.getLogger(__name__)


def generate_qasper(output_dir: str = None) -> dict:
    """Generates a valid QASPER ground truth for ASLDRO. This function generates
    the QASPER ground truth that is included with ASLDRO. For more information see
    :ref:`qasper-3t-hrgt`

    :param output_dir: path to the output directory to create the hrgt.nii.gz and
      hrgt.json files, defaults to None
    :type output_dir: str, optional
    :return: dictionary with entries 'image', the 5D ground truth image, and
      'image_info', a dictionary describing its contents.
    :rtype: dict
    """
    # The masks in QASPER_DATA are fuzzy, i.e. their voxel values are between 0 and 1
    # and represent the fraction of the voxel occupied by that particular region.
    # These need to be combined to create a segmentation mask with integer values
    # for each region.
    inlet_label_value = 1
    porous_label_value = 2
    outlet_label_value = 3
    combine_mask_params = {
        "mask_files": [
            QASPER_DATA["inlet_fuzzy_mask"],
            QASPER_DATA["porous_fuzzy_mask"],
            QASPER_DATA["outlet_fuzzy_mask"],
        ],
        "region_values": [inlet_label_value, porous_label_value, outlet_label_value],
        "region_priority": [1, 2, 3],
        "threshold": 0.05,
    }

    with TemporaryDirectory() as temp_dir:
        json_filename = os.path.join(temp_dir, "params.json")
        seg_mask_filename = os.path.join(temp_dir, "qasper_seg_mask.nii.gz")
        with open(json_filename, "w") as json_file:
            json.dump(combine_mask_params, json_file, indent=4)

        seg_mask = combine_fuzzy_masks(json_filename, seg_mask_filename)

        ## Create the ground truth with quantity values applied to each region
        # calculate the normalised perfusion rate based on the volumes of each
        # region. This then calculates the perfusion rate based on an input flow rate of
        # Q = 1mL/min for f = 100 * Q / (N * V * rho)
        # V = voxel volume in mm
        # N = number of voxels in region
        # rho = mass density in region
        # calculate the number of voxels in each region
        voxel_volume_ml = np.prod(seg_mask.voxel_size_mm) / 1000.0
        number_inlet_voxels = np.sum(seg_mask.image == inlet_label_value)
        number_porous_voxels = np.sum(seg_mask.image == porous_label_value)
        number_outlet_voxels = np.sum(seg_mask.image == outlet_label_value)
        perfusate_lambda = 1.0
        porous_lambda = 0.32

        hrgt_params = {
            "label_values": [0] + combine_mask_params["region_values"],
            "label_names": ["background", "inlet", "porous", "outlet"],
            "quantities": {
                "perfusion_rate": [
                    0.0,
                    100 / (number_inlet_voxels * voxel_volume_ml * perfusate_lambda),
                    100 / (number_porous_voxels * voxel_volume_ml * porous_lambda),
                    100 / (number_outlet_voxels * voxel_volume_ml * perfusate_lambda),
                ],
                "transit_time": [0.0, 1.0, 1.0, 1.0],
                "t1": [0.0, 1.8, 1.8, 1.8],
                "t2": [0.0, 1.2, 0.2, 1.2],
                "t2_star": [0.0, 0.9, 0.1, 0.9],
                "m0": [0.0, 100.0, 32.0, 100.0],
                "lambda_blood_brain": [
                    0.0,
                    perfusate_lambda,
                    porous_lambda,
                    perfusate_lambda,
                ],
            },
            "units": ["ml/100g/min", "s", "s", "s", "s", "", "g/ml"],
            "parameters": {"t1_arterial_blood": 1.80, "magnetic_field_strength": 3.0,},
        }

        hrgt_json_filename = os.path.join(temp_dir, "hrgt_params.json")
        with open(hrgt_json_filename, "w") as json_file:
            json.dump(hrgt_params, json_file, indent=4)

        qasper_hrgt = generate_hrgt(hrgt_json_filename, seg_mask_filename)

    ## Calculate spatially varying transit time maps
    arteriole_end_tt = 0.25  # transit time at the end of the arteriole
    porous_max_tt = 10.0  # max transit time in the porous
    outlet_max_tt = 20  # max transit time at the outlet

    # Use the seg_mask's image affine to create meshgrids in world space.
    i = np.arange(seg_mask.shape[0])
    j = np.arange(seg_mask.shape[1])
    k = np.arange(seg_mask.shape[2])
    ii, jj, kk = np.meshgrid(i, j, k, sparse=False)

    xx, yy, zz = nil.image.coord_transform(ii, jj, kk, seg_mask.affine)

    # the origin in world space of the QASPER model is at the start of the first porous layer,
    # in the middle of the disc. Therefore the middle of the the first 'arteriole' tube
    # is at (x, y, z) = (0, 45.5, 4.75)mm, the circular array origin is at (0, 0, 4.75)

    array_origin = [0.0, 0.0, 4.75]
    func_params = {"loc": [0.0, 45.5, 4.75], "theta": 0, "fwhm": [20, 20, 25]}

    gaussian_array_map = generate_circular_function_array(
        func=generate_gaussian_function,
        xx=xx,
        yy=yy,
        zz=zz,
        array_origin=array_origin,
        array_size=60,
        array_angular_increment=360 / 60,
        func_params=func_params,
    )

    # scale and subtract so that the value at the end of the arteriole is 1.0, and this increases
    # to a value of 10
    transit_time_porous = (
        porous_max_tt * np.ones_like(gaussian_array_map)
        - (porous_max_tt - arteriole_end_tt) * gaussian_array_map
    )
    transit_time_map = np.zeros_like(transit_time_porous)
    porous_mask = seg_mask.image == porous_label_value
    transit_time_map[porous_mask] = transit_time_porous[porous_mask]

    # for the inlet, linearly scale the transit time along the z axis from a value of 0 at z=-20mm
    # to 1.0 at z=9.5mm
    transit_time_inlet = arteriole_end_tt * (zz + 20) / 29.5
    inlet_mask = seg_mask.image == inlet_label_value
    transit_time_map[inlet_mask] = transit_time_inlet[inlet_mask]

    # for the outlet, linearly scale the transit time along the z axis from a value of
    # porous_max_tt at z=0 to outlet_max_tt at 44.5
    outlet_mask = seg_mask.image == outlet_label_value
    transit_time_outlet = ((outlet_max_tt - porous_max_tt) / 44.5) * zz + porous_max_tt
    transit_time_map[outlet_mask] = transit_time_outlet[outlet_mask]

    if transit_time_map.ndim < 4:
        transit_time_map = np.expand_dims(np.atleast_3d(transit_time_map), axis=3)

    qasper_hrgt["image"].image[
        :, :, :, :, list(hrgt_params["quantities"]).index("transit_time")
    ] = transit_time_map

    # save
    if output_dir is not None:
        nib.save(
            qasper_hrgt["image"].nifti_image,
            os.path.join(output_dir, "qasper_hrgt.nii.gz"),
        )

        json_filename = os.path.join(output_dir, "qasper_hrgt.json")
        with open(json_filename, "w") as json_file:
            json.dump(qasper_hrgt["image_info"], json_file, indent=4)

    return qasper_hrgt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Generates a QASPER phantom hrgt for ASLDRO"""
    )

    parser.add_argument(
        "output_dir",
        type=DirType(should_exist=True),
        help="The directory to output to, creating a qasper_hrgt.nii.gz"
        "and qasper_hrgt.json files in this directory. "
        "Must exist. Will overwrite any existing files.",
    )
    args = parser.parse_args()
    generate_qasper(args.output_dir)

