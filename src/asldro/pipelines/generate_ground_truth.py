"""Pipeline to generate a ground truth image and save"""
import logging
import os

import json
import nibabel as nib

from asldro.filters.create_volumes_from_seg_mask import CreateVolumesFromSegMask
from asldro.filters.image_tools import FloatToIntImageFilter
from asldro.filters.json_loader import JsonLoaderFilter
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.filters.ground_truth_loader import GroundTruthLoaderFilter
from asldro.validators.schemas.index import SCHEMAS

logger = logging.getLogger(__name__)


def generate_hrgt(
    hrgt_params_filename: str, seg_mask_filename: str, output_dir: str = None
) -> dict:
    # pylint: disable=too-many-locals, too-many-statements
    """Generates a high-resolution ground truth (hrgt) based on:

        * A segmentation mask image
        * A file describing what values to assign to each region.

    The hrgt is saved in the folder ``output_dir``

    :param hrgt_params_filename: Path to the hrgt parameter JSON file
    :type hrgt_params_filename: str
    :param seg_mask_filename: Path to the segmentation mask NIFTI image
    :type seg_mask_filename: str
    :param output_dir: Directory to save files to, defaults to None
    :type output_dir: str, optional
    :return: dictionary containing the ground truth image, and the ground truth
      parameter file
    :rtype: dict
    """

    # load hrgt_params_filename and validate hrgt_params against the schema
    json_filter = JsonLoaderFilter()
    json_filter.add_input("filename", hrgt_params_filename)
    json_filter.add_input("schema", SCHEMAS["generate_hrgt_params"])

    # load seg_mask_filename
    nifti_filter = NiftiLoaderFilter()
    nifti_filter.add_input("filename", seg_mask_filename)

    # convert to an integer image, if the image is a float
    round_seg_mask_filter = FloatToIntImageFilter()
    round_seg_mask_filter.add_parent_filter(nifti_filter)  # use default method
    round_seg_mask_filter.add_input(
        FloatToIntImageFilter.KEY_METHOD, FloatToIntImageFilter.CEIL
    )

    create_volume_filter = CreateVolumesFromSegMask()
    create_volume_filter.add_parent_filter(
        round_seg_mask_filter, io_map={"image": "seg_mask"}
    )
    create_volume_filter.add_parent_filter(json_filter)

    create_volume_filter.run()

    create_volume_filter.outputs["image_info"]["parameters"] = json_filter.outputs[
        "parameters"
    ]

    ground_truth_filter = GroundTruthLoaderFilter()
    ground_truth_filter.add_inputs(create_volume_filter.outputs["image_info"])
    ground_truth_filter.add_input("image", create_volume_filter.outputs["image"])
    ground_truth_filter.run()  # check that no errors occur
    # save the files
    json_filename = os.path.join(output_dir, "hrgt.json")
    with open(json_filename, "w") as json_file:
        json.dump(create_volume_filter.outputs["image_info"], json_file, indent=4)

    nifti_filename = os.path.join(output_dir, "hrgt.nii.gz")
    nib.save(create_volume_filter.outputs["image"].nifti_image, nifti_filename)

    return create_volume_filter.outputs

