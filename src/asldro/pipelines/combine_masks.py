"""Pipeline to combine fuzzy masks into a single segmentation mask"""

import logging
import os

import json
import nibabel as nib

from asldro.containers.image import BaseImageContainer
from asldro.filters.json_loader import JsonLoaderFilter
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.filters.combine_fuzzy_masks_filter import CombineFuzzyMasksFilter
from asldro.validators.schemas.index import SCHEMAS

logger = logging.getLogger(__name__)


def combine_fuzzy_masks(
    params_filename: str, output_filename: str
) -> BaseImageContainer:
    """Combines fuzzy masks into a single segmentation mask image.

    :param params_filename: Path to the combining masks parameter JSON file.
    :type params_filename: str
    :param output_filename: Path to the output combined mask NIFTI image
    :type output_filename: str
    :return: The combined mask, as an image container.
    :rtype: BaseImageContainer
    """
    # load in the params JSON file and validate against the schema
    json_filter = JsonLoaderFilter()
    json_filter.add_input("filename", params_filename)
    json_filter.add_input("schema", SCHEMAS["combine_masks"])
    json_filter.run()

    # load in the masks, put them in a list
    mask_files = []
    for nifti_filename in json_filter.outputs["mask_files"]:
        nifti_loader = NiftiLoaderFilter()
        nifti_loader.add_input("filename", nifti_filename)
        nifti_loader.run()
        mask_files.append(nifti_loader.outputs["image"])

    combine_masks_filter = CombineFuzzyMasksFilter()

    combine_masks_filter.add_inputs(json_filter.outputs)
    combine_masks_filter.add_input(CombineFuzzyMasksFilter.KEY_FUZZY_MASK, mask_files)
    combine_masks_filter.run()

    # save the file
    nib.save(
        combine_masks_filter.outputs[CombineFuzzyMasksFilter.KEY_SEG_MASK].nifti_image,
        output_filename,
    )

    return combine_masks_filter.outputs[CombineFuzzyMasksFilter.KEY_SEG_MASK]

