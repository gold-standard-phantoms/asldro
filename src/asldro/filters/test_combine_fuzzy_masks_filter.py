""" Tests for CombineFuzzyMasksFilter"""


import pytest

import numpy as np
from numpy.random import default_rng
import numpy.testing

import nibabel as nib

from asldro.filters.basefilter import FilterInputValidationError
from asldro.filters.combine_fuzzy_masks_filter import CombineFuzzyMasksFilter
from asldro.containers.image import NiftiImageContainer
from asldro.utils.filter_validation import validate_filter_inputs


@pytest.fixture(name="validation_data")
def input_validation_dict_fixture():
    """Returns a dictionary containing test data for input validation, and valid data
    for the CombineFuzzyMasksFilter"""

    varied_fuzzy_mask = np.array(
        (
            (1.0, 0.25, 0.50, 0.50),
            (0.25, 0.50, 1.0, 0.0),
            (0.50, 1.0, 0.0, 0.25),
            (0.0, 1.0, 0.25, 0.50),
        )
    )

    valid_region_values = [1, 2, 3, 4]
    valid_region_priority = [4, 3, 2, 1]

    varied_fuzzy_mask_list = [
        NiftiImageContainer(nib.Nifti1Image(varied_fuzzy_mask[i, :], np.eye(4)))
        for i in range(4)
    ]
    valid_fuzzy_mask_image = NiftiImageContainer(
        nib.Nifti1Image(varied_fuzzy_mask, np.eye(4))
    )
    valid_fuzzy_mask_list = [
        NiftiImageContainer(nib.Nifti1Image(np.ones((4, 4, 4)), np.eye(4)))
        for i in range(4)
    ]

    invalid_fuzzy_mask_list_shape = [
        NiftiImageContainer(nib.Nifti1Image(np.ones((4, 4, i)), np.eye(4)))
        for i in range(4)
    ]

    invalid_fuzzy_mask_list_affine = [
        NiftiImageContainer(nib.Nifti1Image(np.ones((4, 4, 4)), np.eye(4)))
        for i in range(4)
    ]
    invalid_fuzzy_mask_list_affine[3].nifti_image.set_sform(
        np.array(
            (
                (1.0, 0.0, 0.0, 10.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )
    )

    return {
        "fuzzy_mask_list": varied_fuzzy_mask_list,
        "fuzzy_mask_image": valid_fuzzy_mask_image,
        "region_values": valid_region_values,
        "region_priority": valid_region_priority,
        "input_validation_dict_fuzzy_mask_list": {
            "fuzzy_mask": [
                False,
                valid_fuzzy_mask_list,
                invalid_fuzzy_mask_list_shape,
                invalid_fuzzy_mask_list_affine,
                1,
                "str",
            ],
            "region_values": [False, valid_region_values, [1, 2, 3], 1, "str"],
            "region_priority": [
                False,
                valid_region_priority,
                [1, 1, 1, 1],
                [1, 2, 3],
                1,
                "str",
            ],
            "threshold": [True, 0.1, 2.0, -1.0, "str"],
        },
        "input_validation_dict_fuzzy_mask_image": {
            "fuzzy_mask": [False, valid_fuzzy_mask_image, 1, "str",],
            "region_values": [False, 1, "str"],
            "region_priority": [True, 1, 0, "str"],
            "threshold": [True, 0.1, 2.0, -1.0, "str"],
        },
    }


def test_combine_fuzzy_masks_filter_validate_inputs(validation_data: dict):
    """Check a FilterInputValidationError is raised when the inputs to the
    CombineFuzzyMasksFilter are incorrect or missing"""

    # check where fuzzy_mask is a list
    validate_filter_inputs(
        CombineFuzzyMasksFilter,
        validation_data["input_validation_dict_fuzzy_mask_list"],
    )

    # check where fuzzy_mask is a single image
    validate_filter_inputs(
        CombineFuzzyMasksFilter,
        validation_data["input_validation_dict_fuzzy_mask_image"],
    )


def test_combine_fuzzy_masks_filter_mock_data_list(validation_data: dict):
    """Test the CombineFuzzyMasksFilter with mock data for the case where
    there are more than one masks to combine"""

    combine_masks_filter = CombineFuzzyMasksFilter()
    combine_masks_filter.add_input("fuzzy_mask", validation_data["fuzzy_mask_list"])
    combine_masks_filter.add_input("region_values", validation_data["region_values"])
    combine_masks_filter.add_input(
        "region_priority", validation_data["region_priority"]
    )
    combine_masks_filter.run()
    numpy.testing.assert_array_equal(
        combine_masks_filter.outputs["seg_mask"].image, [1, 4, 2, 4]
    )

    # change the threshold from the default to 0.75
    combine_masks_filter = CombineFuzzyMasksFilter()
    combine_masks_filter.add_input("fuzzy_mask", validation_data["fuzzy_mask_list"])
    combine_masks_filter.add_input("region_values", validation_data["region_values"])
    combine_masks_filter.add_input("threshold", 0.75)
    combine_masks_filter.add_input(
        "region_priority", validation_data["region_priority"]
    )
    combine_masks_filter.run()
    numpy.testing.assert_array_equal(
        combine_masks_filter.outputs["seg_mask"].image, [1, 4, 2, 0]
    )

    # change the priority order
    combine_masks_filter = CombineFuzzyMasksFilter()
    combine_masks_filter.add_input("fuzzy_mask", validation_data["fuzzy_mask_list"])
    combine_masks_filter.add_input("region_values", validation_data["region_values"])
    combine_masks_filter.add_input("region_priority", [2, 4, 1, 3])
    combine_masks_filter.run()
    numpy.testing.assert_array_equal(
        combine_masks_filter.outputs["seg_mask"].image, [1, 3, 2, 1]
    )


def test_combine_fuzzy_masks_filter_priority_conflict():

    test_data = np.array((1.0))

    test_masks = [
        NiftiImageContainer(nib.Nifti1Image(test_data, np.eye(4))) for i in range(20)
    ]

    rng = default_rng(12345)
    priority = np.arange(1, 21, dtype=np.int16)
    rng.shuffle(priority)
    region_values = np.arange(1, 21)

    for i in enumerate(priority):
        loop_priority = np.roll(priority, i)
        combine_masks_filter = CombineFuzzyMasksFilter()
        combine_masks_filter.add_inputs(
            {
                "fuzzy_mask": test_masks,
                "region_values": region_values.tolist(),
                "region_priority": loop_priority.tolist(),
            }
        )
        combine_masks_filter.run()

        numpy.testing.assert_array_equal(
            combine_masks_filter.outputs["seg_mask"].image,
            region_values[loop_priority == 1],
        )


def test_combine_fuzzy_masks_filter_mock_data_image(validation_data: dict):
    """Test the CombineFuzzyMasksFilter with mock data for the case where
    there is only one mask"""
    # use default threshold
    combine_masks_filter = CombineFuzzyMasksFilter()
    combine_masks_filter.add_input("fuzzy_mask", validation_data["fuzzy_mask_image"])
    combine_masks_filter.add_input("region_values", 10)
    combine_masks_filter.run()
    numpy.testing.assert_array_equal(
        combine_masks_filter.outputs["seg_mask"].image,
        np.array(
            (
                (10.0, 10.0, 10, 10.0),
                (10.0, 10.0, 10.0, 0.0),
                (10.0, 10.0, 0.0, 10.0),
                (0.0, 10.0, 10.0, 10.0),
            )
        ),
    )

    # change threshold
    combine_masks_filter = CombineFuzzyMasksFilter()
    combine_masks_filter.add_input("fuzzy_mask", validation_data["fuzzy_mask_image"])
    combine_masks_filter.add_input("region_values", 10)
    combine_masks_filter.add_input("threshold", 0.75)
    combine_masks_filter.run()
    numpy.testing.assert_array_equal(
        combine_masks_filter.outputs["seg_mask"].image,
        np.array(
            (
                (10.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 10.0, 0.0),
                (0.0, 10.0, 0.0, 0.0),
                (0.0, 10.0, 0.0, 0.0),
            )
        ),
    )

