""" GroundTruthLoaderFilter tests """
# pylint: disable=duplicate-code
from copy import deepcopy

import pytest
import numpy as np
import numpy.testing
import nibabel as nib

from asldro.filters.basefilter import FilterInputValidationError
from asldro.filters.ground_truth_loader import GroundTruthLoaderFilter
from asldro.filters.json_loader import JsonLoaderFilter
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.containers.image import NiftiImageContainer
from asldro.data.filepaths import GROUND_TRUTH_DATA

TEST_VOLUME_DIMENSIONS = (3, 3, 3, 1, 7)
TEST_NIFTI_ONES = nib.Nifti2Image(
    np.ones(TEST_VOLUME_DIMENSIONS),
    affine=np.eye(4),
)

TEST_NIFTI_CON_ONES = NiftiImageContainer(nifti_img=TEST_NIFTI_ONES)

PAR_DICT = {
    "lambda_blood_brain": 0.9,
    "t1_arterial_blood": 1.65,
    "magnetic_field_strength": 3.0,
}

SEG_DICT = {"csf": 3, "grey_matter": 1, "white_matter": 2}

QUANTITIES = [
    "perfusion_rate",
    "transit_time",
    "t1",
    "t2",
    "t2_star",
    "m0",
    "seg_label",
]
UNITS = ["ml/100g/min", "s", "s", "s", "s", "", ""]

INPUT_VALIDATION_DICT = {
    "image": (False, TEST_NIFTI_CON_ONES, TEST_NIFTI_ONES, 1, "str"),
    "quantities": (False, QUANTITIES, "str", 1, PAR_DICT),
    "units": (False, UNITS, UNITS[:6], "str", 1, PAR_DICT),
    "parameters": (False, PAR_DICT, "str", TEST_NIFTI_CON_ONES, 1),
    "segmentation": (False, SEG_DICT, "str", TEST_NIFTI_CON_ONES, 1),
}


def test_ground_truth_loader_validate_inputs():
    """Check a GroundTruthLoaderFilter is raised when the inputs to the
    AppendMetadataFilter are incorrect or missing"""
    filter_to_test = GroundTruthLoaderFilter
    test_filter = filter_to_test()
    test_data = deepcopy(INPUT_VALIDATION_DICT)

    # check with inputs that should pass
    for data_key in test_data:
        test_filter.add_input(data_key, test_data[data_key][1])

    for inputs_key in INPUT_VALIDATION_DICT:
        test_data = deepcopy(INPUT_VALIDATION_DICT)
        test_filter = filter_to_test()
        is_optional: bool = test_data[inputs_key][0]

        # remove key
        test_data.pop(inputs_key)
        for data_key in test_data:
            test_filter.add_input(data_key, test_data[data_key][1])

        # optional inputs should run without issue
        if is_optional:
            test_filter.run()
        else:
            with pytest.raises(FilterInputValidationError):
                test_filter.run()

        # Try data that should fail
        for test_value in INPUT_VALIDATION_DICT[inputs_key][2:]:
            test_filter = filter_to_test()
            for data_key in test_data:
                test_filter.add_input(data_key, test_data[data_key][1])
            test_filter.add_input(inputs_key, test_value)

            with pytest.raises(FilterInputValidationError):
                test_filter.run()


def test_ground_truth_loader_filter_with_mock_data():
    """Test the ground truth loader filter with some mock data"""

    images = [np.ones(shape=(3, 3, 3, 1), dtype=np.float32) * i for i in range(7)]

    stacked_image = np.stack(arrays=images, axis=4)
    # Create a 5D numpy image (float32) where the value of voxel corresponds
    # with the distance across 5th dimension (from 0 to 6 inclusive)
    img = nib.Nifti2Image(dataobj=stacked_image, affine=np.eye(4))

    nifti_image_container = NiftiImageContainer(nifti_img=img)

    ground_truth_filter = GroundTruthLoaderFilter()
    ground_truth_filter.add_input("image", nifti_image_container)
    ground_truth_filter.add_input(
        "quantities",
        [
            "perfusion_rate",
            "transit_time",
            "t1",
            "t2",
            "t2_star",
            "m0",
            "seg_label",
        ],
    )
    ground_truth_filter.add_input(
        "segmentation", {"csf": 3, "grey_matter": 1, "white_matter": 2}
    )
    ground_truth_filter.add_input(
        "parameters",
        {
            "lambda_blood_brain": 0.9,
            "t1_arterial_blood": 1.65,
            "magnetic_field_strength": 3.0,
        },
    )
    ground_truth_filter.add_input("units", ["ml/100g/min", "s", "s", "s", "s", "", ""])

    # Should run without error
    ground_truth_filter.run()

    # Parameters should be piped through individually
    assert ground_truth_filter.outputs["lambda_blood_brain"] == 0.9
    assert ground_truth_filter.outputs["t1_arterial_blood"] == 1.65
    assert ground_truth_filter.outputs["magnetic_field_strength"] == 3.0

    assert ground_truth_filter.outputs["perfusion_rate"].image.dtype == np.float32
    numpy.testing.assert_array_equal(
        ground_truth_filter.outputs["perfusion_rate"].image,
        np.zeros((3, 3, 3), dtype=np.float32),
    )
    assert ground_truth_filter.outputs["perfusion_rate"].metadata == {
        "magnetic_field_strength": 3.0,
        "quantity": "perfusion_rate",
        "units": "ml/100g/min",
    }

    assert ground_truth_filter.outputs["transit_time"].image.dtype == np.float32
    numpy.testing.assert_array_equal(
        ground_truth_filter.outputs["transit_time"].image,
        np.ones((3, 3, 3), dtype=np.float32),
    )
    assert ground_truth_filter.outputs["transit_time"].metadata == {
        "magnetic_field_strength": 3.0,
        "quantity": "transit_time",
        "units": "s",
    }

    assert ground_truth_filter.outputs["t1"].image.dtype == np.float32
    numpy.testing.assert_array_equal(
        ground_truth_filter.outputs["t1"].image,
        np.ones((3, 3, 3), dtype=np.float32) * 2,
    )
    assert ground_truth_filter.outputs["t1"].metadata == {
        "magnetic_field_strength": 3.0,
        "quantity": "t1",
        "units": "s",
    }

    assert ground_truth_filter.outputs["t2"].image.dtype == np.float32
    numpy.testing.assert_array_equal(
        ground_truth_filter.outputs["t2"].image,
        np.ones((3, 3, 3), dtype=np.float32) * 3,
    )
    assert ground_truth_filter.outputs["t2"].metadata == {
        "magnetic_field_strength": 3.0,
        "quantity": "t2",
        "units": "s",
    }

    assert ground_truth_filter.outputs["t2_star"].image.dtype == np.float32
    numpy.testing.assert_array_equal(
        ground_truth_filter.outputs["t2_star"].image,
        np.ones((3, 3, 3), dtype=np.float32) * 4,
    )
    assert ground_truth_filter.outputs["t2_star"].metadata == {
        "magnetic_field_strength": 3.0,
        "quantity": "t2_star",
        "units": "s",
    }

    assert ground_truth_filter.outputs["m0"].image.dtype == np.float32
    numpy.testing.assert_array_equal(
        ground_truth_filter.outputs["m0"].image,
        np.ones((3, 3, 3), dtype=np.float32) * 5,
    )
    assert ground_truth_filter.outputs["m0"].metadata == {
        "magnetic_field_strength": 3.0,
        "quantity": "m0",
        "units": "",
    }

    # Check the seg_label type has changed to a uint16
    assert ground_truth_filter.outputs["seg_label"].image.dtype == np.uint16
    numpy.testing.assert_array_equal(
        ground_truth_filter.outputs["seg_label"].image,
        np.ones((3, 3, 3), dtype=np.uint16) * 6,
    )
    assert ground_truth_filter.outputs["seg_label"].metadata == {
        "magnetic_field_strength": 3.0,
        "quantity": "seg_label",
        "units": "",
        "segmentation": {
            "csf": 3,
            "grey_matter": 1,
            "white_matter": 2,
        },
    }


def test_ground_truth_loader_filter_with_test_data():
    """Test the ground truth loader filter with the included
    test data"""

    json_filter = JsonLoaderFilter()
    json_filter.add_input(
        "filename", GROUND_TRUTH_DATA["hrgt_icbm_2009a_nls_v3"]["json"]
    )

    nifti_filter = NiftiLoaderFilter()
    nifti_filter.add_input(
        "filename", GROUND_TRUTH_DATA["hrgt_icbm_2009a_nls_v3"]["nii"]
    )

    ground_truth_filter = GroundTruthLoaderFilter()
    ground_truth_filter.add_parent_filter(nifti_filter)
    ground_truth_filter.add_parent_filter(json_filter)

    # Should run without error
    ground_truth_filter.run()
