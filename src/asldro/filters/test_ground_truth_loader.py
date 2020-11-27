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


@pytest.fixture(name="input_validation_dict")
def input_validation_dict_fixture():
    """Returns an object of tuples containing test data for
    input validation of the GroundTruthLoaderFilter"""
    test_volume_dimensions = (3, 3, 3, 1, 7)
    test_nifti_ones = nib.Nifti2Image(
        np.ones(test_volume_dimensions),
        affine=np.eye(4),
    )

    test_nifti_con_ones = NiftiImageContainer(nifti_img=test_nifti_ones)

    par_dict = {
        "lambda_blood_brain": 0.9,
        "t1_arterial_blood": 1.65,
        "magnetic_field_strength": 3.0,
    }

    seg_dict = {"csf": 3, "grey_matter": 1, "white_matter": 2}

    quantities = [
        "perfusion_rate",
        "transit_time",
        "t1",
        "t2",
        "t2_star",
        "m0",
        "seg_label",
    ]
    units = ["ml/100g/min", "s", "s", "s", "s", "", ""]

    # tuples where
    # 0:      is_optional
    # 1:      correct values
    # 2:end   values that should fail
    return {
        "image": (False, test_nifti_con_ones, test_nifti_ones, 1, "str"),
        "quantities": (False, quantities, "str", 1, par_dict),
        "units": (False, units, units[:6], "str", 1, par_dict),
        "parameters": (False, par_dict, "str", test_nifti_con_ones, 1),
        "segmentation": (False, seg_dict, "str", test_nifti_con_ones, 1),
        "image_override": (
            True,  # is optional
            {"m0": 10, "t2": 1.1},  # good data
            {"bad_key": 10, "t2": 1.1},  # no bad_key quantity
            {"m0": "a_str"},  # values must be int/float
            "a_str",  # must be dictionary input
        ),
        "parameter_override": (
            True,  # is optional
            {"lambda_blood_brain": 10, "t1_arterial_blood": 1.1},  # good data
            {"lambda_blood_brain": "a_str"},  # values must be int/float
            "a_str",  # must be dictionary input
        ),
    }


def test_ground_truth_loader_validate_inputs(input_validation_dict: dict):
    """Check a GroundTruthLoaderFilter is raised when the inputs to the
    AppendMetadataFilter are incorrect or missing"""
    filter_to_test = GroundTruthLoaderFilter
    test_filter = filter_to_test()
    test_data = deepcopy(input_validation_dict)

    # check with inputs that should pass
    for data_key in test_data:
        test_filter.add_input(data_key, test_data[data_key][1])

    for inputs_key in input_validation_dict:
        test_data = deepcopy(input_validation_dict)
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
        for test_value in input_validation_dict[inputs_key][2:]:
            test_filter = filter_to_test()
            for data_key in test_data:
                test_filter.add_input(data_key, test_data[data_key][1])
            test_filter.add_input(inputs_key, test_value)

            with pytest.raises(FilterInputValidationError):
                test_filter.run()


@pytest.fixture(name="mock_data")
def mock_data_fixture() -> dict:
    """Mock data inputs for GroundTruthLoaderFilter"""
    images = [np.ones(shape=(3, 3, 3, 1), dtype=np.float32) * i for i in range(7)]

    stacked_image = np.stack(arrays=images, axis=4)
    # Create a 5D numpy image (float32) where the value of voxel corresponds
    # with the distance across 5th dimension (from 0 to 6 inclusive)
    img = nib.Nifti2Image(dataobj=stacked_image, affine=np.eye(4))

    nifti_image_container = NiftiImageContainer(nifti_img=img)

    return {
        "image": nifti_image_container,
        "quantities": [
            "perfusion_rate",
            "transit_time",
            "t1",
            "t2",
            "t2_star",
            "m0",
            "seg_label",
        ],
        "segmentation": {"csf": 3, "grey_matter": 1, "white_matter": 2},
        "parameters": {
            "lambda_blood_brain": 0.9,
            "t1_arterial_blood": 1.65,
            "magnetic_field_strength": 3.0,
        },
        "units": ["ml/100g/min", "s", "s", "s", "s", "", ""],
    }


def test_ground_truth_filter_apply_scale_offset(mock_data: dict):
    """Test the ground truth loader when applying a scale and offset to the
    ground truth data"""
    ground_truth_filter = GroundTruthLoaderFilter()
    ground_truth_filter.add_inputs(mock_data)
    ground_truth_filter.add_input(
        "ground_truth_modulate",
        {
            "t1": {"scale": 5},
            "t2": {"offset": -1},
            "t2_star": {"scale": 0.5, "offset": 5},
        },
    )
    ground_truth_filter.run()
    assert ground_truth_filter.outputs["t1"].image.dtype == np.float32
    numpy.testing.assert_array_equal(
        ground_truth_filter.outputs["t1"].image,
        (np.ones((3, 3, 3), dtype=np.float32) * 2) * 5,
    )

    assert ground_truth_filter.outputs["t2"].image.dtype == np.float32
    numpy.testing.assert_array_equal(
        ground_truth_filter.outputs["t2"].image,
        (np.ones((3, 3, 3), dtype=np.float32) * 3) - 1,
    )

    assert ground_truth_filter.outputs["t2_star"].image.dtype == np.float32
    numpy.testing.assert_array_equal(
        ground_truth_filter.outputs["t2_star"].image,
        (np.ones((3, 3, 3), dtype=np.float32) * 4) * 0.5 + 5,
    )


def test_ground_truth_loader_filter_with_mock_data(mock_data: dict):
    """Test the ground truth loader filter with some mock data"""
    ground_truth_filter = GroundTruthLoaderFilter()
    ground_truth_filter.add_inputs(mock_data)
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
        "filename", GROUND_TRUTH_DATA["hrgt_icbm_2009a_nls_3t"]["json"]
    )

    nifti_filter = NiftiLoaderFilter()
    nifti_filter.add_input(
        "filename", GROUND_TRUTH_DATA["hrgt_icbm_2009a_nls_3t"]["nii"]
    )

    ground_truth_filter = GroundTruthLoaderFilter()
    ground_truth_filter.add_parent_filter(nifti_filter)
    ground_truth_filter.add_parent_filter(json_filter)

    # Should run without error
    ground_truth_filter.run()


def test_ground_truth_loader_filter_with_image_overrides(mock_data: dict):
    """Test the image_override input functionality"""
    mock_data[GroundTruthLoaderFilter.KEY_IMAGE_OVERRIDE] = {
        "m0": 5,
        "t1": 1.0,
    }
    gt_filter = GroundTruthLoaderFilter()
    gt_filter.add_inputs(mock_data)
    gt_filter.run()
    numpy.testing.assert_array_equal(
        gt_filter.outputs["m0"].image,
        np.full(shape=(3, 3, 3), fill_value=5),
    )
    numpy.testing.assert_array_equal(
        gt_filter.outputs["t1"].image,
        np.full(shape=(3, 3, 3), fill_value=1.0),
    )


def test_ground_truth_loader_filter_with_parameter_overrides(mock_data: dict):
    """Test the parameter_override input functionality"""
    mock_data[GroundTruthLoaderFilter.KEY_PARAMETER_OVERRIDE] = {
        "lambda_blood_brain": 3.1,
        "a_new_parameter": 1,
    }
    gt_filter = GroundTruthLoaderFilter()
    gt_filter.add_inputs(mock_data)
    gt_filter.run()
    for item in {
        "lambda_blood_brain": 3.1,
        "t1_arterial_blood": 1.65,
        "magnetic_field_strength": 3.0,
        "a_new_parameter": 1,
    }.items():
        assert item in gt_filter.outputs.items()
