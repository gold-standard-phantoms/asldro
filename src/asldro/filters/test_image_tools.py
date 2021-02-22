""" Tests for image_tools.py """

import pytest
import numpy as np
import numpy.testing
import nibabel as nib

from asldro.containers.image import NiftiImageContainer
from asldro.filters.image_tools import FloatToIntImageFilter
from asldro.filters.basefilter import FilterInputValidationError
from asldro.utils.filter_validation import validate_filter_inputs

# [multiplier, expected_datatype]
FTI_DTYPE_TEST_DATA = [
    [2 ** 16 - 1, np.uint16],
    [2 ** 16, np.uint32],
    [2 ** 32 - 1, np.uint32],
    [2 ** 32, np.uint64],
    [2 ** 64 - 1, np.uint64],
    [1, np.uint16],
    [-1, np.int16],
    [-(2 ** 15) + 1, np.int16],
    [-(2 ** 15), np.int32],
    [-(2 ** 31) + 1, np.int32],
    [-(2 ** 31), np.int64],
    [-(2 ** 63) + 1, np.int64],
]

# [multiplier, method, expected]
FTI_METHOD_TEST_DATA = [
    [1.1, "round", 1],
    [1.1, "ceil", 2],
    [1.1, "floor", 1],
    [1.1, "truncate", 1],
    [1.9, "round", 2],
    [1.9, "ceil", 2],
    [1.9, "floor", 1],
    [1.9, "truncate", 1],
    [-1.1, "round", -1],
    [-1.1, "ceil", -1],
    [-1.1, "floor", -2],
    [-1.1, "truncate", -1],
    [-1.9, "round", -2],
    [-1.9, "ceil", -1],
    [-1.9, "floor", -2],
    [-1.9, "truncate", -1],
]


@pytest.fixture(name="validation_data")
def input_validation_data_fixture():
    """Returns a dictionary containing test data for the filter input validation"""

    test_image_float = NiftiImageContainer(
        nib.Nifti1Image(np.ones((4, 4, 4)), np.eye(4))
    )

    test_image_complex = NiftiImageContainer(
        nib.Nifti1Image(np.ones((4, 4, 4)).astype(np.complex64), np.eye(4))
    )

    return {
        "input_validation_dict": {
            "image": [False, test_image_float, test_image_complex, 1.0, "str"],
            "method": [True, "round", "rnd", "str", 1.0],
        },
        "image": test_image_float,
    }


def test_float_to_int_filter_validate_inputs(validation_data: dict):
    """Check that a FilterInputValidationError is raised when the inputs to the
    FloatToIntImageFilter filter are incorrect or missing"""
    validate_filter_inputs(
        FloatToIntImageFilter, validation_data["input_validation_dict"]
    )


@pytest.mark.parametrize("multiplier, expected_datatype", FTI_DTYPE_TEST_DATA)
def test_float_to_int_filter_data_types(multiplier: float, expected_datatype: np.dtype):
    """Test that the FloatToIntImageFilter returns the correct datatype based on
    the values in the input image"""

    float_to_int_filter = FloatToIntImageFilter()
    input_image = NiftiImageContainer(
        nib.Nifti1Image(np.ones((4, 4, 4)) * multiplier, np.eye(4))
    )
    float_to_int_filter.add_input("image", input_image)
    float_to_int_filter.run()
    # check the output data type
    output_image: NiftiImageContainer = float_to_int_filter.outputs["image"]
    assert output_image.image.dtype == expected_datatype
    numpy.testing.assert_array_equal(
        output_image.image, np.rint(input_image.image).astype(expected_datatype),
    )


@pytest.mark.parametrize("multiplier, method, expected", FTI_METHOD_TEST_DATA)
def test_float_to_int_filter_methods(multiplier, method, expected):
    """Tests that the FloatToIntImageFilter returns expected values based on
    the chosen method"""

    float_to_int_filter = FloatToIntImageFilter()
    input_image = NiftiImageContainer(
        nib.Nifti1Image(np.ones((4, 4, 4)) * multiplier, np.eye(4))
    )
    float_to_int_filter.add_input("image", input_image)
    float_to_int_filter.add_input("method", method)
    float_to_int_filter.run()
    output_image: NiftiImageContainer = float_to_int_filter.outputs["image"]
    numpy.testing.assert_array_equal(
        output_image.image, expected,
    )
