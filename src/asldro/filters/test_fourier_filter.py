""" FftFilter and IfftFilter tests """
# pylint: disable=duplicate-code

import pytest
import numpy as np
import numpy.testing
from asldro.filters.basefilter import FilterInputValidationError
from asldro.filters.fourier_filter import FftFilter, IfftFilter
from asldro.filters.json_loader import JsonLoaderFilter
from asldro.filters.ground_truth_loader import GroundTruthLoaderFilter
from asldro.containers.image import (
    COMPLEX_IMAGE_TYPE,
    NiftiImageContainer,
    NumpyImageContainer,
    SPATIAL_DOMAIN,
    INVERSE_DOMAIN,
)
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.data.filepaths import GROUND_TRUTH_DATA

TEST_VOLUME_DIMENSIONS = (32, 32, 32)


def test_fourier_filters_ifft_validation():
    """Check that running an ifft on in SPATIAL_DOMAIN image raises a
    FilterInputValidationError"""
    image_data = np.random.normal(0, 1, TEST_VOLUME_DIMENSIONS)
    image_container = NumpyImageContainer(image=image_data, data_domain=SPATIAL_DOMAIN)

    ifft_filter = IfftFilter()
    ifft_filter.add_input("image", image_container)
    with pytest.raises(FilterInputValidationError):
        ifft_filter.run()


def test_fourier_filters_fft_validation():
    """Check that running an fft on an INVERSE_DOMAIN image raises a
    FilterInputValidationError"""
    image_data = np.random.normal(0, 1, TEST_VOLUME_DIMENSIONS)
    image_container = NumpyImageContainer(image=image_data, data_domain=INVERSE_DOMAIN)

    fft_filter = FftFilter()
    fft_filter.add_input("image", image_container)
    with pytest.raises(FilterInputValidationError):
        fft_filter.run()


def test_fourier_filter_wrong_input_type_error():
    """Check a FilterInputValidationError is raises when the inputs
    to the fourier filter `image` is incorrect or missing"""

    ifft_filter = IfftFilter()
    ifft_filter.add_input("dummy", 1)  # won't run without input
    with pytest.raises(FilterInputValidationError):
        ifft_filter.run()  # image not defined
    ifft_filter.add_input("image", 1)
    with pytest.raises(FilterInputValidationError):
        ifft_filter.run()  # image wrong type

    fft_filter = FftFilter()
    fft_filter.add_input("dummy", 1)  # won't run without input
    with pytest.raises(FilterInputValidationError):
        fft_filter.run()  # image not defined
    fft_filter.add_input("image", 1)
    with pytest.raises(FilterInputValidationError):
        fft_filter.run()  # image wrong type


def test_fourier_filters_with_mock_data():
    """Test the fft filter with some data + its discrete fourier transform"""
    # Create a 3D numpy image of normally distributed noise
    # fft to obtain k-space data, then ifft that to go back
    # to the image
    image_data = np.random.normal(0, 1, TEST_VOLUME_DIMENSIONS)
    kspace_data = np.fft.fftn(image_data)
    inverse_transformed_image_data = np.fft.ifftn(kspace_data)
    image_container = NumpyImageContainer(image=image_data)

    fft_filter = FftFilter()
    fft_filter.add_input("image", image_container)
    ifft_filter = IfftFilter()
    ifft_filter.add_parent_filter(parent=fft_filter)

    # Should run without error
    ifft_filter.run()

    # Check that the output of the fft_filter is in the INVERSE_DOMAIN
    assert fft_filter.outputs["image"].data_domain == INVERSE_DOMAIN

    # Check the output image is labelled as COMPLEX
    assert fft_filter.outputs["image"].image_type == COMPLEX_IMAGE_TYPE

    # Compare the fft_filter output image with kspace_data
    numpy.testing.assert_array_equal(fft_filter.outputs["image"].image, kspace_data)

    # Check that the output of the ifft_filter is in the SPATIAL_DOMAIN
    assert ifft_filter.outputs["image"].data_domain == SPATIAL_DOMAIN

    # Check the output image is labelled as COMPLEX
    assert ifft_filter.outputs["image"].image_type == COMPLEX_IMAGE_TYPE

    # Compare the ifft_filter_output image with inverse_transformed_image_data
    numpy.testing.assert_array_equal(
        ifft_filter.outputs["image"].image, inverse_transformed_image_data
    )


@pytest.mark.slow
def test_fourier_filters_with_test_data():
    """ Tests the fourier filters with some test data """
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

    # Load in the test data
    ground_truth_filter.run()

    m0_image: NiftiImageContainer = ground_truth_filter.outputs["m0"]

    m0_image_array = m0_image.image
    m0_kspace_array = np.fft.fftn(m0_image_array)
    inverse_transformed_m0_image_array = np.fft.ifftn(m0_kspace_array)

    fft_filter = FftFilter()
    fft_filter.add_input("image", m0_image)
    ifft_filter = IfftFilter()
    ifft_filter.add_parent_filter(parent=fft_filter)

    # Should run without error
    ifft_filter.run()

    # Check that the output of the fft_filter is in the INVERSE_DOMAIN
    assert fft_filter.outputs["image"].data_domain == INVERSE_DOMAIN

    # Check the output image is labelled as COMPLEX
    assert fft_filter.outputs["image"].image_type == COMPLEX_IMAGE_TYPE

    # Compare the fft_filter output image with kspace_data
    numpy.testing.assert_array_equal(fft_filter.outputs["image"].image, m0_kspace_array)

    # Check that the output of the ifft_filter is in the SPATIAL_DOMAIN
    assert ifft_filter.outputs["image"].data_domain == SPATIAL_DOMAIN

    # Check the output image is labelled as COMPLEX
    assert ifft_filter.outputs["image"].image_type == COMPLEX_IMAGE_TYPE

    # Compare the ifft_filter_output image with inverse_transformed_image_data
    numpy.testing.assert_array_equal(
        ifft_filter.outputs["image"].image, inverse_transformed_m0_image_array
    )
