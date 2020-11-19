""" PhaseMagnitudeFilter tests """

from typing import Tuple
import pytest
import numpy as np
import numpy.testing as nptesting
from numpy.random import default_rng
import nibabel as nib
from asldro.filters.basefilter import FilterInputValidationError

from asldro.filters.phase_magnitude_filter import PhaseMagnitudeFilter
from asldro.containers.image import (
    BaseImageContainer,
    COMPLEX_IMAGE_TYPE,
    MAGNITUDE_IMAGE_TYPE,
    NiftiImageContainer,
    PHASE_IMAGE_TYPE,
    SPATIAL_DOMAIN,
)


@pytest.fixture(name="test_data")
def fixture_test_image():
    """Create a random test NiftiImageContainer.
    The data are in COMPLEX space, created from images that have
    - Magnitude values drawn from a standard Normal distribution (mean=0, stdev=1)
    - Phase values drawn from a uniform distribution (low=-pi, high=pi)
    - Shape: 3,4,5,6
    :return: a tuple of:
    - the magnitude data (np.ndarray, type: float64)
    - the phase data (np.ndarray, type: float64)
    - the complex data (np.ndarray, type: complex128) in an Image Container
    """
    random_number_generator = default_rng()
    magnitude_data = random_number_generator.uniform(low=0, high=1e4, size=(3, 4, 5, 6))
    phase_data = random_number_generator.uniform(
        low=-np.pi, high=np.pi, size=(3, 4, 5, 6)
    )
    complex_data = magnitude_data * np.exp(1j * phase_data)

    return (
        magnitude_data,
        phase_data,
        NiftiImageContainer(
            nifti_img=nib.Nifti1Image(
                dataobj=complex_data,
                affine=np.eye(4),
            ),
            image_type=COMPLEX_IMAGE_TYPE,
            data_domain=SPATIAL_DOMAIN,
        ),
    )


def test_phase_magnitude_filter(
    test_data: Tuple[np.ndarray, np.ndarray, BaseImageContainer]
):
    """Test the phase_magnitude_filter with known inputs and outputs"""
    magnitude_data, phase_data, image_container = test_data
    phase_magnitude_filter = PhaseMagnitudeFilter()
    phase_magnitude_filter.add_input(PhaseMagnitudeFilter.KEY_IMAGE, image_container)
    phase_magnitude_filter.run()
    nptesting.assert_array_almost_equal(
        phase_magnitude_filter.outputs[PhaseMagnitudeFilter.KEY_MAGNITUDE].image,
        magnitude_data,
    )
    assert (
        phase_magnitude_filter.outputs[PhaseMagnitudeFilter.KEY_MAGNITUDE].image_type
        == MAGNITUDE_IMAGE_TYPE
    )
    assert (
        phase_magnitude_filter.outputs[PhaseMagnitudeFilter.KEY_MAGNITUDE].data_domain
        == SPATIAL_DOMAIN
    )

    nptesting.assert_array_almost_equal(
        phase_magnitude_filter.outputs[PhaseMagnitudeFilter.KEY_PHASE].image,
        phase_data,
    )
    assert (
        phase_magnitude_filter.outputs[PhaseMagnitudeFilter.KEY_PHASE].image_type
        == PHASE_IMAGE_TYPE
    )
    assert (
        phase_magnitude_filter.outputs[PhaseMagnitudeFilter.KEY_PHASE].data_domain
        == SPATIAL_DOMAIN
    )


def test_phase_magnitude_filter_validator_non_image_input():
    """Run the phase_magnitude_filter with a non-image input.
    Check we get a FilterInputValidationError"""
    # non image input
    phase_magnitude_filter = PhaseMagnitudeFilter()
    phase_magnitude_filter.add_input(PhaseMagnitudeFilter.KEY_IMAGE, 0)
    with pytest.raises(FilterInputValidationError):
        phase_magnitude_filter.run()


def test_phase_magnitude_filter_validator_non_complex_input(
    test_data: Tuple[np.ndarray, np.ndarray, BaseImageContainer]
):
    """Run the phase_magnitude_filter with non-complex data.
    Check we get a FilterInputValidationError"""
    _, _, image_container = test_data

    # Non complex input
    image_container.image_type = MAGNITUDE_IMAGE_TYPE

    phase_magnitude_filter = PhaseMagnitudeFilter()
    phase_magnitude_filter.add_input(PhaseMagnitudeFilter.KEY_IMAGE, image_container)
    with pytest.raises(FilterInputValidationError):
        phase_magnitude_filter.run()
