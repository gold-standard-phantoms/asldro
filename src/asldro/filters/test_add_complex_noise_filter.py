""" AddComplexNoiseFilter tests """

from copy import deepcopy
import pytest
import numpy as np
import numpy.testing

from asldro.filters.basefilter import FilterInputValidationError
from asldro.filters.json_loader import JsonLoaderFilter
from asldro.filters.ground_truth_loader import GroundTruthLoaderFilter
from asldro.filters.add_complex_noise_filter import AddComplexNoiseFilter
from asldro.containers.image import (
    BaseImageContainer,
    NiftiImageContainer,
    NumpyImageContainer,
)
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.data.filepaths import (
    HRGT_ICBM_2009A_NLS_V3_JSON,
    HRGT_ICBM_2009A_NLS_V3_NIFTI,
)


def test_add_complex_noise_filter_wrong_input_type_error():
    """ Check a FilterInputValidationError is raised when the inputs
    to the add commplex noise filter are incorrect or missing """
    noise_filter = AddComplexNoiseFilter()
    noise_filter.add_input("snr", 1)
    with pytest.raises(FilterInputValidationError):
        noise_filter.run()  # image not defined
    noise_filter.add_input("image", 1)
    with pytest.raises(FilterInputValidationError):
        noise_filter.run()  # image wrong type

    noise_filter = AddComplexNoiseFilter()
    noise_filter.add_input("image", NumpyImageContainer(image=np.zeros((32, 32, 32))))
    with pytest.raises(FilterInputValidationError):
        noise_filter.run()  # snr not defined
    noise_filter.add_input("snr", "str")
    with pytest.raises(FilterInputValidationError):
        noise_filter.run()  # snr wrong type

    noise_filter = AddComplexNoiseFilter()
    noise_filter.add_input("image", NumpyImageContainer(image=np.zeros((32, 32, 32))))
    noise_filter.add_input("snr", 1)
    noise_filter.add_input("reference_image", 1)
    with pytest.raises(FilterInputValidationError):
        noise_filter.run()  # reference_image wrong type

    noise_filter = AddComplexNoiseFilter()
    noise_filter.add_input("image", NumpyImageContainer(image=np.zeros((32, 32, 32))))
    noise_filter.add_input("snr", 1)
    noise_filter.add_input(
        "reference_image", NumpyImageContainer(image=np.zeros((32, 32, 32)))
    )


def add_complex_noise_function(
    image: np.ndarray, reference_image: np.ndarray, snr: float
):
    """
    Helper function for manually adding complex noise to an input array
    Inputs
        -image: numpy array containing the input image to add noise to
        -reference_image: reference image to use to calculate the noise amplitude
        -snr: signal to noise ratio
        -seed: seed for the random number generator
    """
    ft_image = np.fft.fftn(image)
    noise_amplitude = (
        np.sqrt(reference_image.size)
        * np.mean(reference_image[reference_image.nonzero()])
        / snr
    )

    noise_array_real = np.random.normal(0, noise_amplitude, ft_image.shape)
    noise_array_imag = np.random.normal(0, noise_amplitude, ft_image.shape)

    ft_image_noise = (np.real(ft_image) + noise_array_real) + 1j * (
        np.imag(ft_image) + noise_array_imag
    )
    return np.fft.ifftn(ft_image_noise)


def calculate_snr_dual_image_function(
    image_container: BaseImageContainer, snr: float, mask: np.ndarray = None
):
    """ Calculate the SNR of the images using the subtraction method
    Firbank et. al "A comparison of two methods for measuring the signal to
    noise ratio on MR images", PMB, vol 44, no. 12, pp.N261-N264 (1999)
    """
    noise_filter_1 = AddComplexNoiseFilter()
    noise_filter_1.add_input("image", image_container)
    noise_filter_1.add_input("snr", snr)
    noise_filter_2 = deepcopy(noise_filter_1)
    noise_filter_1.run()
    noise_filter_2.run()
    image_1 = np.abs(noise_filter_1.outputs["image"].image)
    image_2 = np.abs(noise_filter_2.outputs["image"].image)

    if mask is None:
        mask = np.ones(image_1.shape)

    diff = image_1 - image_2

    return np.sqrt(2) * (
        np.mean(image_1[mask.nonzero()]) / np.std(diff[mask.nonzero()])
    )


def test_add_complex_noise_filter_with_mock_data():
    """ Test the add complex noise filter with some data """
    signal_level = 100.0
    snr = 1000.0
    seed = 1234
    np.random.seed(seed)
    image = np.random.normal(signal_level, 10, (128, 128, 128))
    reference_image = image

    np.random.seed(seed)
    image_with_noise = add_complex_noise_function(image, reference_image, snr)

    image_container = NumpyImageContainer(image=image)
    reference_container = NumpyImageContainer(image=reference_image)
    noise_filter_1 = AddComplexNoiseFilter()
    noise_filter_1.add_input("image", image_container)
    noise_filter_1.add_input("snr", snr)

    # noise filter 2, copy of noise_filter_1
    noise_filter_2 = deepcopy(noise_filter_1)

    # noise filter 3 with reference
    noise_filter_3 = deepcopy(noise_filter_2)
    noise_filter_3.add_input("reference_image", reference_container)

    noise_filter_1.run()
    # reset RNG
    np.random.seed(seed)
    noise_filter_2.run()

    # reset RNG
    np.random.seed(seed)
    noise_filter_3.run()

    # Compare output of noise_filter_1 with image_with_noise
    # seed is different so they should not be equal
    with numpy.testing.assert_raises(AssertionError):
        numpy.testing.assert_array_equal(
            noise_filter_1.outputs["image"].image, image_with_noise
        )

    # Compare output of noise_filter_2 with image_with_noise
    # seed is the same so they should be equal
    numpy.testing.assert_array_equal(
        noise_filter_2.outputs["image"].image, image_with_noise
    )

    # Compare output of noise_filter_3 with image_with_noise
    # seed is the same so they should be equal
    numpy.testing.assert_array_equal(
        noise_filter_3.outputs["image"].image, image_with_noise
    )

    # Calculate the SNR of the images using the subtraction method
    calculated_snr = calculate_snr_dual_image_function(image_container, snr)
    print(f"calculated snr = {calculated_snr}, desired snr = {snr}")
    # This should be almost equal to the desired snr
    numpy.testing.assert_array_almost_equal(calculated_snr, snr, 0)


def test_add_complex_noise_filter_with_test_data():
    """ tests the complex noise filter with test data """
    json_filter = JsonLoaderFilter()
    json_filter.add_input("filename", HRGT_ICBM_2009A_NLS_V3_JSON)

    nifti_filter = NiftiLoaderFilter()
    nifti_filter.add_input("filename", HRGT_ICBM_2009A_NLS_V3_NIFTI)

    ground_truth_filter = GroundTruthLoaderFilter()
    ground_truth_filter.add_parent_filter(nifti_filter)
    ground_truth_filter.add_parent_filter(json_filter)

    # Load in the test data
    ground_truth_filter.run()

    image_container: NiftiImageContainer = ground_truth_filter.outputs["m0"]
    mask_container: NiftiImageContainer = ground_truth_filter.outputs["seg_label"]
    reference_container = image_container
    snr = 100.0
    seed = 1234

    np.random.seed(seed)
    image_with_noise = add_complex_noise_function(
        image_container.image, reference_container.image, snr
    )
    noise_filter_1 = AddComplexNoiseFilter()
    noise_filter_1.add_input("image", image_container)
    noise_filter_1.add_input("snr", snr)
    noise_filter_1.add_input("reference_image", reference_container)

    # reset RNG
    np.random.seed(seed)
    noise_filter_1.run()

    # Compare the output of noise filter 1 with image_with_noise, should be identical
    numpy.testing.assert_array_equal(
        noise_filter_1.outputs["image"].image, image_with_noise
    )

    # measure the actual SNR
    calculated_snr = calculate_snr_dual_image_function(
        image_container, snr, mask_container.image
    )
    print(f"calculated snr = {calculated_snr}, desired snr = {snr}")
    # This should be almost equal to the desired SNR
    numpy.testing.assert_array_almost_equal(calculated_snr, snr, 0)


if __name__ == "__main__":
    # test_add_complex_noise_filter_wrong_input_type_error()
    # test_add_complex_noise_filter_with_mock_data()
    test_add_complex_noise_filter_with_test_data()
