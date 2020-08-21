""" Add noise filter tests """

import pytest
import numpy as np
import numpy.testing

from asldro.filters.basefilter import FilterInputValidationError
from asldro.filters.json_loader import JsonLoaderFilter
from asldro.filters.ground_truth_loader import GroundTruthLoaderFilter
from asldro.filters.add_noise_filter import AddNoiseFilter
from asldro.filters.fourier_filter import FftFilter, IfftFilter

from asldro.containers.image import (
    BaseImageContainer,
    NiftiImageContainer,
    NumpyImageContainer,
)


def test_add_noise_filter_validate_inputs():
    """ Check a FilterInputValidationError is raised when the inputs
    to the add commplex noise filter are incorrect or missing """
    noise_filter = AddNoiseFilter()
    noise_filter.add_input("snr", 1)
    with pytest.raises(FilterInputValidationError):
        noise_filter.run()  # image not defined
    noise_filter.add_input("image", 1)
    with pytest.raises(FilterInputValidationError):
        noise_filter.run()  # image wrong type

    noise_filter = AddNoiseFilter()
    noise_filter.add_input("image", NumpyImageContainer(image=np.zeros((32, 32, 32))))
    with pytest.raises(FilterInputValidationError):
        noise_filter.run()  # snr not defined
    noise_filter.add_input("snr", "str")
    with pytest.raises(FilterInputValidationError):
        noise_filter.run()  # snr wrong type

    noise_filter = AddNoiseFilter()
    noise_filter.add_input("image", NumpyImageContainer(image=np.zeros((32, 32, 32))))
    noise_filter.add_input("snr", 1)
    noise_filter.add_input("reference_image", 1)
    with pytest.raises(FilterInputValidationError):
        noise_filter.run()  # reference_image wrong type

    noise_filter = AddNoiseFilter()
    noise_filter.add_input("image", NumpyImageContainer(image=np.zeros((32, 32, 32))))
    noise_filter.add_input("snr", 1)
    noise_filter.add_input(
        "reference_image", NumpyImageContainer(image=np.zeros((32, 32, 31)))
    )
    with pytest.raises(FilterInputValidationError):
        noise_filter.run()  # reference_image wrong shape


def add_noise_function(
    image: np.ndarray,
    snr: float,
    reference_image: np.ndarray = None,
    noise_scaling: float = 1.0,
):
    """
    Adds normally distributed random noise to an input array.

    Arguments:
        image (numpy.ndarray): numpy array containing the input image to add noise to
        reference_image (numpy.ndarray): reference image to use to calculate the noise amplitude
        snr (float): signal to noise ratio
        noise_scaling (float): scales the noise amplitude by this number
    
    Returns:
        numpy.ndarray: the input image with noise added 
    """
    if reference_image is None:
        reference_image = image

    noise_amplitude = (
        noise_scaling
        * np.mean(np.abs(reference_image[reference_image.nonzero()]))
        / snr
    )

    if image.dtype in [np.complex128, np.complex64]:
        image_noise = (
            np.real(image) + np.random.normal(0, noise_amplitude, image.shape)
        ) + 1j * (np.imag(image) + np.random.normal(0, noise_amplitude, image.shape))
    else:
        image_noise = image + np.random.normal(0, noise_amplitude, image.shape)

    return image_noise


def calculate_snr_function(
    image_1: np.ndarray, image_2: np.ndarray, mask: np.ndarray = None
):
    """calculates the snr from two image arrays
    
    Image arrays should be of the same object and with the same amplitude
    of normally distributed random noise added. The noise component must be different
    on each image.  The signal to noise ratio is calculated using the mean value (within
    and optional ROI defined by the input mask) divided by the standard deviation of the
    difference between image_1 and image_2.  This is in accordance with 
    "A comparison of two methods for measuring the signal to
    noise ratio on MR images", PMB, vol 44, no. 12, pp.N261-N264 (1999)

    Args:
        image_1 (np.ndarray): First image
        image_2 (np.ndarray): Second image
        mask (np.ndarray): mask, elements that are non-zero are used to define the
        object. If not supplied then all elements in the images will be considered.

    Returns:
        float: The calculate SNR
    """
    image_1 = np.abs(image_1)
    image_2 = np.abs(image_2)

    if mask is None:
        mask = np.ones(image_1.shape)

    diff = image_1 - image_2

    return np.sqrt(2) * (
        np.mean(image_1[mask.nonzero()]) / np.std(diff[mask.nonzero()])
    )


# Mock Data Fixtures
def image_container_function() -> NumpyImageContainer:
    """ Creates a NumpyImageContainer with mock real data """
    signal_level = 100.0
    np.random.seed(0)
    image = np.random.normal(signal_level, 10, (32, 32, 32))
    return NumpyImageContainer(image=image)


def complex_image_container_function() -> NumpyImageContainer:
    """ Creates a NumpyImageContainer with mock real data """
    signal_level = 100.0
    np.random.seed(0)
    image = np.random.normal(
        signal_level / np.sqrt(2), 10, (32, 32, 32)
    ) + 1j * np.random.normal(signal_level / np.sqrt(2), 10, (32, 32, 32))
    return NumpyImageContainer(image=image)


def ft_image_container_function(img: NumpyImageContainer) -> NumpyImageContainer:
    fft_filter = FftFilter()
    fft_filter.add_input("image", img)
    fft_filter.run()
    return fft_filter.outputs["image"]


@pytest.fixture(name="image_container")
def image_container_fixture() -> NumpyImageContainer:
    """ Fixture that creates and returns a NumpyImageContainer """
    return image_container_function()


@pytest.fixture(name="complex_image_container")
def complex_image_container_fixture() -> NumpyImageContainer:
    """ Fixture that creates and returns a NumpyImageContainer """
    return complex_image_container_function()


@pytest.fixture(name="ft_image_container")
def ft_image_container_fixture() -> NumpyImageContainer:
    """ Fixture that creates and returns the Fourier Transform of image_container """
    return ft_image_container_function(image_container_function())


@pytest.fixture(name="ft_complex_image_container")
def ft_complex_image_container_fixture() -> NumpyImageContainer:
    """ Fixture that creates and returns the Fourier Transform of image_container """
    return ft_image_container_function(complex_image_container_function())


@pytest.fixture
def snr_value() -> float:
    """ Returns the signal-to-noise ratio for testing """
    return 100.0


@pytest.fixture
def seed_value() -> int:
    """ Returns the seed for the random number generator """
    return 1234


# 1. add noise to non-complex image, only image supplied
def test_add_noise_filter_with_mock_data_mag_image_only(
    image_container, snr_value, seed_value
):
    """Test the add noise filter with magnitude (non-complex) image only
    """
    np.random.seed(seed_value)
    # calculate manually
    image_with_noise = add_noise_function(image_container.image, snr_value)

    # calculate using the filter
    add_noise_filter = AddNoiseFilter()
    add_noise_filter.add_input("image", image_container)
    add_noise_filter.add_input("snr", snr_value)
    np.random.seed(seed_value)  # set seed so RNG is in the same state
    add_noise_filter.run()

    image_with_noise_container = add_noise_filter.outputs["image"].clone()

    # image_with_noise and image_with_noise_container.image should be equal
    numpy.testing.assert_array_equal(image_with_noise, image_with_noise_container.image)
    # Run again and then check the SNR
    add_noise_filter = AddNoiseFilter()
    add_noise_filter.add_input("image", image_container)
    add_noise_filter.add_input("snr", snr_value)
    add_noise_filter.run()

    measured_snr = calculate_snr_function(
        image_with_noise_container.image, add_noise_filter.outputs["image"].image,
    )
    print(f"calculated snr = {measured_snr}, desired snr = {snr_value}")
    # This should be almost equal to the desired snr
    numpy.testing.assert_array_almost_equal(measured_snr, snr_value, 0)


# 2. add noise to non-complex image, image and reference in the spatial domain
def test_add_noise_filter_with_mock_data_mag_image_reference_same_domain(
    image_container, snr_value, seed_value
):
    """ Test the add noise filter with an image and reference image, both in the same domain """
    np.random.seed(seed_value)
    # calculate manually
    image_with_noise = add_noise_function(
        image_container.image, snr_value, image_container.image
    )
    # calculate using the filter
    add_noise_filter = AddNoiseFilter()
    add_noise_filter.add_input("image", image_container)
    add_noise_filter.add_input("snr", snr_value)
    add_noise_filter.add_input("reference_image", image_container)
    np.random.seed(seed_value)  # set seed so RNG is in the same state
    add_noise_filter.run()

    image_with_noise_container = add_noise_filter.outputs["image"].clone()

    # image_with_noise and image_with_noise_container.image should be equal
    numpy.testing.assert_array_equal(image_with_noise, image_with_noise_container.image)

    # Run again and then check the SNR
    add_noise_filter = AddNoiseFilter()
    add_noise_filter.add_input("image", image_container)
    add_noise_filter.add_input("snr", snr_value)
    add_noise_filter.add_input("reference_image", image_container)
    add_noise_filter.run()

    measured_snr = calculate_snr_function(
        image_with_noise_container.image, add_noise_filter.outputs["image"].image,
    )
    print(f"calculated snr = {measured_snr}, desired snr = {snr_value}")
    # This should be almost equal to the desired snr
    numpy.testing.assert_array_almost_equal(measured_snr, snr_value, 0)


# 3. add noise to non-complex image, image in spatial domain, reference in inverse domain
# Currently the calculated SNR does not match the desired
def test_add_noise_filter_with_mock_data_mag_image_spatial_reference_inverse(
    image_container, snr_value, seed_value, ft_image_container
):
    """ Test the add noise filter with an image and reference image, image in SPATIAL_DOMAIN,
    reference in INVERSE_DOMAIN """
    np.random.seed(seed_value)
    # calculate manually
    image_with_noise = add_noise_function(
        image_container.image,
        snr_value,
        ft_image_container.image,
        1 / np.sqrt(ft_image_container.image.size),
    )
    image_with_noise_2 = add_noise_function(
        image_container.image,
        snr_value,
        ft_image_container.image,
        1 / np.sqrt(ft_image_container.image.size),
    )
    print(f"manual snr = {calculate_snr_function(image_with_noise,image_with_noise_2)}")
    # calculate using the filter
    add_noise_filter = AddNoiseFilter()
    add_noise_filter.add_input("image", image_container)
    add_noise_filter.add_input("snr", snr_value)
    add_noise_filter.add_input("reference_image", ft_image_container)
    np.random.seed(seed_value)  # set seed so RNG is in the same state
    add_noise_filter.run()

    image_with_noise_container = add_noise_filter.outputs["image"].clone()

    # image_with_noise and image_with_noise_container.image should be equal
    numpy.testing.assert_array_equal(image_with_noise, image_with_noise_container.image)

    # Run again and then check the SNR
    add_noise_filter = AddNoiseFilter()
    add_noise_filter.add_input("image", image_container)
    add_noise_filter.add_input("snr", snr_value)
    add_noise_filter.add_input("reference_image", ft_image_container)
    add_noise_filter.run()

    measured_snr = calculate_snr_function(
        image_with_noise_container.image, add_noise_filter.outputs["image"].image,
    )
    print(f"calculated snr = {measured_snr}, desired snr = {snr_value}")
    # This isn't equal to the desired SNR
    with numpy.testing.assert_raises(AssertionError):
        numpy.testing.assert_array_almost_equal(measured_snr, snr_value, 0)


# 4. add noise to complex image, no reference supplied
def test_add_noise_filter_with_mock_data_complex_image(
    complex_image_container, snr_value, seed_value
):
    """ Test the add noise filter with a complex image """
    np.random.seed(seed_value)
    # calculate manually
    image_with_noise = add_noise_function(complex_image_container.image, snr_value,)
    image_with_noise_2 = add_noise_function(complex_image_container.image, snr_value,)
    print(f"manual snr = {calculate_snr_function(image_with_noise,image_with_noise_2)}")

    # calculate using the filter
    add_noise_filter = AddNoiseFilter()
    add_noise_filter.add_input("image", complex_image_container)
    add_noise_filter.add_input("snr", snr_value)
    np.random.seed(seed_value)  # set seed so RNG is in the same state
    add_noise_filter.run()

    image_with_noise_container = add_noise_filter.outputs["image"].clone()

    # image_with_noise and image_with_noise_container.image should be equal
    numpy.testing.assert_array_equal(image_with_noise, image_with_noise_container.image)

    # Run again and then check the SNR
    add_noise_filter = AddNoiseFilter()
    add_noise_filter.add_input("image", complex_image_container)
    add_noise_filter.add_input("snr", snr_value)
    add_noise_filter.run()

    measured_snr = calculate_snr_function(
        image_with_noise_container.image, add_noise_filter.outputs["image"].image,
    )
    print(f"calculated snr = {measured_snr}, desired snr = {snr_value}")
    # This should be almost equal to the desired snr
    numpy.testing.assert_array_almost_equal(measured_snr, snr_value, 0)


# 5. add noise to complex image, image and reference in inverse domain
def test_add_noise_filter_with_mock_data_complex_image_reference_inverse(
    ft_complex_image_container, snr_value, seed_value
):
    """ Test the add noise filter with a complex data in the INVERSE_DOMAIN """
    np.random.seed(seed_value)
    # calculate manually
    image_with_noise = add_noise_function(ft_complex_image_container.image, snr_value,)
    image_with_noise_2 = add_noise_function(
        ft_complex_image_container.image, snr_value,
    )
    print(f"manual snr = {calculate_snr_function(image_with_noise,image_with_noise_2)}")

    # calculate using the filter
    add_noise_filter = AddNoiseFilter()
    add_noise_filter.add_input("image", ft_complex_image_container)
    add_noise_filter.add_input("snr", snr_value)
    np.random.seed(seed_value)  # set seed so RNG is in the same state
    add_noise_filter.run()

    image_with_noise_container = add_noise_filter.outputs["image"].clone()

    # image_with_noise and image_with_noise_container.image should be equal
    numpy.testing.assert_array_equal(image_with_noise, image_with_noise_container.image)

    # Run again and then check the SNR
    add_noise_filter = AddNoiseFilter()
    add_noise_filter.add_input("image", ft_complex_image_container)
    add_noise_filter.add_input("snr", snr_value)
    add_noise_filter.run()

    measured_snr = calculate_snr_function(
        image_with_noise_container.image, add_noise_filter.outputs["image"].image,
    )
    print(f"calculated snr = {measured_snr}, desired snr = {snr_value}")
    # This should be almost equal to the desired snr
    numpy.testing.assert_array_almost_equal(measured_snr, snr_value, 0)


# 6. add noise to complex image, image in spatial domain, reference in inverse domain
# Currently the calculated SNR does not match the desired
def test_add_noise_filter_with_mock_data_complex_image_spatial_reference_inverse(
    complex_image_container, ft_complex_image_container, snr_value, seed_value
):
    """ Test the add noise filter with a complex image SPATIAL_DOMAIN, complex reference in the INVERSE_DOMAIN """
    np.random.seed(seed_value)
    # calculate manually
    image_with_noise = add_noise_function(
        complex_image_container.image,
        snr_value,
        ft_complex_image_container.image,
        1.0 / np.sqrt(ft_complex_image_container.image.size),
    )
    image_with_noise_2 = add_noise_function(
        complex_image_container.image,
        snr_value,
        ft_complex_image_container.image,
        1.0 / np.sqrt(ft_complex_image_container.image.size),
    )
    print(f"manual snr = {calculate_snr_function(image_with_noise,image_with_noise_2)}")

    # calculate using the filter
    add_noise_filter = AddNoiseFilter()
    add_noise_filter.add_input("image", complex_image_container)
    add_noise_filter.add_input("snr", snr_value)
    add_noise_filter.add_input("reference_image", ft_complex_image_container)
    np.random.seed(seed_value)  # set seed so RNG is in the same state
    add_noise_filter.run()

    image_with_noise_container = add_noise_filter.outputs["image"].clone()

    # image_with_noise and image_with_noise_container.image should be equal
    numpy.testing.assert_array_equal(image_with_noise, image_with_noise_container.image)

    # Run again and then check the SNR
    add_noise_filter = AddNoiseFilter()
    add_noise_filter.add_input("image", complex_image_container)
    add_noise_filter.add_input("snr", snr_value)
    add_noise_filter.add_input("reference_image", ft_complex_image_container)
    add_noise_filter.run()

    measured_snr = calculate_snr_function(
        image_with_noise_container.image, add_noise_filter.outputs["image"].image,
    )
    print(f"calculated snr = {measured_snr}, desired snr = {snr_value}")
    # This isn't equal to the desired SNR
    with numpy.testing.assert_raises(AssertionError):
        numpy.testing.assert_array_almost_equal(measured_snr, snr_value, 0)


# 7. add noise to complex image, image in inverse domain, reference in spatial domain
# Currently the calculated SNR does not match the desired
def test_add_noise_filter_with_mock_data_complex_image_inverse_reference_spatial(
    complex_image_container, ft_complex_image_container, snr_value, seed_value
):
    """ Test the add noise filter with a complex image SPATIAL_DOMAIN, complex reference in the INVERSE_DOMAIN """
    np.random.seed(seed_value)
    # calculate manually
    image_with_noise = add_noise_function(
        ft_complex_image_container.image,
        snr_value,
        complex_image_container.image,
        np.sqrt(complex_image_container.image.size),
    )
    image_with_noise_2 = add_noise_function(
        ft_complex_image_container.image,
        snr_value,
        complex_image_container.image,
        np.sqrt(complex_image_container.image.size),
    )
    print(f"manual snr = {calculate_snr_function(image_with_noise,image_with_noise_2)}")

    # calculate using the filter
    add_noise_filter = AddNoiseFilter()
    add_noise_filter.add_input("image", ft_complex_image_container)
    add_noise_filter.add_input("snr", snr_value)
    add_noise_filter.add_input("reference_image", complex_image_container)
    np.random.seed(seed_value)  # set seed so RNG is in the same state
    add_noise_filter.run()

    image_with_noise_container = add_noise_filter.outputs["image"].clone()

    # image_with_noise and image_with_noise_container.image should be equal
    numpy.testing.assert_array_equal(image_with_noise, image_with_noise_container.image)

    # Run again and then check the SNR
    add_noise_filter = AddNoiseFilter()
    add_noise_filter.add_input("image", ft_complex_image_container)
    add_noise_filter.add_input("snr", snr_value)
    add_noise_filter.add_input("reference_image", complex_image_container)
    add_noise_filter.run()

    measured_snr = calculate_snr_function(
        image_with_noise_container.image, add_noise_filter.outputs["image"].image,
    )
    print(f"calculated snr = {measured_snr}, desired snr = {snr_value}")
    # This isn't equal to the desired SNR
    with numpy.testing.assert_raises(AssertionError):
        numpy.testing.assert_array_almost_equal(measured_snr, snr_value, 0)


if __name__ == "__main__":
    # test_add_noise_filter_validate_inputs()
    img = image_container_function()
    SNR = snr_value()
    SEED = seed_value()
    test_add_noise_filter_with_mock_data_mag_image_only(img, SNR, SEED)

