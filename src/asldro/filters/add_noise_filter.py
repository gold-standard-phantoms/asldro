""" Add noise filter """
import numpy as np

from asldro.containers.image import BaseImageContainer
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError


class AddNoiseFilter(BaseFilter):
    """
    A filter that simulates adds random noise to an input image.
    Inputs:
        -'image', type = BaseImageContainer: An input image which noise will be added to
        -'snr', type = float: the desired signal-to-noise ratio
        -'reference_image', type = BaseImageContainer: The reference image (optional)
    The reference image is used to calculate the amplitude of the random noise
    to add to the image.  If no reference image is supplied, the input image
    will be used.
    The noise is added pseudo-randomly based on the state of numpy.random. This should be
    appropriately controlled prior to running the filter
    Output:
        'image', type = BaseImageContainer: The input image with noise added.
    """

    def __init__(self):
        super().__init__(name="add noise")

    def _run(self):
        """ Calculate the noise amplitude, adds input image, and
        return the result"""

        input_image: BaseImageContainer = self.inputs["image"]
        snr: float = self.inputs["snr"]

        # If present load the reference image, if not
        # copy the input_image
        if "reference_image" in self.inputs:
            reference_image: BaseImageContainer = self.inputs["reference_image"]
        else:
            reference_image: BaseImageContainer = input_image

        # Calculate the noise amplitude (i.e. its standard deviation)
        noise_amplitude = (
            np.sqrt(reference_image.image.size)
            * np.mean(reference_image.image[reference_image.image.nonzero()])
            / (snr)
        )

        # Make an image container for the image with noise
        image_with_noise: BaseImageContainer = input_image.clone()

        # Create noise arrays with mean=0, sd=noise_amplitude, and same dimensions
        # as the input image.
        if input_image.image.dtype in [np.complex128, np.complex64]:
            # Data are complex, create the real and imaginary components separately
            image_with_noise.image = (
                np.real(input_image.image)
                + np.random.normal(0, noise_amplitude, input_image.shape)
            ) + 1j * (
                np.imag(input_image.image)
                + np.random.normal(0, noise_amplitude, input_image.shape)
            )
        else:
            # Data are not complex
            image_with_noise.image = np.imag(input_image.image) + np.random.normal(
                0, noise_amplitude, input_image.shape
            )

        self.outputs["image"] = image_with_noise

    def _validate_inputs(self):
        """
        'image' must be derived from BaseImageContainer
        'snr' must be a float
        'reference_image' if present must be derived from BaseImageContainer
        """
        if "image" not in self.inputs:
            raise FilterInputValidationError(f"{self} does not have defined `image`")

        if "snr" not in self.inputs:
            raise FilterInputValidationError(f"{self} does not have defined `snr`")

        input_image = self.inputs["image"]
        if not isinstance(input_image, BaseImageContainer):
            raise FilterInputValidationError(
                f"Input 'image' is not a BaseImageContainer (is {type(input_image)})"
            )

        input_snr = self.inputs["snr"]
        if not isinstance(input_snr, float):
            raise FilterInputValidationError(
                f"Input 'snr' is not a float (is {type(input_snr)})"
            )

        if "reference_image" in self.inputs:
            input_reference_image = self.inputs["reference_image"]
            if not isinstance(input_reference_image, BaseImageContainer):
                raise FilterInputValidationError(
                    f"Input 'reference_image' is not a BaseImageContainer"
                    f"(is {type(input_reference_image)})"
                )
