""" Add complex noise filter """
import numpy as np

from asldro.containers.image import BaseImageContainer
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.filters.fourier_filter import FftFilter, IfftFilter


class AddComplexNoiseFilter(BaseFilter):
    """ A filter that simulates adds random noise to the real and imaginary
    channels of the fourier transform of the input image.\\
    Inputs:\\
        -'image', type = BaseImageContainer: An input image which noise will be added to\\
        -'snr', type = float: the desired signal-to-noise ratio\\
        -'reference_image', type = BaseImageContainer: The reference image (optional)\\
    \\
    The reference image is used to calculate the amplitude of the random noise
    to add to the image.  If no reference image is supplied, the input image
    will be used.\\
    The noise is added pseud-randomly based on the state of numpy.random. This should be\\
    appropriately controlled prior to running the filter\\
    Output:\\
        'image', type = BaseImageContainer: The input image with complex noise added. 
    """

    def __init__(self):
        super().__init__(name="add complex noise")

    def _run(self):
        """ Fourier transforms the input and reference images, calculates
        the noise amplitude, adds this to the FT of the input image, then
        inverse fourier transforms to obtain the output image """

        input_image: BaseImageContainer = self.inputs["image"]
        image_fft_filter = FftFilter()
        image_fft_filter.add_input("image", input_image)
        image_fft_filter.run()
        ft_image: BaseImageContainer = image_fft_filter.outputs["image"]

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

        k_space_image = ft_image.image

        # Create noise arrays with mean=0, sd=noise_amplitude, and same dimensions
        # as k_space_image
        noise_array_real = np.random.normal(0, noise_amplitude, k_space_image.shape)
        noise_array_imag = np.random.normal(0, noise_amplitude, k_space_image.shape)

        # Add the noise arrays to the k space image
        k_space_with_noise = (np.real(k_space_image) + noise_array_real) + 1j * (
            np.imag(k_space_image) + noise_array_imag
        )

        # Make an image container for the k space image with noise
        ft_image_with_noise: BaseImageContainer = ft_image.clone()
        ft_image_with_noise.image = k_space_with_noise

        # Inverse Fourier Transform and set the output
        ifft_filter = IfftFilter()
        ifft_filter.add_input("image", ft_image_with_noise)
        ifft_filter.run()
        self.outputs["image"] = ifft_filter.outputs["image"]

    def _validate_inputs(self):
        """
        'image' must be derived from BaseImageContainer
        'snr' must be a float
        'reference_image' if present must be derived from BaseImageContainer
        'seed' if present must be an integer
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
