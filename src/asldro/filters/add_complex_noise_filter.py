""" Add complex noise filter block """
from asldro.containers.image import BaseImageContainer
from asldro.filters.basefilter import FilterInputValidationError
from asldro.filters.filter_block import FilterBlock
from asldro.filters.fourier_filter import FftFilter, IfftFilter
from asldro.filters.add_noise_filter import AddNoiseFilter


class AddComplexNoiseFilter(FilterBlock):
    """ A filter that adds normally distributed random noise
    to the real and imaginary parts of the fourier transform of the input image.

    Inputs:
        'image' (BaseImageContainer): An input image which noise will be added to
        'snr' (float): the desired signal-to-noise ratio
        'reference_image' (BaseImageContainer): The reference image (optional)

    The reference image is used to calculate the amplitude of the random noise
    to add to the image.  If no reference image is supplied, the input image
    will be used.

    The noise is added pseudo-randomly based on the state of numpy.random. This should be
    appropriately controlled prior to running the filter

    Outputs:
        'image' (BaseImageContainer): The input image with noise added.
    """

    def __init__(self):
        super().__init__(name="add complex noise")

    def _create_filter_block(self):
        """ Fourier transforms the input and reference images, calculates
        the noise amplitude, adds this to the FT of the input image, then
        inverse fourier transforms to obtain the output image """

        input_image: BaseImageContainer = self.inputs["image"]
        # Fourier transform the input image
        image_fft_filter = FftFilter()
        image_fft_filter.add_input("image", input_image)

        # Create the noise filter
        add_noise_filter = AddNoiseFilter()
        add_noise_filter.add_parent_filter(image_fft_filter)
        add_noise_filter.add_input("snr", self.inputs["snr"])

        # If present load the reference image, if not, copy the input_image
        if "reference_image" in self.inputs:
            add_noise_filter.add_input(
                "reference_image", self.inputs["reference_image"]
            )
        else:
            add_noise_filter.add_input("reference_image", self.inputs["image"])

        # Inverse Fourier Transform and set the output
        ifft_filter = IfftFilter()
        ifft_filter.add_parent_filter(add_noise_filter)
        return ifft_filter

    def _validate_inputs(self):
        """
        'image' must be derived from BaseImageContainer. The image type should not
        be complex.
        'snr' must be a float
        'reference_image' if present must be derived from BaseImageContainer
        """
        if "image" not in self.inputs:
            raise FilterInputValidationError(f"{self} does not have defined `image`")

        if "snr" not in self.inputs:
            raise FilterInputValidationError(f"{self} does not have defined `snr`")

        input_image: BaseImageContainer = self.inputs["image"]
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
