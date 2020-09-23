""" Add complex noise filter block """
from asldro.containers.image import BaseImageContainer
from asldro.filters.basefilter import FilterInputValidationError
from asldro.filters.filter_block import FilterBlock
from asldro.filters.fourier_filter import FftFilter, IfftFilter
from asldro.filters.add_noise_filter import AddNoiseFilter
from asldro.validators.parameters import (
    ParameterValidator,
    Parameter,
    isinstance_validator,
    greater_than_validator,
)


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

    KEY_IMAGE = AddNoiseFilter.KEY_IMAGE
    KEY_REF_IMAGE = AddNoiseFilter.KEY_REF_IMAGE
    KEY_SNR = AddNoiseFilter.KEY_SNR

    def __init__(self):
        super().__init__(name="add complex noise")

    def _create_filter_block(self):
        """ Fourier transforms the input and reference images, calculates
        the noise amplitude, adds this to the FT of the input image, then
        inverse fourier transforms to obtain the output image """

        input_image: BaseImageContainer = self.inputs[self.KEY_IMAGE]
        # Fourier transform the input image
        image_fft_filter = FftFilter()
        image_fft_filter.add_input(self.KEY_IMAGE, input_image)

        # Create the noise filter
        add_noise_filter = AddNoiseFilter()
        add_noise_filter.add_parent_filter(image_fft_filter)
        add_noise_filter.add_input(self.KEY_SNR, self.inputs[self.KEY_SNR])

        # If present load the reference image, if not, copy the input_image
        if self.KEY_REF_IMAGE in self.inputs:
            add_noise_filter.add_input(
                self.KEY_REF_IMAGE, self.inputs[self.KEY_REF_IMAGE]
            )
        else:
            add_noise_filter.add_input(self.KEY_REF_IMAGE, self.inputs[self.KEY_IMAGE])

        # Inverse Fourier Transform and set the output
        ifft_filter = IfftFilter()
        ifft_filter.add_parent_filter(add_noise_filter)
        return ifft_filter

    def _validate_inputs(self):
        """
        'image' must be derived from BaseImageContainer.
        'snr' must be a positive float
        'reference_image' if present must be derived from BaseImageContainer
        """
        input_validator = ParameterValidator(
            parameters={
                self.KEY_IMAGE: Parameter(
                    validators=isinstance_validator(BaseImageContainer)
                ),
                self.KEY_SNR: Parameter(
                    validators=[isinstance_validator(float), greater_than_validator(0)]
                ),
                self.KEY_REF_IMAGE: Parameter(
                    validators=isinstance_validator(BaseImageContainer), optional=True
                ),
            }
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)
