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
    greater_than_equal_to_validator,
)


class AddComplexNoiseFilter(FilterBlock):
    """A filter that adds normally distributed random noise
    to the real and imaginary parts of the fourier transform of the input image.

    **Inputs**

    Input parameters are all keyword arguments for the :class:`AddComplexNoiseFilter.add_inputs()`
    member function.  They are also accessible via class constants, for example
    :class:`AddComplexNoiseFilter.KEY_SNR`.

    :param 'image': An input image which noise will be added to. Can be either scalar or complex.
        If it is complex, normally distributed random noise will be added to both real
        and imaginary parts.
    :type 'image': BaseImageContainer
    :param 'snr': the desired signal-to-noise ratio (>= 0). A value of zero means that no noise
        is added to the input image.
    :type 'snr': float
    :param 'reference_image': The reference image that is used to calculate the amplitude of
        the random noise to add to `'image'`. The shape of this must match the shape of `'image'`.
        If this is not supplied then `'image'` will be used for calculating the noise amplitude.
    :type 'reference_image': BaseImageContainer, optional

    **Outputs**

    :param 'image': The input image with complex noise added.
    :type 'image': BaseImageContainer


    The noise is added pseudo-randomly based on the state of numpy.random. This should be
    appropriately controlled prior to running the filter

    """

    KEY_IMAGE = AddNoiseFilter.KEY_IMAGE
    KEY_REF_IMAGE = AddNoiseFilter.KEY_REF_IMAGE
    KEY_SNR = AddNoiseFilter.KEY_SNR

    def __init__(self):
        super().__init__(name="Add Complex Noise")

    def _create_filter_block(self):
        """Fourier transforms the input and reference images, calculates
        the noise amplitude, adds this to the FT of the input image, then
        inverse fourier transforms to obtain the output image"""

        input_image: BaseImageContainer = self.inputs[self.KEY_IMAGE]

        # if self.inputs["snr"] == 0  then the input image should just be
        # passed through to the output.
        # Because this is a filter block it needs to be done by another filter - fortunately
        # the AddNoiseFilter does just this when AddNoiseFilter.inputs["snr"] = 0
        if self.inputs[self.KEY_SNR] == 0:

            add_noise_filter = AddNoiseFilter()
            add_noise_filter.add_input(self.KEY_IMAGE, input_image)
            add_noise_filter.add_input(self.KEY_SNR, self.inputs[self.KEY_SNR])
            return add_noise_filter
        # snr is greater than 0 so run the filter block normally
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
        'snr' must be a float and >= 0
        'reference_image' if present must be derived from BaseImageContainer
        """
        input_validator = ParameterValidator(
            parameters={
                self.KEY_IMAGE: Parameter(
                    validators=isinstance_validator(BaseImageContainer)
                ),
                self.KEY_SNR: Parameter(
                    validators=[
                        isinstance_validator(float),
                        greater_than_equal_to_validator(0),
                    ]
                ),
                self.KEY_REF_IMAGE: Parameter(
                    validators=isinstance_validator(BaseImageContainer), optional=True
                ),
            }
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)
