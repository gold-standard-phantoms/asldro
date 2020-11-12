""" Fourier Transform filter """
import numpy as np

from asldro.containers.image import (
    BaseImageContainer,
    COMPLEX_IMAGE_TYPE,
    SPATIAL_DOMAIN,
    INVERSE_DOMAIN,
)
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.validators.parameters import (
    ParameterValidator,
    Parameter,
    isinstance_validator,
)


class FftFilter(BaseFilter):
    """A filter for performing a n-dimensional fast fourier transform of input.
    Input is either a NumpyImageContainer or NiftiImageContainer.
    Output is a complex numpy array of the discrete fourier transform named 'kdata'"""

    # Key constants
    KEY_IMAGE = "image"

    def __init__(self):
        super().__init__(name="FFT")

    def _run(self):
        """performs a n-dimensional fast fourier transform on the input Image Container
        and creates an 'output' with the result in an Image Container of the equivalent type
        to the input.  The input image must have data_domain == SPATIAL_DOMAIN, and
        the output image (k-space data) will have data_domain == INVERSE_DOMAIN
        """
        image_container: BaseImageContainer = self.inputs[self.KEY_IMAGE].clone()
        image_container.image = np.fft.fftn(image_container.image)
        image_container.data_domain = INVERSE_DOMAIN
        image_container.image_type = COMPLEX_IMAGE_TYPE
        self.outputs[self.KEY_IMAGE] = image_container

    def _validate_inputs(self):
        """Input must be derived from BaseImageContainer
        and data_domain should be SPATIAL_DOMAIN"""

        input_validator = ParameterValidator(
            parameters={
                self.KEY_IMAGE: Parameter(
                    validators=isinstance_validator(BaseImageContainer)
                )
            }
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)

        if self.inputs[self.KEY_IMAGE].data_domain != SPATIAL_DOMAIN:
            raise FilterInputValidationError(
                f"Input image is not in the spatial domain "
                f"(is {self.inputs[self.KEY_IMAGE].data_domain}"
            )


class IfftFilter(BaseFilter):
    """A filter for performing a n-dimensional inverse fast fourier transform of input.
    Input is a numpy array named 'kdata'.
    Output is a complex numpy array of the inverse discrete fourier transform named 'image'"""

    # Key constants
    KEY_IMAGE = "image"

    def __init__(self):
        super().__init__(name="IFFT")

    def _run(self):
        """performs a n-dimensional inverse fast fourier transform on the input Image Container
        and creates an 'output' with the result in an Image Container of the equivalent type
        to the input.  The input image (k-space data) must have data_domain == INVERSE_DOMAIN, and
        the output image will have data_domain == SPATIAL_DOMAIN
        """
        image_container: BaseImageContainer = self.inputs[self.KEY_IMAGE].clone()
        image_container.image = np.fft.ifftn(image_container.image)
        image_container.data_domain = SPATIAL_DOMAIN
        image_container.image_type = COMPLEX_IMAGE_TYPE
        self.outputs[self.KEY_IMAGE] = image_container

    def _validate_inputs(self):
        """Input must be derived from BaseImageContainer
        and data_domain should be INVERSE_DOMAIN"""
        input_validator = ParameterValidator(
            parameters={
                self.KEY_IMAGE: Parameter(
                    validators=isinstance_validator(BaseImageContainer)
                )
            }
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)

        if self.inputs[self.KEY_IMAGE].data_domain != INVERSE_DOMAIN:
            raise FilterInputValidationError(
                f"Input image is not in the inverse domain "
                f"(is {self.inputs[self.KEY_IMAGE].data_domain}"
            )
