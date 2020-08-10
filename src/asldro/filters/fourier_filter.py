""" Fourier Transform filter """
import numpy as np

from asldro.containers.image import BaseImageContainer, SPATIAL_DOMAIN, INVERSE_DOMAIN
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError


class FftFilter(BaseFilter):
    """ A filter for performing a n-dimensional fast fourier transform of input.
    Input is either a NumpyImageContainer or NiftiImageContainer.
    Output is a complex numpy array of the discrete fourier transform named 'kdata'"""

    def __init__(self):
        super().__init__(name="fft")

    def _run(self):
        """ performs a n-dimensional fast fourier transform on the input Image Container
        and creates an 'output' with the result in an Image Container of the equivalent type
        to the input.  The input image must have data_domain == SPATIAL_DOMAIN, and
        the output image (k-space data) will have data_domain == INVERSE_DOMAIN
        """
        image_container: BaseImageContainer = self.inputs["image"].clone()
        image_container.image = np.fft.fftn(image_container.image)
        image_container.data_domain = INVERSE_DOMAIN
        self.outputs["image"] = image_container

    def _validate_inputs(self):
        """" Input must be derived from BaseImageContainer
        and data_domain should be SPATIAL_DOMAIN"""
        if "image" not in self.inputs:
            raise FilterInputValidationError(f"{self} does not defined `image`")

        input_value = self.inputs["image"]
        if not isinstance(input_value, BaseImageContainer):
            raise FilterInputValidationError(
                f"Input image is not a BaseImageContainer (is {type(input_value)})"
            )
        if input_value.data_domain != SPATIAL_DOMAIN:
            raise FilterInputValidationError(
                f"Input image is not in the spatial domain (is {input_value.data_domain}"
            )


class IfftFilter(BaseFilter):
    """ A filter for performing a n-dimensional inverse fast fourier transform of input.
    Input is a numpy array named 'kdata'.
    Output is a complex numpy array of the inverse discrete fourier transform named 'image' """

    def __init__(self):
        super().__init__(name="ifft")

    def _run(self):
        """ performs a n-dimensional inverse fast fourier transform on the input Image Container
        and creates an 'output' with the result in an Image Container of the equivalent type
        to the input.  The input image (k-space data) must have data_domain == INVERSE_DOMAIN, and
        the output image will have data_domain == SPATIAL_DOMAIN
        """
        image_container: BaseImageContainer = self.inputs["image"].clone()
        image_container.image = np.fft.ifftn(image_container.image)
        image_container.data_domain = SPATIAL_DOMAIN
        self.outputs["image"] = image_container

    def _validate_inputs(self):
        """" Input must be derived from BaseImageContainer
        and data_domain should be INVERSE_DOMAIN"""
        if "image" not in self.inputs:
            raise FilterInputValidationError(f"{self} does not defined `image`")
        input_value = self.inputs["image"]
        if not isinstance(input_value, BaseImageContainer):
            raise FilterInputValidationError(
                f"Input image is not a BaseImageContainer (is {type(input_value)})"
            )
        if input_value.data_domain != INVERSE_DOMAIN:
            raise FilterInputValidationError(
                f"Input image is not in the inverse domain (is {input_value.data_domain}"
            )
