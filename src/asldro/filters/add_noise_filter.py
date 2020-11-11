""" Add noise filter """
import logging
import numpy as np

from asldro.containers.image import BaseImageContainer, SPATIAL_DOMAIN, INVERSE_DOMAIN
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.validators.parameters import (
    ParameterValidator,
    Parameter,
    isinstance_validator,
    greater_than_equal_to_validator,
)

logger = logging.getLogger(__name__)


class AddNoiseFilter(BaseFilter):
    """
    A filter that adds normally distributed random noise to an input image.

    **Inputs**

    Input parameters are all keyword arguments for the :class:`AddNoiseFilter.add_inputs()`
    member function.  They are also accessible via class constants, for example
    :class:`AddNoiseFilter.KEY_SNR`.

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

    :param 'image': The input image with noise added.
    :type 'image': BaseImageContainer

    `'reference_image'` can be in a different data domain to the `'image'`.  For example, `'image'`
    might be in the inverse domain (i.e. fourier transformed) whereas `'reference_image'` is in
    the spatial domain.
    Where data domains differ the following scaling is applied to the noise amplitude:
        * `'image'` is `SPATIAL_DOMAIN` and 'reference_image' is `INVERSE_DOMAIN`: 1/N
        * `'image'` is `INVERSE_DOMAIN` and 'reference_image' is `SPATIAL_DOMAIN`: N
    Where N is `reference_image.image.size`

    The noise is added pseudo-randomly based on the state of numpy.random. This should be
    appropriately controlled prior to running the filter

    Note that the actual SNR (as calculated using "A comparison of two methods for
    measuring the signal to noise ratio on MR images", PMB, vol 44, no. 12, pp.N261-N264 (1999))
    will not match the desired SNR under the following circumstances:
        * `'image'` is `SPATIAL_DOMAIN` and `'reference_image'` is `INVERSE_DOMAIN`
        * `'image'` is `INVERSE_DOMAIN` and `'reference_image'` is `SPATIAL_DOMAIN`
    In the second case, performing an inverse fourier transform on the output image with
    noise results in a spatial domain image where the calculated SNR matches the desired SNR.
    This is how the :class:`AddNoiseFilter` is used within the :class:`AddComplexNoiseFilter`
    """

    # Key constants
    KEY_IMAGE = "image"
    KEY_SNR = "snr"
    KEY_REF_IMAGE = "reference_image"

    def __init__(self):
        super().__init__(name="Add Noise")

    def _run(self):
        """Calculate the noise amplitude, adds input image, and
        return the result"""

        input_image: BaseImageContainer = self.inputs[self.KEY_IMAGE]
        snr: float = self.inputs[self.KEY_SNR]

        # if snr is 0 then no noise needs to be added to the image
        if snr == 0:
            self.outputs[self.KEY_IMAGE] = input_image
        else:

            # If present load the reference image, if not
            # copy the input_image
            if self.KEY_REF_IMAGE in self.inputs:
                reference_image: BaseImageContainer = self.inputs[self.KEY_REF_IMAGE]
            else:
                reference_image: BaseImageContainer = input_image

            noise_amplitude_scaling: float = 1.0  # default if domains match
            # Otherwise correct for differences in scaling due to fourier transform
            logger.debug("input image domain is %s", input_image.data_domain)
            logger.debug("reference_image domain is %s", reference_image.data_domain)
            if (
                input_image.data_domain == SPATIAL_DOMAIN
                and reference_image.data_domain == INVERSE_DOMAIN
            ):
                noise_amplitude_scaling = 1.0 / np.sqrt(reference_image.image.size)
            if (
                input_image.data_domain == INVERSE_DOMAIN
                and reference_image.data_domain == SPATIAL_DOMAIN
            ):
                noise_amplitude_scaling = np.sqrt(reference_image.image.size)

            # Calculate the noise amplitude (i.e. its standard deviation) using the non-zero voxels
            # in the magnitude of the reference image (in case it is complex)
            logger.debug("Noise amplitude scaling: %s", noise_amplitude_scaling)
            noise_amplitude = (
                noise_amplitude_scaling
                * np.mean(
                    np.abs(reference_image.image[reference_image.image.nonzero()])
                )
                / (snr)
            )

            logger.debug("Noise amplitude: %s", noise_amplitude)

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
                image_with_noise.image = input_image.image + np.random.normal(
                    0, noise_amplitude, input_image.shape
                )

            self.outputs[self.KEY_IMAGE] = image_with_noise

    def _validate_inputs(self):
        """
        'image' must be derived from BaseImageContainer
        'snr' must be a positive float
        'reference_image' if present must be derived from BaseImageContainer.
        image.shape and reference_image.shape must be equal
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

        # If 'reference_image' is supplied, check that its dimensions match 'image'
        if self.KEY_REF_IMAGE in self.inputs:
            input_reference_image = self.inputs[self.KEY_REF_IMAGE]
            input_image = self.inputs[self.KEY_IMAGE]
            if not isinstance(input_reference_image, BaseImageContainer):
                raise FilterInputValidationError(
                    f"Input 'reference_image' is not a BaseImageContainer"
                    f"(is {type(input_reference_image)})"
                )
            if not input_image.shape == input_reference_image.shape:
                raise FilterInputValidationError(
                    f"Shape of inputs 'image' and 'reference_image' are not equal"
                    f"Shape of 'image' is {input_image.shape}"
                    f"Shape of 'reference_image' is {input_reference_image.shape}"
                )
