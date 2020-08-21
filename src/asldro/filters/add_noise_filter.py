""" Add noise filter """
import numpy as np

from asldro.containers.image import BaseImageContainer, SPATIAL_DOMAIN, INVERSE_DOMAIN
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError


class AddNoiseFilter(BaseFilter):
    """
    A filter that adds normally distributed random noise to an input image.

    Inputs:
        'image' (BaseImageContainer): An input image which noise will be added to
        'snr' (float): the desired signal-to-noise ratio
        'reference_image' (BaseImageContainer): The reference image (optional)

    'image' can be either scalar or complex.  If it is complex, normally distributed random
    noise will be added to both real and imaginary parts.

    'reference image' is used to calculate the amplitude of the random noise
    to add to 'image'. The shape of both of these must match.

    'reference_image' can be in a different data domain to the 'image'.  For example, 'image'
    might be in the inverse domain (i.e. fourier transformed) whereas 'reference_image' is in
    the spatial domain. Where data domains differ the following scaling is applied to the noise amplitude:
        'image' is SPATIAL_DOMAIN and 'reference_image' is INVERSE_DOMAIN: 1/N
        'image' is INVERSE_DOMAIN and 'reference_image' is SPATIAL_DOMAIN: N
    Where N is 'reference_image.image.size'

    If 'reference_image' is not supplied, 'image' will be used to calculate the noise amplitude.

    The noise is added pseudo-randomly based on the state of numpy.random. This should be
    appropriately controlled prior to running the filter

    Output:
        'image' (BaseImageContainer): The input image with noise added.

    Note that the actual SNR (as calculated using "A comparison of two methods for measuring the signal to
    noise ratio on MR images", PMB, vol 44, no. 12, pp.N261-N264 (1999)) will not match the desired SNR
    under the following circumstances:
        'image' is SPATIAL_DOMAIN and 'reference_image' is INVERSE_DOMAIN
        'image' is INVERSE_DOMAIN and 'reference_image' is SPATIAL_DOMAIN
    In the second case, performing an inverse fourier transform on the output image with noise results in 
    a spatial domain image where the calculated SNR matches the desired SNR.  This is how the AddNoiseFilter
    is used within the AddComplexNoiseFilter
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

        noise_amplitude_scaling: float = 1.0  # default if domains match
        # Otherwise correct for differences in scaling due to fourier transform
        # print(f"input image domain is {input_image.data_domain}")
        # print(f"reference_image domain is {reference_image.data_domain}")
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
        # print(f"noise amplitude scaling {noise_amplitude_scaling}")
        noise_amplitude = (
            noise_amplitude_scaling
            * np.mean(np.abs(reference_image.image[reference_image.image.nonzero()]))
            / (snr)
        )

        print(f"noise amplitude {noise_amplitude}")

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

        self.outputs["image"] = image_with_noise

    def _validate_inputs(self):
        """
        'image' must be derived from BaseImageContainer
        'snr' must be a float
        'reference_image' if present must be derived from BaseImageContainer. image.shape and reference_image.shape must be equal
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
            if not input_image.shape == input_reference_image.shape:
                raise FilterInputValidationError(
                    f"Shape of inputs 'image' and 'reference_image' are not equal"
                    f"Shape of 'image' is {input_image.shape}"
                    f"Shape of 'reference_image' is {input_reference_image.shape}"
                )
