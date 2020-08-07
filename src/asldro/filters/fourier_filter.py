""" Fourier Transform filter """
import numpy as np

from asldro.containers.image import (
    NumpyImageContainer,
    NiftiImageContainer,
    SPATIAL_DOMAIN,
    INVERSE_DOMAIN,
)
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

        if isinstance(self.inputs["image"], NumpyImageContainer):
            image_container: NumpyImageContainer = self.inputs["image"]
            self.outputs["image"] = NumpyImageContainer(
                image=np.fft.fftn(image_container.image),
                affine=image_container.affine,
                space_units=image_container.space_units,
                time_units=image_container.time_units,
                voxel_size=image_container.voxel_size_mm,
                time_step=image_container.time_step_seconds,
                data_domain=INVERSE_DOMAIN,
            )
        elif isinstance(self.inputs["image"], NiftiImageContainer):
            image_container: NiftiImageContainer = self.inputs["image"]
            nifti_image_type = image_container.nifti_type
            self.outputs["image"] = NiftiImageContainer(
                nifti_img=nifti_image_type(
                    dataobj=np.fft.fftn(image_container.image),
                    affine=image_container.affine,
                    header=image_container.header,
                ),
                data_domain=INVERSE_DOMAIN,
            )

    def _validate_inputs(self):
        """"  Input must be a NumpyImageContainer or NiftiImageContainer,
        and data_domain should be SPATIAL_DOMAIN"""
        input_value = self.inputs["image"]
        if not isinstance(input_value, (NumpyImageContainer, NiftiImageContainer)):
            raise FilterInputValidationError(
                f"Input image is not a NumpyImageContainer or NiftiImageContainer (is {type(input_value)})"
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
        if isinstance(self.inputs["image"], NumpyImageContainer):
            image_container: NumpyImageContainer = self.inputs["image"]
            self.outputs["image"] = NumpyImageContainer(
                image=np.fft.ifftn(image_container.image),
                affine=image_container.affine,
                space_units=image_container.space_units,
                time_units=image_container.time_units,
                voxel_size=image_container.voxel_size_mm,
                time_step=image_container.time_step_seconds,
                data_domain=SPATIAL_DOMAIN,
            )
        elif isinstance(self.inputs["image"], NiftiImageContainer):
            image_container: NiftiImageContainer = self.inputs["image"]
            nifti_image_type = image_container.nifti_type
            self.outputs["image"] = NiftiImageContainer(
                nifti_img=nifti_image_type(
                    dataobj=np.fft.ifftn(image_container.image),
                    affine=image_container.affine,
                    header=image_container.header,
                ),
                data_domain=SPATIAL_DOMAIN,
            )

    def _validate_inputs(self):
        """" Input must be a NumpyImageContainer or NiftiImageContainer,
        and data_domain should be INVERSE_DOMAIN"""
        input_value = self.inputs["image"]
        if not isinstance(input_value, (NumpyImageContainer, NiftiImageContainer)):
            raise FilterInputValidationError(
                f"Input image is not a NumpyImageContainer or NiftiImageContainer (is {type(input_value)})"
            )
        if input_value.data_domain != INVERSE_DOMAIN:
            raise FilterInputValidationError(
                f"Input image is not in the spatial domain (is {input_value.data_domain}"
            )
