""" Resample Filter """

import numpy as np
from nilearn.image import resample_img
import nibabel as nib
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.containers.image import BaseImageContainer
from asldro.validators.parameters import (
    ParameterValidator,
    Parameter,
    isinstance_validator,
    for_each_validator,
)


class ResampleFilter(BaseFilter):
    r"""A filter that can resample an image based on a target shape and affine. Note that nilearn
    actually applies the inverse of the target affine.

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`ResampleFilter.add_inputs()`
    member function. They are also accessible via class constants, for example
    :class:`ResampleFilter.KEY_AFFINE`

    :param 'image': Image to resample
    :type 'image': BaseImageContainer
    :param 'affine': Image is resampled according to this 4x4 affine matrix
    :type 'affine': np.ndarray(4)
    :param 'shape': Image is resampled according to this new shape.
    :type 'shape': Tuple[int, int, int]

    **Outputs**
    Once run, the filter will populate the dictionary :class:`ResampleFilter.outputs` with
    the following entries:

    :param 'image': The input image, resampled in accordance with the input shape and affine.
    :type 'image': BaseImageContainer

    The metadata property of the :class:`ResampleFilter.outputs["image"]` is
    updated with the field ``voxel_size``, corresponding to the size of each voxel.

    """
    KEY_IMAGE = "image"
    KEY_AFFINE = "affine"
    KEY_SHAPE = "shape"

    def __init__(self):
        super().__init__(name="Resample image")

    def _run(self):
        # Clone the image and perform the resampling
        resampled_image = self.inputs[self.KEY_IMAGE].as_nifti().clone()
        resampled_image.nifti_image = resample_img(
            resampled_image.nifti_image,
            target_affine=self.inputs[self.KEY_AFFINE],
            target_shape=self.inputs[self.KEY_SHAPE],
        )
        self.outputs[self.KEY_IMAGE] = resampled_image
        self.outputs[self.KEY_IMAGE].metadata["voxel_size"] = list(
            nib.affines.voxel_sizes(resampled_image.nifti_image.affine)
        )

    def _validate_inputs(self):
        """Checks that the inputs meet their validation criteria

        `'image'` must be derived from a BaseImageContainer
        `'affine'` must be a numpy.ndarray of shape (4,4)
        `'shape'` must be a Tuple of integers of length 3

        """
        input_validator = ParameterValidator(
            parameters={
                self.KEY_IMAGE: Parameter(
                    validators=[isinstance_validator(BaseImageContainer)]
                ),
                self.KEY_AFFINE: Parameter(
                    validators=[isinstance_validator(np.ndarray)]
                ),
                self.KEY_SHAPE: Parameter(
                    validators=[
                        isinstance_validator(tuple),
                        for_each_validator(isinstance_validator(int)),
                    ]
                ),
            }
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)

        # Further validation that can't be handled bby the parameter validator
        # check that ResampleFilter.KEY_AFFINE is of size 4x4
        if self.inputs[self.KEY_AFFINE].shape != (4, 4):
            raise FilterInputValidationError

        # Check that the tuple ResampleFilter.KEY_SHAPE has length 3
        if len(self.inputs[self.KEY_SHAPE]) != 3:
            raise FilterInputValidationError
