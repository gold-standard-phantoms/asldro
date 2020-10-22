""" Resample Filter """

import numpy as np
from copy import deepcopy
from nilearn.image import resample_img
import nibabel as nib
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.containers.image import (
    BaseImageContainer,
    NumpyImageContainer,
    NiftiImageContainer,
    UNITS_MILLIMETERS,
    UNITS_SECONDS,
)
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
        is_input_image_numpy_image_container = isinstance(
            self.inputs[self.KEY_IMAGE], NumpyImageContainer
        )
        if is_input_image_numpy_image_container:
            image = NiftiImageContainer(
                nib.Nifti2Image(
                    dataobj=self.inputs[self.KEY_IMAGE].image,
                    affine=self.inputs[self.KEY_IMAGE].affine,
                )
            )
        else:
            image: NiftiImageContainer = self.inputs[self.KEY_IMAGE]

        affine = self.inputs[self.KEY_AFFINE]
        shape = self.inputs[self.KEY_SHAPE]

        resampled_nifti = resample_img(
            image._nifti_image, target_affine=affine, target_shape=shape
        )

        if is_input_image_numpy_image_container:
            self.outputs[self.KEY_IMAGE] = NumpyImageContainer(
                image=np.asanyarray(resampled_nifti.dataobj),
                affine=resampled_nifti.affine,
                data_domain=self.inputs[self.KEY_IMAGE].data_domain,
                image_type=self.inputs[self.KEY_IMAGE].image_type,
                space_units=UNITS_MILLIMETERS,
                time_units=UNITS_SECONDS,
                voxel_size=self.inputs[self.KEY_IMAGE].voxel_size_mm,
                time_step=self.inputs[self.KEY_IMAGE].time_step_seconds,
            )
        else:
            self.outputs[self.KEY_IMAGE] = NiftiImageContainer(
                nifti_img=resampled_nifti,
                data_domain=self.inputs[self.KEY_IMAGE].data_domain,
                image_type=self.inputs[self.KEY_IMAGE].image_type,
            )

        # copy across metadata
        self.outputs[self.KEY_IMAGE].metadata = deepcopy(
            self.inputs[self.KEY_IMAGE].metadata
        )

        self.outputs[self.KEY_IMAGE].metadata["voxel_size"] = tuple(
            nib.affines.voxel_sizes(resampled_nifti.affine)
        )

    def _validate_inputs(self):
        """ Checks that the inputs meet their validation criteria

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
