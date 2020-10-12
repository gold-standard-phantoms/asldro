"""Transform resample image filter"""

import numpy as np
from asldro.filters.basefilter import FilterInputValidationError
from asldro.filters.filter_block import BaseFilter
from asldro.filters.affine_matrix_filter import AffineMatrixFilter
from asldro.filters.resample_filter import ResampleFilter
from asldro.containers.image import (
    BaseImageContainer,
    NumpyImageContainer,
    NiftiImageContainer,
)
from asldro.validators.parameters import (
    ParameterValidator,
    Parameter,
    isinstance_validator,
    for_each_validator,
    range_inclusive_validator,
    greater_than_validator,
)
from asldro.utils.resampling import transform_resample_affine


class TransformResampleImageFilter(BaseFilter):

    KEY_TARGET_SHAPE = "target_shape"
    KEY_ROTATION_ORIGIN = "rotation_origin"
    KEY_ROTATION = "rotation"
    KEY_TRANSLATION = "translation"
    KEY_IMAGE = "image"

    def __init__(self):
        super().__init__(name="Transform and Resample Object")

    def _run(self):
        r""" Transforms the object in world-space, then creates a resampled image at 
        with the specified acquisition shape 
        
    
        """

        input_image: BaseImageContainer = self.inputs[self.KEY_IMAGE]
        translation = self.inputs[self.KEY_TRANSLATION]
        rotation = self.inputs[self.KEY_ROTATION]
        rotation_origin = self.inputs[self.KEY_ROTATION_ORIGIN]
        target_shape = self.inputs[self.KEY_TARGET_SHAPE]

        (target_affine, sampled_image_affine) = transform_resample_affine(
            input_image, translation, rotation, rotation_origin, target_shape
        )

        resample_filter = ResampleFilter()
        resample_filter.add_input(ResampleFilter.KEY_IMAGE, input_image)
        resample_filter.add_input(ResampleFilter.KEY_AFFINE, target_affine)
        resample_filter.add_input(ResampleFilter.KEY_SHAPE, target_shape)
        resample_filter.run()

        self.outputs[self.KEY_IMAGE] = resample_filter.outputs[ResampleFilter.KEY_IMAGE]
        if isinstance(self.inputs[self.KEY_IMAGE], NumpyImageContainer):
            self.outputs[self.KEY_IMAGE]._affine = sampled_image_affine
        else:
            self.outputs[self.KEY_IMAGE]._nifti_image.set_sform(sampled_image_affine)

    def _validate_inputs(self):
        """ Checks that the inputs meet their validation criteria
        `'object'` must be derived from BaseImageContainer
        `'target_shape'` (optional)must be a Tuple of ints of length 3, values > 0
        `'rotation'` (optional) must be a Tuple of floats of length 3, each value -180 to 180
        inclusive, default (optional) = (0.0, 0.0, 0.0)
        `'rotation_origin'` (optional) must be a Tuple of floats of length 3,
        default = (0.0, 0.0, 0.0)
        `'translation'` (optional) must be a Tuple of floats of length 3, default = (0.0, 0.0, 0.0)
        """

        input_validator = ParameterValidator(
            parameters={
                self.KEY_IMAGE: Parameter(
                    validators=isinstance_validator(BaseImageContainer)
                ),
                self.KEY_ROTATION: Parameter(
                    validators=[
                        isinstance_validator(tuple),
                        for_each_validator(isinstance_validator(float)),
                        for_each_validator(range_inclusive_validator(-180, 180)),
                    ],
                    optional=True,
                    default_value=(0.0, 0.0, 0.0),
                ),
                self.KEY_ROTATION_ORIGIN: Parameter(
                    validators=[
                        isinstance_validator(tuple),
                        for_each_validator(isinstance_validator(float)),
                    ],
                    optional=True,
                    default_value=(0.0, 0.0, 0.0),
                ),
                self.KEY_TRANSLATION: Parameter(
                    validators=[
                        isinstance_validator(tuple),
                        for_each_validator(isinstance_validator(float)),
                    ],
                    optional=True,
                    default_value=(0.0, 0.0, 0.0),
                ),
                self.KEY_TARGET_SHAPE: Parameter(
                    validators=[
                        isinstance_validator(tuple),
                        for_each_validator(isinstance_validator(int)),
                        for_each_validator(greater_than_validator(0)),
                    ],
                    optional=True,
                    default_value=(9999, 9999, 9999),
                ),
            }
        )

        # validate, returning a dictionary which also includes default parameters
        new_params = input_validator.validate(
            self.inputs, error_type=FilterInputValidationError
        )

        # Further validation that can't be handled by the parameter validator

        if new_params[self.KEY_TARGET_SHAPE] == (9999, 9999, 9999):
            new_params[self.KEY_TARGET_SHAPE] = self.inputs[self.KEY_IMAGE].shape

        # Check that the tuple self.KEY_ROTATION's length is 3
        if len(new_params[self.KEY_ROTATION]) != 3:
            raise FilterInputValidationError

        # Check that the tuple self.KEY_ROTATION_ORIGIN's length is 3
        if len(new_params[self.KEY_ROTATION_ORIGIN]) != 3:
            raise FilterInputValidationError

        # Check that the tuple self.KEY_TRANSLATION's length is 3
        if len(new_params[self.KEY_TRANSLATION]) != 3:
            raise FilterInputValidationError

        # Check that the tuple self.KEY_SCALE's length is 3
        if len(new_params[self.KEY_TARGET_SHAPE]) != 3:
            raise FilterInputValidationError

        # merge the updated parameters from the output with the input parameters
        self.inputs = {**self._i, **new_params}

