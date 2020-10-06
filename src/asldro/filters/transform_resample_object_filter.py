"""Transform resample object filter"""

import numpy as np
from typing import Tuple, List
from asldro.filters.basefilter import FilterInputValidationError
from asldro.filters.filter_block import FilterBlock
from asldro.filters.affine_matrix_filter import AffineMatrixFilter
from asldro.filters.resample_filter import ResampleFilter
from asldro.containers.image import BaseImageContainer
from asldro.validators.parameters import (
    ParameterValidator,
    Parameter,
    isinstance_validator,
    for_each_validator,
    range_inclusive_validator,
    greater_than_validator,
)


class TransformResampleObjectFilter(FilterBlock):

    KEY_OBJECT = "object"
    KEY_ACQUISITION_SHAPE = "acquisition_shape"
    KEY_ROTATION_ORIGIN = "rotation_origin"
    KEY_ROTATION = "rotation"
    KEY_TRANSLATION = "translation"
    KEY_IMAGE = "image"

    def __init__(self):
        super().__init__(name="Transform and Resample Object")

    def _create_filter_block(self):
        """ Transforms the object in world-space, then creates a resampled image at 
        with the specified acquisition shape """

        object_image: BaseImageContainer = self.inputs[self.KEY_OBJECT]

        translation = self.inputs[self.KEY_TRANSLATION]
        rotation = self.inputs[self.KEY_ROTATION]
        rotation_origin = self.inputs[self.KEY_ROTATION_ORIGIN]

        affine_object_transform_filter = AffineMatrixFilter()
        # we are moving the object, so the affine needs to be constructed using the inverse of the
        # transformation parameters
        # use the inverse of the object affine to move to world-space

        # affine_object_transform_filter.add_input(
        #    AffineMatrixFilter.KEY_AFFINE, np.linalg.inv(object_image.affine)
        # )
        affine_object_transform_filter.add_input(
            AffineMatrixFilter.KEY_ROTATION, tuple(-1 * np.array(rotation))
        )
        affine_object_transform_filter.add_input(
            AffineMatrixFilter.KEY_ROTATION_ORIGIN,
            tuple(-1 * np.array(rotation_origin)),
        )
        affine_object_transform_filter.add_input(
            AffineMatrixFilter.KEY_TRANSLATION, tuple(-1 * np.array(translation))
        )
        # affine_object_transform_filter.add_input(
        #    AffineMatrixFilter.KEY_AFFINE_LAST, (object_image.affine)
        # )

        affine_acquisition_filter = AffineMatrixFilter()
        scale: np.array = (
            np.array(object_image.shape)
            / np.array(self.inputs[self.KEY_ACQUISITION_SHAPE])
        )
        acquisition_offset: np.array = np.array(object_image.affine[:3, 3] / scale)

        affine_acquisition_filter.add_parent_filter(
            affine_object_transform_filter, io_map={"affine": "affine_last"}
        )
        affine_acquisition_filter.add_input(AffineMatrixFilter.KEY_SCALE, tuple(scale))
        affine_acquisition_filter.add_input(
            AffineMatrixFilter.KEY_TRANSLATION, tuple(acquisition_offset)
        )

        resample_filter = ResampleFilter()
        resample_filter.add_parent_filter(affine_acquisition_filter)
        resample_filter.add_input(
            ResampleFilter.KEY_SHAPE, self.inputs[self.KEY_ACQUISITION_SHAPE]
        )
        resample_filter.add_input(
            ResampleFilter.KEY_IMAGE, self.inputs[self.KEY_OBJECT]
        )

        return resample_filter

    def _validate_inputs(self):
        """ Checks that the inputs meet their validation criteria
        `'object'` must be derived from BaseImageContainer
        `'acquisition_shape'` (optional)must be a Tuple of ints of length 3, values > 0
        `'rotation'` (optional) must be a Tuple of floats of length 3, each value -180 to 180
        inclusive, default (optional) = (0.0, 0.0, 0.0)
        `'rotation_origin'` (optional) must be a Tuple of floats of length 3,
        default = (0.0, 0.0, 0.0)
        `'translation'` (optional) must be a Tuple of floats of length 3, default = (0.0, 0.0, 0.0)
        """

        input_validator = ParameterValidator(
            parameters={
                self.KEY_OBJECT: Parameter(
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
                self.KEY_ACQUISITION_SHAPE: Parameter(
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

        if new_params[self.KEY_ACQUISITION_SHAPE] == (9999, 9999, 9999):
            new_params[self.KEY_ACQUISITION_SHAPE] = self.inputs[self.KEY_OBJECT].shape

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
        if len(new_params[self.KEY_ACQUISITION_SHAPE]) != 3:
            raise FilterInputValidationError

        # merge the updated parameters from the output with the input parameters
        self.inputs = {**self._i, **new_params}

