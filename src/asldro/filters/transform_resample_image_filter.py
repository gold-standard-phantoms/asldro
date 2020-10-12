"""Transform resample image filter"""

import numpy as np
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


class TransformResampleImageFilter(FilterBlock):

    KEY_TARGET_SHAPE = "target_shape"
    KEY_ROTATION_ORIGIN = "rotation_origin"
    KEY_ROTATION = "rotation"
    KEY_TRANSLATION = "translation"
    KEY_IMAGE = "image"

    def __init__(self):
        super().__init__(name="Transform and Resample Object")

    def _create_filter_block(self):
        r""" Transforms the object in world-space, then creates a resampled image at 
        with the specified acquisition shape 
        
        The transformation affine is calculated using four separate AffineMatrixFilters which are
        linked together as follows:
        .. math::
            &\mathbf{A}=(\mathbf{T(\Delta r_{\text{im}})})
            \cdot(\mathbf{S})
            \cdot(\mathbf{R}\mathbf{T(\Delta r)})
            \cdot(\mathbf{R}^{-1}\mathbf{T(r_0)}\mathbf{R}\mathbf{T(r_0)}^{-1})
        
        Parenthesis indicate grouping by AffineMatrixFilter
        """

        input_image: BaseImageContainer = self.inputs[self.KEY_IMAGE]

        translation = self.inputs[self.KEY_TRANSLATION]
        rotation = self.inputs[self.KEY_ROTATION]
        rotation_origin = self.inputs[self.KEY_ROTATION_ORIGIN]

        # Affine matrix used for rotation in the motion model: :math:`\mathbf{R}`
        # calculate this separetely because it and its inverse is required later
        motion_model_rotation_affine = AffineMatrixFilter()
        motion_model_rotation_affine.add_input(
            AffineMatrixFilter.KEY_ROTATION, rotation
        )

        # affine_1: the rotation in world space, with the inverse of the rotation appended:
        # :math:`\mathbf{R}^{-1}\mathbf{T(r_0)}\mathbf{R}\mathbf{T(r_0)}^{-1}`
        affine_1 = AffineMatrixFilter()
        affine_1.add_input(AffineMatrixFilter.KEY_ROTATION, rotation)
        affine_1.add_input(AffineMatrixFilter.KEY_ROTATION_ORIGIN, rotation_origin)
        affine_1.add_parent_filter(
            motion_model_rotation_affine,
            io_map={
                AffineMatrixFilter.KEY_AFFINE_INVERSE: AffineMatrixFilter.KEY_AFFINE_LAST
            },
        )

        # affine_2: translation in the motion model.
        # :math:`\mathbf{R}\mathbf{T(\Delta r)}`
        affine_2 = AffineMatrixFilter()
        affine_2.add_input(AffineMatrixFilter.KEY_TRANSLATION, translation)
        # map affine_1.outputs["affine"] to affine_2.inputs["affine"]
        affine_2.add_parent_filter(affine_1)
        # map motion_model_rotation_affine.outputs["affine"] to affine_2.inputs["affine_last"]
        affine_2.add_parent_filter(
            motion_model_rotation_affine,
            io_map={AffineMatrixFilter.KEY_AFFINE: AffineMatrixFilter.KEY_AFFINE_LAST},
        )

        # affine_3: scale voxels to their new size
        # :math:`\mathbf{S}`
        scale: np.array = (
            np.array(self.inputs[self.KEY_TARGET_SHAPE]) / np.array(input_image.shape)
        )

        affine_3 = AffineMatrixFilter()
        affine_3.add_input(AffineMatrixFilter.KEY_SCALE, tuple(scale))
        # map affine_2.outputs["affine"] to affine_3.inputs["affine"]
        affine_3.add_parent_filter(affine_2)

        # affine_4: shift to the centre of the image
        # :math:`\mathbf{T(\Delta r_{\text{im}})}`
        image_centre_offset = np.array(self.inputs[self.KEY_TARGET_SHAPE]) / 2
        affine_4 = AffineMatrixFilter()
        affine_4.add_input(
            AffineMatrixFilter.KEY_TRANSLATION, tuple(image_centre_offset)
        )
        # map affine_3.outputs["affine"] to affine_4.inputs["affine"]
        affine_4.add_parent_filter(affine_3)

        resample_filter = ResampleFilter()
        # map affine_4.output["affine_inverse"] to resample_filter.inputs["affine"]
        resample_filter.add_parent_filter(
            affine_4,
            io_map={AffineMatrixFilter.KEY_AFFINE_INVERSE: ResampleFilter.KEY_AFFINE},
        )
        resample_filter.add_input(
            ResampleFilter.KEY_SHAPE, self.inputs[self.KEY_TARGET_SHAPE]
        )
        resample_filter.add_input(ResampleFilter.KEY_IMAGE, self.inputs[self.KEY_IMAGE])
        # TODO: update the resampled image's affine so that it is equal to
        # :math:`(\mathbf{S} \mathbf{A_{\text{i}}}^{-1})^{-1}`
        #

        return resample_filter

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

