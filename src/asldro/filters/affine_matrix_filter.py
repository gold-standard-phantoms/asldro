""" Affine Matrix Filter """

from typing import Tuple
import numpy as np
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.validators.parameters import (
    ParameterValidator,
    Parameter,
    isinstance_validator,
    for_each_validator,
    range_inclusive_validator,
)


class AffineMatrixFilter(BaseFilter):
    r"""A filter that creates an affine transformation matrix
    based on input parameters for rotation, translation, and scaling.

    Conventions are for RAS+ coordinate systems only

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`AffineMatrixFilter.add_inputs()`
    member function. They are also accessible via class constants,
    for example :class:`AffineMatrixFilter.KEY_ROTATION`

    :param 'rotation': [:math:`\theta_x`, :math:`\theta_y`, :math:`\theta_z`]
        angles to rotate about the x, y and z axes in degrees(-180 to 180 degrees inclusive),
        defaults to (0, 0, 0)
    :type 'rotation': Tuple[float, float, float], optional
    :param 'rotation_origin': [:math:`x_r`, :math:`y_r`, :math:`z_r`]
        coordinates of the point to perform rotations about, defaults to (0, 0, 0)
    :type 'rotation_origin': Tuple[float, float, float], optional
    :param 'translation': [:math:`\Delta x`, :math:`\Delta y`, :math:`\Delta z`]
        amount to translate along the x, y and z axes. defaults to (0, 0, 0)
    :type 'translation': Tuple[float, float, float], optional
    :param 'scale': [:math:`s_x`, :math:`s_y`, :math:`s_z`]
        scaling factors along each axis, defaults to (1, 1, 1)
    :type 'scale': Tuple[float, float, float], optional
    :param 'affine':  4x4 affine matrix to apply transformation to, defaults to `numpy.eye(4)`
    :type 'affine': np.ndarray(4), optional
    :param 'affine_last': input 4x4 affine matrix that is applied last, defaults to `numpy.eye(4)`
    :type 'affine_last': np.ndarray(4), optional

    **Outputs**

    Once run, the filter will populate the dictionary :class:`AffineMatrixFilter.outputs` with the
    following entries

    :param 'affine':  4x4 affine matrix with all transformations combined.
    :type 'affine': np.ndarray(4)
    :param 'affine_inverse': 4x4 affine matrix that is the inverse of `'affine'`
    :type 'affine_inverse': np.ndarray(4)

    The output affine matrix is calculated as follows:

    .. math::

        &\mathbf{M} = \mathbf{B}\mathbf{S}\mathbf{T}\mathbf{T_{r}}\mathbf{R_z}
        \mathbf{R_y}\mathbf{R_x}\mathbf{T_{r}}^{-1}\mathbf{M_\text{in}}\\
        \\
        \text{where,}&\\
        &\mathbf{A} = \text{Existing affine matrix}\\
        &\mathbf{B} = \text{Affine matrix to combine last}\\
        &\mathbf{S} = \begin{pmatrix} s_x & 0 & 0 & 0 \\ 0 & s_y & 0 & 0 \\ 0 & 0 & s_z & 0 \\
        0& 0 & 0& 1 \end{pmatrix}=\text{scaling matrix}\\
        &\mathbf{T} = \begin{pmatrix} 1 & 0 & 0 & \Delta x \\ 0 & 1& 0 & \Delta y \\
        0 & 0 & 1& \Delta z \\
        0& 0 & 0& 1 \end{pmatrix}=\text{translation matrix}\\
        &\mathbf{T_r} = \begin{pmatrix} 1 & 0 & 0 & x_r \\ 0 & 1& 0 & y_r \\
        0 & 0 & 1& z_r \\ 0& 0 & 0& 1 \end{pmatrix}=
        \text{translation to rotation centre matrix}\\
        &\mathbf{R_x} = \begin{pmatrix} 1 & 0 & 0 & 0\\ 0 & \cos{\theta_x}& -\sin{\theta_x} & 0\\
        0 & \sin{\theta_x} & \cos{\theta_x}& 0\\ 0& 0 & 0& 1 \end{pmatrix}=
        \text{rotation about x matrix}\\
        &\mathbf{R_y} = \begin{pmatrix} \cos{\theta_y} & 0 & \sin{\theta_y} & 0\\
        0 & 1 & 0 & 0\\ -\sin{\theta_y} & 0 & \cos{\theta_y}& 0\\ 0& 0 & 0& 1 \end{pmatrix}=
        \text{rotation about y matrix}\\
        &\mathbf{R_z} = \begin{pmatrix} \cos{\theta_z}& -\sin{\theta_z} & 0 & 0\\
        \sin{\theta_z} & \cos{\theta_z}& 0 &0\\ 0& 0& 1 & 0\\ 0& 0 & 0& 1 \end{pmatrix}=
        \text{rotation about z matrix}\\

    """

    KEY_ROTATION = "rotation"
    KEY_ROTATION_ORIGIN = "rotation_origin"
    KEY_TRANSLATION = "translation"
    KEY_SCALE = "scale"
    KEY_AFFINE = "affine"
    KEY_AFFINE_LAST = "affine_last"
    KEY_AFFINE_INVERSE = "affine_inverse"

    def __init(self):
        super().__init__(name="Compute Affine Matrix")

    def _run(self):
        # construct individual transformation matrices
        input_affine: np.ndarray = self.inputs[self.KEY_AFFINE]
        affine_last: np.ndarray = self.inputs[self.KEY_AFFINE_LAST]
        rotation_angles: Tuple[float, float, float] = np.radians(
            self.inputs[self.KEY_ROTATION]
        )
        rotation_origin: Tuple[float, float, float] = self.inputs[
            self.KEY_ROTATION_ORIGIN
        ]
        translation: Tuple[float, float, float] = self.inputs[self.KEY_TRANSLATION]
        scale: Tuple[float, float, float] = self.inputs[self.KEY_SCALE]

        scale_matrix = np.array(
            (
                (scale[0], 0, 0, 0),
                (0, scale[1], 0, 0),
                (0, 0, scale[2], 0),
                (0, 0, 0, 1),
            )
        )
        translation_matrix = np.array(
            (
                (1, 0, 0, translation[0]),
                (0, 1, 0, translation[1]),
                (0, 0, 1, translation[2]),
                (0, 0, 0, 1),
            )
        )
        rotation_origin_translation_matrix = np.array(
            (
                (1, 0, 0, rotation_origin[0]),
                (0, 1, 0, rotation_origin[1]),
                (0, 0, 1, rotation_origin[2]),
                (0, 0, 0, 1),
            )
        )

        rotation_x_matrix = np.array(
            (
                (1, 0, 0, 0),
                (0, np.cos(rotation_angles[0]), -np.sin(rotation_angles[0]), 0),
                (0, np.sin(rotation_angles[0]), np.cos(rotation_angles[0]), 0),
                (0, 0, 0, 1),
            )
        )
        rotation_y_matrix = np.array(
            (
                (np.cos(rotation_angles[1]), 0, np.sin(rotation_angles[1]), 0),
                (0, 1, 0, 0),
                (-np.sin(rotation_angles[1]), 0, np.cos(rotation_angles[1]), 0),
                (0, 0, 0, 1),
            )
        )
        rotation_z_matrix = np.array(
            (
                (np.cos(rotation_angles[2]), -np.sin(rotation_angles[2]), 0, 0),
                (np.sin(rotation_angles[2]), np.cos(rotation_angles[2]), 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            )
        )

        # inverse of rotation_origin_translation_matrix  doesn't need to be inverted,
        # it is simply 2*np.eye(4) - rotation_origin_translation_matrix
        inv_rotation_origin_translation_matrix = (
            2 * np.eye(4) - rotation_origin_translation_matrix
        )

        # combine
        output_affine: np.ndarray = (
            affine_last
            @ scale_matrix
            @ translation_matrix
            @ rotation_origin_translation_matrix
            @ rotation_z_matrix
            @ rotation_y_matrix
            @ rotation_x_matrix
            @ inv_rotation_origin_translation_matrix
            @ input_affine
        )

        self.outputs[self.KEY_AFFINE] = output_affine
        self.outputs[self.KEY_AFFINE_INVERSE] = np.linalg.inv(output_affine)

    def _validate_inputs(self):
        """ Checks that the inputs meet their validation criteria

        `'rotation'` (optional) must be a Tuple of floats of length 3, each value -180 to 180
        inclusive,default (optional) = (0.0, 0.0, 0.0)
        `'rotation_origin'` (optional) must be a Tuple of floats of length 3,
        default = (0.0, 0.0, 0.0)
        `'translation'` (optional) must be a Tuple of floats of length 3, default = (0.0, 0.0, 0.0)
        `'scale'` (optional) must be a Tuple of floats of length 3, default = (1.0, 1.0, 1.0)
        `'affine'` (optional) must be a numpy.ndarray of shape (4,4), default = numpy.eye(4)
        """

        input_validator = ParameterValidator(
            parameters={
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
                self.KEY_SCALE: Parameter(
                    validators=[
                        isinstance_validator(tuple),
                        for_each_validator(isinstance_validator(float)),
                    ],
                    optional=True,
                    default_value=(1.0, 1.0, 1.0),
                ),
                self.KEY_AFFINE: Parameter(
                    validators=[isinstance_validator(np.ndarray)],
                    optional=True,
                    default_value=np.eye(4),
                ),
                self.KEY_AFFINE_LAST: Parameter(
                    validators=[isinstance_validator(np.ndarray)],
                    optional=True,
                    default_value=np.eye(4),
                ),
            }
        )

        # validate, returning a dictionary which also includes default parameters
        new_params = input_validator.validate(
            self.inputs, error_type=FilterInputValidationError
        )

        # Further validation that can't be handled by the parameter validator
        # Check that AffineMatrixFilter.KEY_AFFINE is of size 4x4
        if new_params[self.KEY_AFFINE].shape != (4, 4):
            raise FilterInputValidationError
        # Check that AffineMatrixFilter.KEY_AFFINE_LAST is of size 4x4
        if new_params[self.KEY_AFFINE_LAST].shape != (4, 4):
            raise FilterInputValidationError

        # Check that the tuple AffineMatrixFilter.KEY_ROTATION's length is 3
        if len(new_params[self.KEY_ROTATION]) != 3:
            raise FilterInputValidationError

        # Check that the tuple AffineMatrixFilter.KEY_ROTATION_ORIGIN's length is 3
        if len(new_params[self.KEY_ROTATION_ORIGIN]) != 3:
            raise FilterInputValidationError

        # Check that the tuple AffineMatrixFilter.KEY_TRANSLATION's length is 3
        if len(new_params[self.KEY_TRANSLATION]) != 3:
            raise FilterInputValidationError

        # Check that the tuple AffineMatrixFilter.KEY_SCALE's length is 3
        if len(new_params[self.KEY_SCALE]) != 3:
            raise FilterInputValidationError

        # merge the updated parameters from the output with the input parameters
        self.inputs = {**self._i, **new_params}
