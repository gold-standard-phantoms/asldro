""" AcquireMriImageFilter Class"""

from asldro.containers.image import BaseImageContainer
from asldro.filters.basefilter import FilterInputValidationError
from asldro.filters.filter_block import FilterBlock
from asldro.filters.add_complex_noise_filter import AddComplexNoiseFilter
from asldro.filters.transform_resample_image_filter import TransformResampleImageFilter
from asldro.filters.mri_signal_filter import MriSignalFilter

from asldro.validators.parameters import (
    ParameterValidator,
    Parameter,
    isinstance_validator,
)


class AcquireMriImageFilter(FilterBlock):
    r"""A filter block that simulates the acquisition of an MRI image based on
    ground truth inputs.

    Combines:

    1. :class:`.MriSignalFilter`
    2. :class:`.TransformResampleImageFilter`
    3. :class:`.AddComplexNoiseFilter`

    Returns :class:`AddComplexNoiseFilter`

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`AcquireMriImageFilter.add_inputs()`
    member function. They are also accessible via class constants,
    for example :class:`AcquireMriImageFilter.KEY_T1`

    :param 't1':  Longitudinal relaxation time in seconds
    :type 't1': BaseImageContainer
    :param 't2': Transverse relaxation time in seconds
    :type 't2': BaseImageContainer
    :param 't2_star': Transverse relaxation time including time-invariant magnetic
        field inhomogeneities.
    :type 't2_star': BaseImageContainer
    :param 'm0': Equilibrium magnetisation
    :type 'm0': BaseImageContainer
    :param 'mag_eng': Added to M0 before relaxation is calculated,
        provides a means to encode another signal into the MRI signal (non-complex data).
    :type 'mag_enc': BaseImageContainer, optional.
    :param 'acq_contrast': Determines which signal model to use:
      ``"ge"`` (case insensitive) for Gradient Echo, ``"se"`` (case insensitive) for Spin Echo,
      ``"ir"`` (case insensitive) for Inversion Recovery.
    :type 'acq_contrast': str
    :param 'echo_time': The echo time in seconds
    :type 'echo_time': float
    :param 'repetition_time': The repeat time in seconds
    :type 'repetition_time': float
    :param 'excitation_flip_angle': Excitation pulse flip angle in degrees. Only used when
        ``"acq_contrast"`` is ``"ge"`` or ``"ir"``.
    :type 'excitation_flip_angle': float
    :param 'inversion_flip_angle': Inversion pulse flip angle in degrees. Only used when
        ``acq_contrast`` is ``"ir"``.
    :type 'inversion_flip_angle': float, optional
    :param 'inversion_time': The inversion time in seconds. Only used when
        ``acq_contrast`` is ``"ir"``.
    :param 'image_flavour': sets the metadata ``image_flavour`` in the output image to this.
    :type 'image_flavour': str, optional
    :param 'translation': :math:`[\Delta x,\Delta y,\Delta z]`
        amount to translate along the x, y and z axes.
    :type translation: Tuple[float, float, float], optional
    :param 'rotation': :math:`[\theta_x,\theta_y,\theta_z]`
        angles to rotate about the x, y and z axes in degrees(-180 to 180 degrees inclusive).
    :type 'rotation': Tuple[float, float, float], optional
    :param 'rotation_origin': :math:`[x_r,y_r,z_r]`
        coordinates of the point to perform rotations about.
    :type 'rotation_origin': Tuple[float, float, float], optional
    :param target_shape: :math:`[L_t,M_t,N_t]` target shape for the acquired image
    :type target_shape: Tuple[int, int, int], optional
    :param 'snr': the desired signal-to-noise ratio (>= 0). A value of zero means that no noise
        is added to the input image.
    :type 'snr': float
    :param 'reference_image': The reference image that is used to calculate the amplitude of
        the random noise to add to `'image'`. The shape of this must match the shape of `'image'`.
        If this is not supplied then `'image'` will be used for calculating the noise amplitude.
    :type 'reference_image': BaseImageContainer, optional

    **Outputs**

    :param 'image': Synthesised MRI image.
    :type 'image': BaseImageContainer

    """

    # Key constants
    KEY_T1 = MriSignalFilter.KEY_T1
    KEY_T2 = MriSignalFilter.KEY_T2
    KEY_T2_STAR = MriSignalFilter.KEY_T2_STAR
    KEY_M0 = MriSignalFilter.KEY_M0
    KEY_MAG_ENC = MriSignalFilter.KEY_MAG_ENC
    KEY_ACQ_CONTRAST = MriSignalFilter.KEY_ACQ_CONTRAST
    KEY_ECHO_TIME = MriSignalFilter.KEY_ECHO_TIME
    KEY_REPETITION_TIME = MriSignalFilter.KEY_REPETITION_TIME
    KEY_EXCITATION_FLIP_ANGLE = MriSignalFilter.KEY_EXCITATION_FLIP_ANGLE
    KEY_INVERSION_FLIP_ANGLE = MriSignalFilter.KEY_INVERSION_FLIP_ANGLE
    KEY_INVERSION_TIME = MriSignalFilter.KEY_INVERSION_TIME
    KEY_IMAGE = MriSignalFilter.KEY_IMAGE
    KEY_IMAGE_FLAVOUR = MriSignalFilter.KEY_IMAGE_FLAVOUR

    KEY_TARGET_SHAPE = TransformResampleImageFilter.KEY_TARGET_SHAPE
    KEY_ROTATION_ORIGIN = TransformResampleImageFilter.KEY_ROTATION_ORIGIN
    KEY_ROTATION = TransformResampleImageFilter.KEY_ROTATION
    KEY_TRANSLATION = TransformResampleImageFilter.KEY_TRANSLATION

    KEY_REF_IMAGE = AddComplexNoiseFilter.KEY_REF_IMAGE
    KEY_SNR = AddComplexNoiseFilter.KEY_SNR

    def __init__(self):
        super().__init__(name="Acquire MRI Image Filter")

    def _create_filter_block(self):
        """Runs:
        1. MriSignalFilter
        2. TransformResampleFilter
        3. AddComplexNoiseFilter

        Returns AddComplexNoiseFilter
        """

        add_complex_noise_filter = AddComplexNoiseFilter()

        # MriSignalFilter
        # add required inputs - these should always be present
        mri_signal_filter = MriSignalFilter()
        mri_signal_filter.add_input(MriSignalFilter.KEY_T1, self.inputs[self.KEY_T1])
        mri_signal_filter.add_input(MriSignalFilter.KEY_T2, self.inputs[self.KEY_T2])
        mri_signal_filter.add_input(
            MriSignalFilter.KEY_T2_STAR, self.inputs[self.KEY_T2_STAR]
        )
        mri_signal_filter.add_input(MriSignalFilter.KEY_M0, self.inputs[self.KEY_M0])
        mri_signal_filter.add_input(
            MriSignalFilter.KEY_ACQ_CONTRAST, self.inputs[self.KEY_ACQ_CONTRAST]
        )
        mri_signal_filter.add_input(
            MriSignalFilter.KEY_ECHO_TIME, self.inputs[self.KEY_ECHO_TIME]
        )
        mri_signal_filter.add_input(
            MriSignalFilter.KEY_REPETITION_TIME, self.inputs[self.KEY_REPETITION_TIME]
        )

        mri_signal_filter.add_input(
            MriSignalFilter.KEY_EXCITATION_FLIP_ANGLE,
            self.inputs[self.KEY_EXCITATION_FLIP_ANGLE],
        )
        # add optional inputs if present
        if self.inputs.get(self.KEY_MAG_ENC) is not None:
            mri_signal_filter.add_input(
                MriSignalFilter.KEY_MAG_ENC, self.inputs[self.KEY_MAG_ENC]
            )

        if self.inputs.get(self.KEY_INVERSION_FLIP_ANGLE) is not None:
            mri_signal_filter.add_input(
                MriSignalFilter.KEY_INVERSION_FLIP_ANGLE,
                self.inputs[self.KEY_INVERSION_FLIP_ANGLE],
            )
        if self.inputs.get(self.KEY_INVERSION_TIME) is not None:
            mri_signal_filter.add_input(
                MriSignalFilter.KEY_INVERSION_TIME, self.inputs[self.KEY_INVERSION_TIME]
            )
        if self.inputs.get(self.KEY_IMAGE_FLAVOUR) is not None:
            mri_signal_filter.add_input(
                MriSignalFilter.KEY_IMAGE_FLAVOUR, self.inputs[self.KEY_IMAGE_FLAVOUR]
            )

        # TransformResampleImageFilter
        transform_resample_image_filter = TransformResampleImageFilter()
        # Add mri_signal_filter as parent
        transform_resample_image_filter.add_parent_filter(mri_signal_filter)
        # all other parameters are optional
        if self.inputs.get(self.KEY_ROTATION) is not None:
            transform_resample_image_filter.add_input(
                TransformResampleImageFilter.KEY_ROTATION,
                self.inputs[self.KEY_ROTATION],
            )
        if self.inputs.get(self.KEY_ROTATION_ORIGIN) is not None:
            transform_resample_image_filter.add_input(
                TransformResampleImageFilter.KEY_ROTATION_ORIGIN,
                self.inputs[self.KEY_ROTATION_ORIGIN],
            )
        if self.inputs.get(self.KEY_TARGET_SHAPE) is not None:
            transform_resample_image_filter.add_input(
                TransformResampleImageFilter.KEY_TARGET_SHAPE,
                self.inputs[self.KEY_TARGET_SHAPE],
            )
        if self.inputs.get(self.KEY_TRANSLATION) is not None:
            transform_resample_image_filter.add_input(
                TransformResampleImageFilter.KEY_TRANSLATION,
                self.inputs[self.KEY_TRANSLATION],
            )

        # AddComplexNoiseFilter
        add_complex_noise_filter = AddComplexNoiseFilter()
        # add transform_resample_image_filter as parent
        add_complex_noise_filter.add_parent_filter(transform_resample_image_filter)
        # add required inputs - these should always be present
        add_complex_noise_filter.add_input(self.KEY_SNR, self.inputs[self.KEY_SNR])
        # add optional input ref_image
        if self.inputs.get(self.KEY_REF_IMAGE) is not None:
            add_complex_noise_filter.add_input(
                AddComplexNoiseFilter.KEY_REF_IMAGE,
                self.inputs[self.KEY_REF_IMAGE],
            )

        # return add_complex_noise_filter
        return add_complex_noise_filter

    def _validate_inputs(self):
        """
        Checks that the inputs meet their validation criteria
        Note that values are only checked if they are present and the correct
        type, as more rigorous checking is performed in each corresponding filter.
        't1': BaseImageContainer
        't2': BaseImageContainer
        't2_star': BaseImageContainer
        'm0': BaseImageContainer
        'mag_enc': BaseImageContainer, optional
        'acq_contrast': str
        'echo_time': float
        'repetition_time': float
        'excitation_flip_angle': float, optional
        'inversion_flip_angle': float, optional
        'inversion_time': float, optional
        'image_flavour': str, optional
        'target_shape': Tuple[int, int, int], optional
        'rotation': Tuple[float, float, float], optional
        'rotation_origin': Tuple[float, float, float], optional
        'translation': Tuple[float, float, float], optional
        'reference_image': BaseImageContainer, optional
        'snr': float
        """
        input_validator = ParameterValidator(
            parameters={
                self.KEY_M0: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                    ]
                ),
                self.KEY_T1: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                    ]
                ),
                self.KEY_T2: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                    ]
                ),
                self.KEY_T2_STAR: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                    ],
                ),
                self.KEY_MAG_ENC: Parameter(
                    validators=[isinstance_validator(BaseImageContainer)], optional=True
                ),
                self.KEY_ACQ_CONTRAST: Parameter(
                    validators=[
                        isinstance_validator(str),
                    ]
                ),
                self.KEY_ECHO_TIME: Parameter(
                    validators=[
                        isinstance_validator(float),
                    ]
                ),
                self.KEY_REPETITION_TIME: Parameter(
                    validators=[
                        isinstance_validator(float),
                    ]
                ),
                self.KEY_EXCITATION_FLIP_ANGLE: Parameter(
                    validators=[
                        isinstance_validator(float),
                    ],
                ),
                self.KEY_INVERSION_FLIP_ANGLE: Parameter(
                    validators=[
                        isinstance_validator(float),
                    ],
                    optional=True,
                ),
                self.KEY_INVERSION_TIME: Parameter(
                    validators=[
                        isinstance_validator(float),
                    ],
                    optional=True,
                ),
                self.KEY_IMAGE_FLAVOUR: Parameter(
                    validators=[
                        isinstance_validator(str),
                    ],
                    optional=True,
                ),
                self.KEY_TARGET_SHAPE: Parameter(
                    validators=[
                        isinstance_validator(tuple),
                    ],
                    optional=True,
                ),
                self.KEY_ROTATION: Parameter(
                    validators=[
                        isinstance_validator(tuple),
                    ],
                    optional=True,
                ),
                self.KEY_ROTATION_ORIGIN: Parameter(
                    validators=[
                        isinstance_validator(tuple),
                    ],
                    optional=True,
                ),
                self.KEY_TRANSLATION: Parameter(
                    validators=[
                        isinstance_validator(tuple),
                    ],
                    optional=True,
                ),
                self.KEY_REF_IMAGE: Parameter(
                    validators=[isinstance_validator(BaseImageContainer)], optional=True
                ),
                self.KEY_SNR: Parameter(
                    validators=[
                        isinstance_validator(float),
                    ]
                ),
            }
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)
