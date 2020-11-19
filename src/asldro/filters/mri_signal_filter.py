""" MRI Signal Filter """

import numpy as np
from asldro.containers.image import BaseImageContainer, COMPLEX_IMAGE_TYPE
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.validators.parameters import (
    ParameterValidator,
    Parameter,
    greater_than_equal_to_validator,
    from_list_validator,
    isinstance_validator,
)


class MriSignalFilter(BaseFilter):
    r""" A filter that generates either the Gradient Echo, Spin Echo or
    Inversion Recovery MRI signal.

    * Gradient echo is with arbitrary excitation flip angle.
    * Spin echo assumes perfect 90° excitation and 180° refocusing pulses.
    * Inversion recovery can have arbitrary inversion pulse and excitation pulse flip angles.

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`MriSignalFilter.add_inputs()`
    member function. They are also accessible via class constants,
    for example :class:`MriSignalFilter.KEY_T1`

    :param 't1':  Longitudinal relaxation time in seconds (>=0, non-complex data)
    :type 't1': BaseImageContainer
    :param 't2': Transverse relaxation time in seconds (>=0, non-complex data)
    :type 't2': BaseImageContainer
    :param 't2_star': Transverse relaxation time including time-invariant magnetic
        field inhomogeneities, only required for gradient echo (>=0, non-complex data)
    :type 't2_star': BaseImageContainer
    :param 'm0': Equilibrium magnetisation (>=0, non-complex data)
    :type 'm0': BaseImageContainer
    :param 'mag_eng': Added to M0 before relaxation is calculated,
        provides a means to encode another signal into the MRI signal (non-complex data)
    :type 'mag_enc': BaseImageContainer, optional.
    :param 'acq_contrast': Determines which signal model to use:
      ``"ge"`` (case insensitive) for Gradient Echo, ``"se"`` (case insensitive) for Spin Echo,
      ``"ir"`` (case insensitive) for Inversion Recovery.
    :type 'acq_contrast': str
    :param 'echo_time': The echo time in seconds (>=0)
    :type 'echo_time': float
    :param 'repetition_time': The repeat time in seconds (>=0)
    :type 'repetition_time': float
    :param 'excitation_flip_angle': Excitation pulse flip angle in degrees. Only used when
        ``"acq_contrast"`` is ``"ge"`` or ``"ir"``.  Defaults to 90.0
    :type 'excitation_flip_angle': float, optional
    :param 'inversion_flip_angle': Inversion pulse flip angle in degrees. Only used when
        ``acq_contrast`` is ``"ir"``. Defaults to 180.0
    :type 'inversion_flip_angle': float, optional
    :param 'inversion_time': The inversion time in seconds. Only used when
        ``acq_contrast`` is ``"ir"``. Defaults to 1.0.
    :param 'image_flavour': sets the metadata ``image_flavour`` in the output image to this.
    :type 'image_flavour': str

    **Outputs**

    Once run, the filter will populate the dictionary :class:`MriSignalFilter.outputs` with the
    following entries

    :param 'image': An image of the generated MRI signal. Will be of the same class
      as the input ``t1``
    :type 'image': BaseImageContainer

    The following parameters are added to :class:`MriSignalFilter.outputs["image"].metadata`:

    * ``acq_contrast``
    * ``echo time``
    * ``excitation_flip_angle``
    * ``image_flavour``
    * ``inversion_time``
    * ``inversion_flip_angle``
    * ``mr_acq_type`` = "3D"

    ``image_flavour`` is obtained (in order of precedence):

    #. If present, from the input ``image_flavour``
    #. If present, derived from the metadata in the input ``mag_enc``
    #. "OTHER"

    The following equations are used to compute the MRI signal:


    *Gradient Echo*

    .. math::
        S(\text{TE},\text{TR}, \theta_1) = \sin\theta_1\cdot(\frac{M_0
        \cdot(1-e^{-\frac{TR}{T_{1}}})}
        {1-\cos\theta_1 e^{-\frac{TR}{T_{1}}}-e^{-\frac{TR}{T_{2}}}\cdot
        \left(e^{-\frac{TR}{T_{1}}}-\cos\theta_1\right)}  + M_{\text{enc}})
        \cdot e^{-\frac{\text{TE}}{T^{*}_2}}


    *Spin Echo*

    .. math::
       S(\text{TE},\text{TR}) = (M_0 \cdot (1-e^{-\frac{\text{TR}}{T_1}}) + M_{\text{enc}})
       \cdot e^{-\frac{\text{TE}}{T_2}}


    *Inversion Recovery*

    .. math::
        &S(\text{TE},\text{TR}, \text{TI}, \theta_1, \theta_2) =
        \sin\theta_1 \cdot (\frac{M_0(1-\left(1-\cos\theta_{2}\right)
        e^{-\frac{TI}{T_{1}}}-\cos\theta_{2}e^{-\frac{TR}{T_{1}}})}
        {1-\cos\theta_{1}\cos\theta_{2}e^{-\frac{TR}{T_{1}}}}+ M_\text{enc})
        \cdot e^{-\frac{TE}{T_{2}}}\\
        &\theta_1 = \text{excitation pulse flip angle}\\
        &\theta_2 = \text{inversion pulse flip angle}

    """

    # Key constants
    KEY_T1 = "t1"
    KEY_T2 = "t2"
    KEY_T2_STAR = "t2_star"
    KEY_M0 = "m0"
    KEY_MAG_ENC = "mag_enc"
    KEY_ACQ_CONTRAST = "acq_contrast"
    KEY_ECHO_TIME = "echo_time"
    KEY_REPETITION_TIME = "repetition_time"
    KEY_EXCITATION_FLIP_ANGLE = "excitation_flip_angle"
    KEY_INVERSION_FLIP_ANGLE = "inversion_flip_angle"
    KEY_INVERSION_TIME = "inversion_time"
    KEY_IMAGE = "image"
    KEY_IMAGE_FLAVOUR = "image_flavour"
    KEY_ACQ_TYPE = "mr_acq_type"

    # Value constants
    CONTRAST_GE = "ge"
    CONTRAST_SE = "se"
    CONTRAST_IR = "ir"

    def __init__(self):
        super().__init__(name="MRI Signal Model")

    def _run(self):

        t1: np.ndarray = self.inputs[self.KEY_T1].image
        t2: np.ndarray = self.inputs[self.KEY_T2].image
        m0: np.ndarray = self.inputs[self.KEY_M0].image

        metadata = {}
        if self.inputs.get(self.KEY_MAG_ENC) is not None:
            mag_enc: np.ndarray = self.inputs[self.KEY_MAG_ENC].image
            metadata = self.inputs[self.KEY_MAG_ENC].metadata
        else:
            mag_enc: np.ndarray = np.zeros(t1.shape)
            metadata = {}

        # mag_enc might not have "image_flavour" set
        if metadata.get("image_flavour") is None:
            metadata["image_flavour"] = "OTHER"

        # if present override image_flavour with the input
        if self.inputs.get(self.KEY_IMAGE_FLAVOUR) is not None:
            metadata["image_flavour"] = self.inputs.get(self.KEY_IMAGE_FLAVOUR)

        acq_contrast: str = self.inputs[self.KEY_ACQ_CONTRAST]
        echo_time: float = self.inputs[self.KEY_ECHO_TIME]
        repetition_time: float = self.inputs[self.KEY_REPETITION_TIME]

        mri_signal: np.ndarray = np.zeros(t1.shape)

        # pre-calculate the exponent exp(-echo_time/t2) as it is used multiple times
        exp_te_t2 = np.exp(
            -np.divide(echo_time, t2, out=np.zeros_like(t2), where=t2 != 0)
        )

        # pre-calculate the exponent exp(-repetition_time/t1) as it is used multiple times
        exp_tr_t1 = np.exp(
            -np.divide(repetition_time, t1, out=np.zeros_like(t1), where=t1 != 0)
        )

        # add common fields to metadata
        metadata[self.KEY_ACQ_CONTRAST] = acq_contrast
        metadata[self.KEY_ECHO_TIME] = echo_time
        metadata[self.KEY_REPETITION_TIME] = repetition_time
        metadata[
            self.KEY_ACQ_TYPE
        ] = "3D"  # 2D not currently supported so everything is 3D

        # Gradient Echo Contrast. Equation is from p246 in the book MRI from Picture to Proton,
        # second edition, 2006, McRobbie et. al.
        if acq_contrast.lower() == self.CONTRAST_GE:
            t2_star: np.ndarray = self.inputs[self.KEY_T2_STAR].image
            flip_angle = np.radians(self.inputs.get(self.KEY_EXCITATION_FLIP_ANGLE))
            # pre-calculate the exponent exp(-echo_time/t2_star)
            exp_t2_star = np.exp(
                -np.divide(
                    echo_time, t2_star, out=np.zeros_like(t2_star), where=t2_star != 0
                )
            )
            # pre-calculate the exponent exp(-repetition_time/t2)
            exp_tr_t2 = np.exp(
                -np.divide(repetition_time, t2, out=np.zeros_like(t2), where=t2 != 0)
            )

            # pre-calculate the numerator and denominator for use in np.divide to avoid runtime
            # divide-by-zero
            numerator = m0 * (1 - exp_tr_t1)
            denominator = (
                1
                - np.cos(flip_angle) * exp_tr_t1
                - exp_tr_t2 * (exp_tr_t1 - np.cos(flip_angle))
            )

            mri_signal = (
                np.sin(flip_angle)
                * (
                    np.divide(
                        numerator,
                        denominator,
                        out=np.zeros_like(denominator),
                        where=denominator != 0,
                    )
                    + mag_enc
                )
                * exp_t2_star
            )
            metadata[self.KEY_EXCITATION_FLIP_ANGLE] = self.inputs.get(
                self.KEY_EXCITATION_FLIP_ANGLE
            )

        # Spin Echo Contrast, equation is the standard spin-echo signal equation assuming a 90°
        # excitation pulse and 180° refocusing pulse. Equation is from p69 in the book
        # MRI from Picture to Proton, second edition, 2006, McRobbie et. al.
        elif acq_contrast.lower() == self.CONTRAST_SE:

            mri_signal = (
                m0
                * (
                    1
                    - np.exp(
                        -np.divide(
                            repetition_time, t1, out=np.zeros_like(t1), where=t1 != 0
                        )
                    )
                )
                + mag_enc
            ) * exp_te_t2
            # for spin echo the flip angle is assumed to be 90°
            metadata[self.KEY_EXCITATION_FLIP_ANGLE] = 90.0

        # Inversion Recovery contrast.  Equation is from equation 7 in
        # http://www.paul-tofts-phd.org.uk/talks/ismrm2009_rt.pdf
        elif acq_contrast.lower() == self.CONTRAST_IR:
            flip_angle = np.radians(self.inputs.get(self.KEY_EXCITATION_FLIP_ANGLE))
            inversion_time = self.inputs.get(self.KEY_INVERSION_TIME)
            inversion_flip_angle = np.radians(
                self.inputs.get(self.KEY_INVERSION_FLIP_ANGLE)
            )
            # pre-calculate the exponent exp(-inversion_time/t1)
            exp_ti_t1 = np.exp(
                -np.divide(inversion_time, t1, out=np.zeros_like(t1), where=t1 != 0)
            )
            numerator = m0 * (
                1
                - (1 - np.cos(inversion_flip_angle)) * exp_ti_t1
                - np.cos(inversion_flip_angle) * exp_tr_t1
            )

            denominator = (
                1 - np.cos(flip_angle) * np.cos(inversion_flip_angle) * exp_tr_t1
            )

            mri_signal = (
                np.sin(flip_angle)
                * (
                    np.divide(
                        numerator,
                        denominator,
                        out=np.zeros_like(denominator),
                        where=denominator != 0,
                    )
                    + mag_enc
                )
                * exp_te_t2
            )
            # add ir specific metadata
            metadata[self.KEY_EXCITATION_FLIP_ANGLE] = self.inputs.get(
                self.KEY_EXCITATION_FLIP_ANGLE
            )
            metadata[self.KEY_INVERSION_FLIP_ANGLE] = self.inputs.get(
                self.KEY_INVERSION_FLIP_ANGLE
            )
            metadata[self.KEY_INVERSION_TIME] = inversion_time

        self.outputs[self.KEY_IMAGE]: BaseImageContainer = self.inputs[
            self.KEY_T1
        ].clone()
        self.outputs[self.KEY_IMAGE].image = mri_signal
        # merge the metadata field with the constructed one (we don't want to merge)
        self.outputs[self.KEY_IMAGE].metadata = {
            **self.outputs[self.KEY_IMAGE].metadata,
            **metadata,
        }
        self.outputs[self.KEY_IMAGE].metadata.pop("units", None)
        self.outputs[self.KEY_IMAGE].metadata.pop("quantity", None)

    def _validate_inputs(self):
        """ Checks that the inputs meet their validation critera
        't1' must be derived from BaseImageContainer, >=0, and non-complex
        't2' must be derived from BaseImageContainer, >=0, and non-complex
        't2_star' must be derived from BaseImageContainer, >=0, and non-complex
            Only required if 'acq_contrast' == 'ge'
        'm0' must be derived from BaseImageContainer, >=0, and non-complex
        'mag_enc' (optional) must be derived from BaseImageContainer and non-complex
        'acq_contrast' must be a string and equal to "ge" or "se" (case insensitive)
        'echo_time' must be a float and >= 0
        'repetition_time' must be a float and >= 0
        'excitation_flip_angle' must be a float and >=0
        'inversion_flip_angle' must be a float and >=0
        'inversion_time' must be a float and >=0

        All images must have the same dimensions

        """
        input_validator = ParameterValidator(
            parameters={
                self.KEY_M0: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                        greater_than_equal_to_validator(0),
                    ]
                ),
                self.KEY_T1: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                        greater_than_equal_to_validator(0),
                    ]
                ),
                self.KEY_T2: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                        greater_than_equal_to_validator(0),
                    ]
                ),
                self.KEY_T2_STAR: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                        greater_than_equal_to_validator(0),
                    ],
                    optional=True,
                ),
                self.KEY_MAG_ENC: Parameter(
                    validators=[isinstance_validator(BaseImageContainer)], optional=True
                ),
                self.KEY_ACQ_CONTRAST: Parameter(
                    validators=[
                        isinstance_validator(str),
                        from_list_validator(
                            [self.CONTRAST_GE, self.CONTRAST_SE, self.CONTRAST_IR],
                            case_insensitive=True,
                        ),
                    ]
                ),
                self.KEY_ECHO_TIME: Parameter(
                    validators=[
                        isinstance_validator(float),
                        greater_than_equal_to_validator(0),
                    ]
                ),
                self.KEY_REPETITION_TIME: Parameter(
                    validators=[
                        isinstance_validator(float),
                        greater_than_equal_to_validator(0),
                    ]
                ),
                self.KEY_EXCITATION_FLIP_ANGLE: Parameter(
                    validators=[isinstance_validator(float),], optional=True,
                ),
                self.KEY_INVERSION_FLIP_ANGLE: Parameter(
                    validators=[isinstance_validator(float),], optional=True,
                ),
                self.KEY_INVERSION_TIME: Parameter(
                    validators=[
                        isinstance_validator(float),
                        greater_than_equal_to_validator(0),
                    ],
                    optional=True,
                ),
                self.KEY_IMAGE_FLAVOUR: Parameter(
                    validators=[isinstance_validator(str),], optional=True
                ),
            }
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)

        # Parameters that are conditionally required based on the value of "acq_contrast"
        # if the acquisition contrast is gradient echo ("ge")
        if self.inputs[self.KEY_ACQ_CONTRAST].lower() == self.CONTRAST_GE:
            # 't2_star' must be present in inputs
            if self.inputs.get(self.KEY_T2_STAR) is None:
                raise FilterInputValidationError(
                    "Acquisition contrast is ge, 't2_star' image required"
                )
        # if the acquisition contrast is gradient echo ("ge") or inversion recovery ("ir")
        if self.inputs[self.KEY_ACQ_CONTRAST].lower() in (
            self.CONTRAST_GE,
            self.CONTRAST_IR,
        ):
            # 'excitation_flip_angle' must be present in inputs
            if self.inputs.get(self.KEY_EXCITATION_FLIP_ANGLE) is None:
                raise FilterInputValidationError(
                    f"Acquisition contrast is {self.inputs[self.KEY_ACQ_CONTRAST]},"
                    " 'excitation_flip_angle' required"
                )

        # if the acquisition contrast is inversion recovery ("ir")
        if self.inputs[self.KEY_ACQ_CONTRAST].lower() == self.CONTRAST_IR:
            if self.inputs.get(self.KEY_INVERSION_FLIP_ANGLE) is None:
                raise FilterInputValidationError(
                    f"Acquisition contrast is {self.inputs[self.KEY_ACQ_CONTRAST]},"
                    " 'inversion_flip_angle' required"
                )
            if self.inputs.get(self.KEY_INVERSION_TIME) is None:
                raise FilterInputValidationError(
                    f"Acquisition contrast is {self.inputs[self.KEY_ACQ_CONTRAST]},"
                    " 'inversion_time' required"
                )
            if self.inputs.get(self.KEY_REPETITION_TIME) < (
                self.inputs.get(self.KEY_ECHO_TIME)
                + self.inputs.get(self.KEY_INVERSION_TIME)
            ):
                raise FilterInputValidationError(
                    "repetition_time must be greater than echo_time + inversion_time"
                )

        # Check repetition_time is not < echo_time for ge and se
        if self.inputs.get(self.KEY_REPETITION_TIME) < self.inputs.get(
            self.KEY_ECHO_TIME
        ):
            raise FilterInputValidationError(
                "repetition_time must be greater than echo_time"
            )

        # Check that all the input images are all the same dimensions
        input_keys = self.inputs.keys()
        keys_of_images = [
            key
            for key in input_keys
            if isinstance(self.inputs[key], BaseImageContainer)
        ]

        list_of_image_shapes = [self.inputs[key].shape for key in keys_of_images]
        if list_of_image_shapes.count(list_of_image_shapes[0]) != len(
            list_of_image_shapes
        ):
            raise FilterInputValidationError(
                [
                    "Input image shapes do not match.",
                    [
                        f"{keys_of_images[i]}: {list_of_image_shapes[i]}, "
                        for i in range(len(list_of_image_shapes))
                    ],
                ]
            )

        # Check that all the input images are not of image_type == "COMPLEX_IMAGE_TYPE"
        for key in keys_of_images:
            if self.inputs[key].image_type == COMPLEX_IMAGE_TYPE:
                raise FilterInputValidationError(
                    f"{key} has image type {COMPLEX_IMAGE_TYPE}, this is not supported"
                )
