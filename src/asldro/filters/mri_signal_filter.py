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

KEY_T1 = "t1"
KEY_T2 = "t2"
KEY_T2_STAR = "t2_star"
KEY_M0 = "m0"
KEY_MAG_ENC = "mag_enc"
KEY_ACQ_CONTRAST = "acq_contrast"
KEY_ACQ_TE = "acq_te"
KEY_ACQ_TR = "acq_tr"
KEY_IMAGE = "image"

CONTRAST_GE = "ge"
CONTRAST_SE = "se"


class MriSignalFilter(BaseFilter):
    """ A filter that generates either the Gradient Echo or Spin Echo MRI signal. The
    Gradient Echo assumes a perfect 90° excitation, and the Spin Echo perfect 90° excitation
    and 180° refocusing pulses.

    ### Inputs:
        't1' (BaseImageContainer): Longitudinal relaxation time in seconds (>=0, non-complex data)
        't2' (BaseImageContainer): Transverse relaxation time in seconds (>=0, non-complex data)
        't2_star' (BaseImageContainer): Transverse relaxation time including time-invariant magnetic
        field inhomogeneities, only required for gradient echo (>=0, non-complex data)
        'm0' (BaseImageContainer): Equilibrium magnetisation (>=0, non-complex data)
        'mag_eng' (BaseImageContainer), optional: Added to M0 before relaxation is calculated,
        provides a means to encode another signal into the MRI signal (non-complex data)
        'acq_contrast' (str): Determines which signal model to use:
            "ge" (case insensitive) for Gradient Echo
            "se" (case insensitive) for Spin Echo
        'acq_te' (float): The echo time in seconds (>=0)
        'acq_tr' (float): The repeat time in seconds (>=0)

    ### Output:
        'image' (BaseImageContainer): An image with the MRI signal.  This will be the same class
        as the input 't1'
        
    """

    def __init(self):
        super().__init__(name="MRI Signal Model")

    def _run(self):

        t1: np.ndarray = self.inputs[KEY_T1].image
        t2: np.ndarray = self.inputs[KEY_T2].image
        m0: np.ndarray = self.inputs[KEY_M0].image
        if self.inputs.get(KEY_T2_STAR) is not None:
            t2_star: np.ndarray = self.inputs[KEY_T2_STAR].image
        if self.inputs.get(KEY_MAG_ENC) is not None:
            mag_enc: np.ndarray = self.inputs[KEY_MAG_ENC].image
        else:
            mag_enc: np.ndarray = np.zeros(t1.shape)
        acq_contrast: str = self.inputs[KEY_ACQ_CONTRAST]
        acq_te: float = self.inputs[KEY_ACQ_TE]
        acq_tr: float = self.inputs[KEY_ACQ_TR]

        mri_signal: np.ndarray = np.zeros(t1.shape)

        if acq_contrast.lower() == CONTRAST_GE:
            # pre-calculate the exponent exp(-acq_te/t2_star) as it is used multiple times
            exp_t2_star = np.exp(
                -np.divide(
                    acq_te, t2_star, out=np.zeros_like(t2_star), where=t2_star != 0
                )
            )
            mri_signal = (
                m0
                * (
                    1
                    - np.exp(
                        -np.divide(acq_tr, t1, out=np.zeros_like(t1), where=t1 != 0)
                    )
                )
                + mag_enc
            ) * exp_t2_star

        elif acq_contrast.lower() == CONTRAST_SE:
            # pre-calculate the exponent exp(-acq_te/t2) as it is used multiple times
            exp_t2 = np.exp(
                -np.divide(acq_te, t2, out=np.zeros_like(t2), where=t2 != 0)
            )
            mri_signal = (
                m0
                * (
                    1
                    - np.exp(
                        -np.divide(acq_tr, t1, out=np.zeros_like(t1), where=t1 != 0)
                    )
                )
                + mag_enc
            ) * exp_t2

        self.outputs[KEY_IMAGE]: BaseImageContainer = self.inputs[KEY_T1].clone()
        self.outputs[KEY_IMAGE].image = mri_signal

    def _validate_inputs(self):
        """ Checks that the inputs meet their validation critera
        't1' must be derived from BaseImageContainer, >=0, and non-complex
        't2' must be derived from BaseImageContainer, >=0, and non-complex
        't2_star' must be derived from BaseImageContainer, >=0, and non-complex
        Only required is 'acq_contrast' == 'ge'
        'm0' must be derived from BaseImageContainer, >=0, and non-complex
        'mag_enc' (optional) must be derived from BaseImageContainer and non-complex
        'acq_type' must be a string and equal to "2d" or "3d" (case insensitive)
        'acq_contrast' must be a string and equal to "ge" or "se" (case insensitive)
        'acq_te' must be a float and >= 0
        'acq_tr' must be a float and >= 0

        All images must have the same dimensions

        2D is not currently supported, so if 'acq_type' == "2d" a FilterInputValidationError
        will be raised

        """
        input_validator = ParameterValidator(
            parameters={
                KEY_M0: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                        greater_than_equal_to_validator(0),
                    ]
                ),
                KEY_T1: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                        greater_than_equal_to_validator(0),
                    ]
                ),
                KEY_T2: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                        greater_than_equal_to_validator(0),
                    ]
                ),
                KEY_T2_STAR: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                        greater_than_equal_to_validator(0),
                    ],
                    optional=True,
                ),
                KEY_MAG_ENC: Parameter(
                    validators=[isinstance_validator(BaseImageContainer),],
                    optional=True,
                ),
                KEY_ACQ_CONTRAST: Parameter(
                    validators=[
                        isinstance_validator(str),
                        from_list_validator(
                            [CONTRAST_GE, CONTRAST_SE], case_insensitive=True
                        ),
                    ]
                ),
                KEY_ACQ_TE: Parameter(
                    validators=[
                        isinstance_validator(float),
                        greater_than_equal_to_validator(0),
                    ]
                ),
                KEY_ACQ_TR: Parameter(
                    validators=[
                        isinstance_validator(float),
                        greater_than_equal_to_validator(0),
                    ]
                ),
            }
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)

        # if the acquisition contrast is gradient echo ("ge") then 't2_star' must be present
        # in inputs
        if self.inputs[KEY_ACQ_CONTRAST] == CONTRAST_GE:
            if self.inputs.get(KEY_T2_STAR) is None:
                raise FilterInputValidationError(
                    "Acquisition contrast is ge, 't2_star' image required"
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
