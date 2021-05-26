"""ASL quantification filter class"""


import numpy as np
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.filters.gkm_filter import GkmFilter
from asldro.containers.image import BaseImageContainer
from asldro.validators.parameters import (
    Parameter,
    ParameterValidator,
    isinstance_validator,
    greater_than_equal_to_validator,
    from_list_validator,
    range_inclusive_validator,
)


class AslQuantificationFilter(BaseFilter):
    r"""
    A filter that calculates the perfusion rate for arterial spin labelling data.

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`AslQuantificationFilter.add_input()`
    member function. They are also accessible via class constants, for example
    :class:`AslQuantificationFilter.KEY_CONTROL`

    :param 'control': the control image (3D or 4D timeseries)
    :type 'control': BaseImageContainer
    :param 'label': the label image (3D or 4D timeseries)
    :type 'label': BaseImageContainer
    :param 'm0': equilibrium magnetisation image
    :type 'm0': BaseImageContainer
    :param 'label_type': the type of labelling used: "pasl" for pulsed ASL  "pcasl" or "casl" for
        for continuous ASL.
    :type 'label_type': "str"
    :param 'lambda_blood_brain': The blood-brain-partition-coefficient (0 to 1 inclusive)
    :type 'lambda_blood_brain': float
    :param 'label_duration': The length of the labelling pulse, seconds (0 to 100 inclusive)
    :type 'label_duration': float
    :param 'post_label_delay': The duration between the end of the labelling pulse and the imaging
        excitation pulse, seconds (0 to 100 inclusive)
    :type 'post_label_delay': float
    :param 'label_efficiency': The degree of inversion of the labelling (0 to 1 inclusive)
    :type 'label_efficiency': float
    :param 't1_arterial_blood': Longitudinal relaxation time of arterial blood,
        seconds (0 exclusive to 100 inclusive)
    :type 't1_arterial_blood': float
    :param 'model': defines which model to use

        * 'whitepaper' uses the single-subtraction white paper equation

    :type 'model': str

    **Outputs**

    :param 'perfusion_rate': map of the calculated perfusion rate
    :type 'perfusion_rate': BaseImageContainer


    The following equations are used to calculate the perfusion rate, depending on the
    input ``model``:

    * If ``model`` is equal to 'whitepaper' then the equations used are those featured in the paper
      Alsop et al. Recommended implementation of arterial spin-labeled perfusion MRI for clinical
      applications: A consensus of the ISMRM perfusion study group and the European consortium for
      ASL in dementia. Magnetic Resonance in Medicine 2014;73:102–116 doi: 10.1002/mrm.25197.

        * For ``label_type`` 'pcasl' or 'casl'

        .. math::
            f = \frac{6000 \cdot\ (\text{SI}_{\text{control}} - \text{SI}_{\text{label}}) \cdot
            e^{\frac{\text{PLD}}{T_{1,b}}}}{2 \cdot \alpha \cdot T_{1,b} \cdot \text{SI}_{\text{M0}}
            \cdot (1-e^{-\frac{\tau}{T_{1,b}}})}
            \text{where:}


    """

    KEY_CONTROL = "control"
    KEY_LABEL = "label"
    KEY_MODEL = "model"
    KEY_M0 = GkmFilter.KEY_M0
    KEY_PERFUSION_RATE = GkmFilter.KEY_PERFUSION_RATE
    KEY_LABEL_TYPE = GkmFilter.KEY_LABEL_TYPE
    KEY_LABEL_DURATION = GkmFilter.KEY_LABEL_DURATION
    KEY_LABEL_EFFICIENCY = GkmFilter.KEY_LABEL_EFFICIENCY
    KEY_LAMBDA_BLOOD_BRAIN = GkmFilter.KEY_LAMBDA_BLOOD_BRAIN
    KEY_T1_ARTERIAL_BLOOD = GkmFilter.KEY_T1_ARTERIAL_BLOOD
    KEY_POST_LABEL_DELAY = GkmFilter.KEY_POST_LABEL_DELAY

    WHITEPAPER = "whitepaper"

    def __init__(self):
        super().__init__(name="ASL Quantification")

    def _run(self):
        """Calculates the perfusion rate based on the inputs"""
        # if any of the images are 4D, take the average along the 4th dimension.
        images = {}
        for key in [self.KEY_M0, self.KEY_CONTROL, self.KEY_LABEL]:
            if len(self.inputs[key].shape) == 4:
                # take the average along the 4th (time) dimension
                images[key] = np.average(self.inputs[key].image, axis=3)
            else:
                images[key] = self.inputs[key].image

        # create output image and set to zero
        self.outputs[self.KEY_PERFUSION_RATE] = self.inputs[self.KEY_CONTROL].clone()
        self.outputs[self.KEY_PERFUSION_RATE].image = np.zeros_like(
            images[self.KEY_CONTROL]
        )

        if self.inputs[self.KEY_MODEL] == self.WHITEPAPER:
            if self.inputs[self.KEY_LABEL_TYPE].lower() in [
                GkmFilter.CASL,
                GkmFilter.PCASL,
            ]:
                self.outputs[self.KEY_PERFUSION_RATE].image = self.asl_quant_wp_casl(
                    control=images[self.KEY_CONTROL],
                    label=images[self.KEY_LABEL],
                    m0=images[self.KEY_M0],
                    lambda_blood_brain=self.inputs[self.KEY_LAMBDA_BLOOD_BRAIN],
                    label_duration=self.inputs[self.KEY_LABEL_DURATION],
                    post_label_delay=self.inputs[self.KEY_POST_LABEL_DELAY],
                    label_efficiency=self.inputs[self.KEY_LABEL_EFFICIENCY],
                    t1_arterial_blood=self.inputs[self.KEY_T1_ARTERIAL_BLOOD],
                )
                self.outputs[self.KEY_PERFUSION_RATE].metadata.pop(
                    "RepetitionTime", None
                )
                self.outputs[self.KEY_PERFUSION_RATE].metadata.pop("EchoTime", None)
                self.outputs[self.KEY_PERFUSION_RATE].metadata.pop("M0", None)
                self.outputs[self.KEY_PERFUSION_RATE].metadata.pop("FlipAngle", None)
                self.outputs[self.KEY_PERFUSION_RATE].metadata["asl_context"] = "cbf"
                self.outputs[self.KEY_PERFUSION_RATE].metadata["Units"] = "ml/100g/min"
                self.outputs[self.KEY_PERFUSION_RATE].metadata["ImageType"] = [
                    "DERIVED",
                    "PRIMARY",
                    "PERFUSION",
                    "RCBF",
                ]

    def _validate_inputs(self):
        """Checks the inputs meet their validation criteria
        'control' must be derived from BaseImageContainer
        'label' must be derived from BaseImageContainer
        'm0' must be derived from BaseImageContainer
        'label_type' must be a str and equal to "pasl", "casl" or "pcasl" (case-insensitive)
        'model' must be a str and equal to "whitepaper"
        'label_duration' must be a float and >= 0
        'post_label_delay' must be a float and >= 0
        'label_efficiency' must be a float between 0 and 1 inclusive
        'lambda_blood_brain' must be a float between 0 and 1 inclusive
        't1_arterial_blood' must be a float and >=0
        """

        input_validator = ParameterValidator(
            parameters={
                self.KEY_M0: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                    ]
                ),
                self.KEY_CONTROL: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                    ]
                ),
                self.KEY_LABEL: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                    ]
                ),
                self.KEY_LABEL_TYPE: Parameter(
                    validators=from_list_validator(
                        [GkmFilter.CASL, GkmFilter.PCASL, GkmFilter.PASL],
                        case_insensitive=True,
                    )
                ),
                self.KEY_LABEL_DURATION: Parameter(
                    validators=[
                        greater_than_equal_to_validator(0),
                        isinstance_validator(float),
                    ]
                ),
                self.KEY_POST_LABEL_DELAY: Parameter(
                    validators=[
                        greater_than_equal_to_validator(0),
                        isinstance_validator(float),
                    ]
                ),
                self.KEY_LABEL_EFFICIENCY: Parameter(
                    validators=[
                        range_inclusive_validator(0, 1),
                        isinstance_validator(float),
                    ]
                ),
                self.KEY_LAMBDA_BLOOD_BRAIN: Parameter(
                    validators=[
                        range_inclusive_validator(0, 1),
                        isinstance_validator(float),
                    ]
                ),
                self.KEY_T1_ARTERIAL_BLOOD: Parameter(
                    validators=[
                        greater_than_equal_to_validator(0),
                        isinstance_validator(float),
                    ]
                ),
                self.KEY_MODEL: Parameter(
                    validators=from_list_validator(
                        [self.WHITEPAPER],
                        case_insensitive=True,
                    )
                ),
            }
        )

        input_validator.validate(self.inputs, error_type=FilterInputValidationError)

        # Check that all the input images have the same first 3 dimensions (can differ in the 4th)
        input_keys = self.inputs.keys()
        keys_of_images = [
            key
            for key in input_keys
            if isinstance(self.inputs[key], BaseImageContainer)
        ]

        list_of_image_shapes = [self.inputs[key].shape[:3] for key in keys_of_images]
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

    @staticmethod
    def asl_quant_wp_casl(
        control: np.ndarray,
        label: np.ndarray,
        m0: np.ndarray,
        lambda_blood_brain: float,
        label_duration: float,
        post_label_delay: float,
        label_efficiency: float,
        t1_arterial_blood: float,
    ) -> np.ndarray:
        """
        Performs ASL quantification using the White Paper equation for continuous ASL

        :param control: [description]
        :type control: np.ndarray
        :param label: [description]
        :type label: np.ndarray
        :param m0: [description]
        :type m0: np.ndarray
        :param lambda_blood_brain: [description]
        :type lambda_blood_brain: float
        :param label_duration: [description]
        :type label_duration: float
        :param post_label_delay: [description]
        :type post_label_delay: float
        :param label_efficiency: [description]
        :type label_efficiency: float
        :param t1_arterial_blood: [description]
        :type t1_arterial_blood: float
        :return: [description]
        :rtype: np.ndarray
        """
        return np.divide(
            6000
            * lambda_blood_brain
            * (control - label)
            * np.exp(post_label_delay / t1_arterial_blood),
            2
            * label_efficiency
            * t1_arterial_blood
            * m0
            * (1 - np.exp(-label_duration / t1_arterial_blood)),
            out=np.zeros_like(m0),
            where=m0 != 0,
        )
