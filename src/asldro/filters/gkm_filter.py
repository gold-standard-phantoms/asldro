""" General Kinetic Model Filter """

import numpy as np
from asldro.containers.image import BaseImageContainer
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.validators.parameters import (
    ParameterValidator,
    Parameter,
    range_inclusive_validator,
    greater_than_equal_to_validator,
    from_list_validator,
    isinstance_validator,
)

KEY_PERFUSION_RATE = "perfusion_rate"
KEY_TRANSIT_TIME = "transit_time"
KEY_M0 = "m0"
KEY_LABEL_TYPE = "label_type"
KEY_LABEL_DURATION = "label_duration"
KEY_SIGNAL_TIME = "signal_time"
KEY_LABEL_EFFICIENCY = "label_efficiency"
KEY_LAMBDA_BLOOD_BRAIN = "lambda_blood_brain"
KEY_T1_ARTERIAL_BLOOD = "t1_arterial_blood"
KEY_T1_TISSUE = "t1_tissue"
KEY_DELTA_M = "delta_m"

CASL = "casl"
PCASL = "pcasl"
PASL = "pasl"

KEYS_TUPLE = (
    KEY_PERFUSION_RATE,
    KEY_TRANSIT_TIME,
    KEY_M0,
    KEY_LABEL_TYPE,
    KEY_LABEL_DURATION,
    KEY_SIGNAL_TIME,
    KEY_LABEL_EFFICIENCY,
    KEY_LAMBDA_BLOOD_BRAIN,
    KEY_T1_ARTERIAL_BLOOD,
    KEY_T1_TISSUE,
)


class GkmFilter(BaseFilter):
    """
    A filter that generates the ASL signal using the General Kinetic Model.
    From: Buxton et. al, 'A general kinetic model for quantitative perfusion imaging with arterial
    spin labeling', Magnetic Resonance in Medicine, vol. 40, no. 3, pp. 383-396, 1998.
    https://doi.org/10.1002/mrm.1910400308

    Inputs:
        'perfusion_rate' (BaseImageContainer): Map of perfusion rate, in ml/100g/min (>=0)
        'transit_time' (BaseImageContainer):  Map of the time taken for the labelled bolus
        to reach the voxel, seconds (>=0).
        'm0' (BaseImageContainer or float): The equilibrium magnetisation, can be a map or
        single value (>=0).
        'label_type' (str): Determined which GKM equations to use:
             "casl" OR "pcasl" (case insensitive) for the continuous model
             "pasl" (case insensitive) for the pulsed model
        'label_duration' (float): The length of the labelling pulse, seconds (0 to 100 inclusive)
        'signal_time' (float): The time after labelling commences to generate signal,
        seconds (0 to 100 inclusive)
        'label_efficiency' (float): The degree of inversion of the labelling (0 to 1 inclusive)
        'lambda_blood_brain' (float): The blood-brain-partition-coefficient (0 to 1 inclusive)
        't1_arterial_blood' (float): Longitudinal relaxation time of arterial blood,
        seconds (>0, to 100)
        't1_tissue' (BaseImageContainer): Longitudinal relaxation time of the tissue,
        seconds (>0, to 100)

    Outputs:
        'delta_m' (BaseImageContainer): An image with synthetic ASL perfusion contrast. This will
        be the same class as the input 'perfusion_rate'

    """

    def __init__(self):
        super().__init__(name="General Kinetic Model")

    def _run(self):
        """ Generates the delta_m signal based on the inputs """

        perfusion_rate: np.ndarray = self.inputs[KEY_PERFUSION_RATE].image / 6000.0
        transit_time: np.ndarray = self.inputs[KEY_TRANSIT_TIME].image
        t1_tissue: np.ndarray = self.inputs[KEY_T1_TISSUE].image

        label_duration: float = self.inputs[KEY_LABEL_DURATION]
        signal_time: float = self.inputs[KEY_SIGNAL_TIME]
        label_efficiency: float = self.inputs[KEY_LABEL_EFFICIENCY]
        lambda_blood_brain: float = self.inputs[KEY_LAMBDA_BLOOD_BRAIN]
        t1_arterial_blood: float = self.inputs[KEY_T1_ARTERIAL_BLOOD]

        # if m0 is an image load that, if not then make a ndarray
        # with the same value (makes the calculations more straightforward)
        if isinstance(self.inputs[KEY_M0], BaseImageContainer):
            m0: np.ndarray = self.inputs[KEY_M0].image
        else:
            m0: np.ndarray = self.inputs[KEY_M0] * np.ones(perfusion_rate.shape)

        t1_prime: np.ndarray = 1 / (1 / t1_tissue + perfusion_rate / lambda_blood_brain)

        # create boolean masks for each of the states of the delivery curve
        condition_bolus_not_arrived = 0 < signal_time <= transit_time
        condition_bolus_arriving = (transit_time < signal_time) & (
            signal_time < transit_time + label_duration
        )
        condition_bolus_arrived = signal_time >= transit_time + label_duration

        delta_m = np.zeros(perfusion_rate.shape)

        if self.inputs[KEY_LABEL_TYPE].lower() == PASL:
            # do GKM for PASL
            print("General Kinetic Model for Pulsed ASL")
            k: np.ndarray = (1 / t1_arterial_blood - 1 / t1_prime)
            q_pasl_arriving = (
                np.exp(k * signal_time)
                * (np.exp(-k * transit_time) - np.exp(-k * signal_time))
                / (k * (signal_time - transit_time))
            )
            q_pasl_arrived = (
                np.exp(k * signal_time)
                * (
                    np.exp(-k * transit_time)
                    - np.exp(-k * (transit_time + label_duration))
                )
                / (k * label_duration)
            )

            delta_m_arriving = (
                2
                * m0
                * perfusion_rate
                * (signal_time - transit_time)
                * label_efficiency
                * np.exp(-signal_time / t1_arterial_blood)
                * q_pasl_arriving
            )
            delta_m_arrived = (
                2
                * m0
                * perfusion_rate
                * label_efficiency
                * label_duration
                * np.exp(-signal_time / t1_arterial_blood)
                * q_pasl_arrived
            )

        elif self.inputs[KEY_LABEL_TYPE].lower() in [CASL, PCASL]:
            # do GKM for CASL/pCASL
            print("General Kinetic Model for Continuous/pseudo-Continuous ASL")
            q_ss_arriving = 1 - np.exp(-(signal_time - transit_time) / t1_prime)
            q_ss_arrived = 1 - np.exp(-label_duration / t1_prime)

            delta_m_arriving = (
                2
                * m0
                * perfusion_rate
                * t1_prime
                * label_efficiency
                * np.exp(-transit_time / t1_arterial_blood)
                * q_ss_arriving
            )
            delta_m_arrived = (
                2
                * m0
                * perfusion_rate
                * t1_prime
                * label_efficiency
                * np.exp(-transit_time / t1_arterial_blood)
                * np.exp(-(signal_time - label_duration - transit_time) / t1_prime)
                * q_ss_arrived
            )

        # combine the different arrival states into delta_m
        delta_m[condition_bolus_not_arrived] = 0.0
        delta_m[condition_bolus_arriving] = delta_m_arriving[condition_bolus_arriving]
        delta_m[condition_bolus_arrived] = delta_m_arrived[condition_bolus_arrived]

        # copy 'perfusion_rate' image container and set the image to delta_m
        self.outputs[KEY_DELTA_M]: BaseImageContainer = self.inputs[
            KEY_PERFUSION_RATE
        ].clone()
        self.outputs[KEY_DELTA_M].image = delta_m

    def _validate_inputs(self):
        """ Checks that the inputs meet their validation criteria
        'perfusion_rate' must be derived from BaseImageContainer and be >= 0
        'transit_time' must be derived from BaseImageContainer and be >= 0
        'm0' must be either a float or derived from BaseImageContainer and be >= 0
        'label_type' must be a string and equal to "CASL" OR "pCASL" OR "PASL"
        'label_duration" must be a float between 0 and 100
        'signal_time' must be a float between 0 and 100
        'label_efficiency' must be a float between 0 and 1
        'lambda_blood_brain' must be a float between 0 and 1
        't1_arterial_blood' must be a float between 0 and 100

        all BaseImageContainers supplied should be the same dimensions
         """
        input_validator = ParameterValidator(
            parameters={
                KEY_PERFUSION_RATE: Parameter(
                    validators=[
                        greater_than_equal_to_validator(0),
                        isinstance_validator(BaseImageContainer),
                    ]
                ),
                KEY_TRANSIT_TIME: Parameter(
                    validators=[
                        greater_than_equal_to_validator(0),
                        isinstance_validator(BaseImageContainer),
                    ]
                ),
                KEY_M0: Parameter(
                    validators=[
                        greater_than_equal_to_validator(0),
                        isinstance_validator((BaseImageContainer, float)),
                    ]
                ),
                KEY_T1_TISSUE: Parameter(
                    validators=[
                        range_inclusive_validator(0, 100),
                        isinstance_validator(BaseImageContainer),
                    ]
                ),
                KEY_LABEL_TYPE: Parameter(
                    validators=from_list_validator(
                        [CASL, PCASL, PASL], case_insensitive=True
                    ),
                ),
                KEY_LABEL_DURATION: Parameter(
                    validators=[
                        range_inclusive_validator(0, 100),
                        isinstance_validator(float),
                    ]
                ),
                KEY_SIGNAL_TIME: Parameter(
                    validators=[
                        range_inclusive_validator(0, 100),
                        isinstance_validator(float),
                    ]
                ),
                KEY_LABEL_EFFICIENCY: Parameter(
                    validators=[
                        range_inclusive_validator(0, 1),
                        isinstance_validator(float),
                    ]
                ),
                KEY_LAMBDA_BLOOD_BRAIN: Parameter(
                    validators=[
                        range_inclusive_validator(0, 1),
                        isinstance_validator(float),
                    ]
                ),
                KEY_T1_ARTERIAL_BLOOD: Parameter(
                    validators=[
                        range_inclusive_validator(0, 100),
                        isinstance_validator(float),
                    ]
                ),
            }
        )

        input_validator.validate(self.inputs, error_type=FilterInputValidationError)

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
