""" General Kinetic Model Filter """

import numpy as np
from asldro.containers.image import BaseImageContainer
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError

KEY_PERFUSION_RATE = "perfusion_rate"
KEY_TRANSIT_TIME = "transit_time"
KEY_M0 = "m0"
KEY_LABEL_TYPE = "label_type"
KEY_LABEL_DURATION = "label_duration"
KEY_SIGNAL_TIME = "signal_time"
KEY_LABEL_EFFICIENCY = "label_efficiency"
KEY_LAMBDA_BLOOD_BRAIN = "lambda_blood_brain"
KEY_T1_ARTERIAL_BLOOD = "t1_arterial_blood"
KEY_DELTA_M = "delta_m"

CASL = "CASL"
PCASL = "pCASL"
PASL = "PASL"

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
)


class GkmFilter(BaseFilter):
    """
    A filter that generates the ASL signal using the General Kinetic Model.

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
    Outputs:
        'delta_m' (BaseImageContainer): An image with synthetic ASL perfusion contrast

    """

    def __init__(self):
        super().__init__(name="General Kinetic Model")

    def _run(self):
        """ Generates the delta_m signal based on the inputs """

        perfusion_rate: np.ndarray = self.inputs[KEY_PERFUSION_RATE].image / 6000.0
        transit_time: np.ndarray = self.inputs[KEY_TRANSIT_TIME].image

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

        t1_prime: np.ndarray = 1 / (
            1 / t1_arterial_blood + perfusion_rate / lambda_blood_brain
        )

        # create boolean masks for each of the states of the delivery curve
        condition_bolus_not_arrived = 0 < signal_time <= transit_time
        condition_bolus_arriving = (transit_time < signal_time) & (
            signal_time < transit_time + label_duration
        )
        condition_bolus_arrived = signal_time >= transit_time + label_duration

        delta_m = np.zeros(perfusion_rate.shape)

        if self.inputs[KEY_LABEL_TYPE] == PASL:
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

            # combine the different arrival states into delta_m
            delta_m[condition_bolus_not_arrived] = 0.0
            delta_m[condition_bolus_arriving] = delta_m_arriving[
                condition_bolus_arriving
            ]
            delta_m[condition_bolus_arrived] = delta_m_arrived[condition_bolus_arrived]

        elif self.inputs[KEY_LABEL_TYPE] in [CASL, PCASL]:
            # do GKM for CASL/pCASL
            print("General Kinetic Model for Continuous/pseudo-Continuous ASL")

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

        # Dictionary for data validation.  Each entry has key corresponding to
        # the inputs keyword for the filter, then the value is a tuple with:
        # [0] - the data type
        # [1:] - the data range.
        #       - for numeric types this is the min and max (inclusive)
        #       - for BaseImageContainer this is the min and max of the .image (inclusive)
        #       = for strings this is a one or more strings to match against
        inputs_datavalid_dict = {
            KEY_PERFUSION_RATE: (BaseImageContainer, 0, float("Inf")),
            KEY_TRANSIT_TIME: (BaseImageContainer, 0, float("Inf")),
            KEY_M0: ((BaseImageContainer, float), 0, float("Inf")),
            KEY_LABEL_TYPE: (str, CASL, PCASL, PASL),
            KEY_LABEL_DURATION: (float, 0, 100),
            KEY_SIGNAL_TIME: (float, 0, 100),
            KEY_LABEL_EFFICIENCY: (float, 0, 1),
            KEY_LAMBDA_BLOOD_BRAIN: (float, 0, 1),
            KEY_T1_ARTERIAL_BLOOD: (float, 0, 100),
        }
        # Loop over the keys
        for key in KEYS_TUPLE:
            # Check if key is present
            if key not in self.inputs:
                raise FilterInputValidationError(f"{self} does not have defined {key}")

            # Key present, check data type
            input_value = self.inputs[key]
            if not isinstance(input_value, inputs_datavalid_dict[key][0]):
                raise FilterInputValidationError(
                    f"{self} is not a {inputs_datavalid_dict[key][0]} (is {type(input_value)})"
                )

            # Data type OK, check value is within limits
            # float or int
            if isinstance(input_value, (int, float)):
                min_val = inputs_datavalid_dict[key][1]
                max_val = inputs_datavalid_dict[key][2]
                if input_value < min_val or input_value > max_val:
                    raise FilterInputValidationError(
                        f"{self} is not between {min_val} and {max_val} (is {input_value}"
                    )
            # BaseImageContainer derived
            if isinstance(input_value, BaseImageContainer):
                min_val = inputs_datavalid_dict[key][1]
                max_val = inputs_datavalid_dict[key][2]
                if (input_value.image < min_val).any() or (
                    input_value.image > max_val
                ).any():
                    raise FilterInputValidationError(
                        f"{self} has values outside of the range {min_val} to {max_val}"
                    )

            # string
            if isinstance(input_value, str):
                match_strings = inputs_datavalid_dict[key][1:]
                if input_value not in match_strings:
                    raise FilterInputValidationError(
                        f"{self} is not between {min_val} and {max_val} (is {input_value}"
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
