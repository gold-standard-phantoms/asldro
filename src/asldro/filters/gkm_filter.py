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
    A filter that generates the ASL signal using the General Kinetic Model

    Inputs:

    Outputs:

    """

    def __init__(self):
        super().__init__(name="General Kinetic Model")

    def _run(self):

        if self.inputs[KEY_LABEL_TYPE] == PASL:
            # do GKM for PASL
            print("General Kinetic Model for Pulsed ASL")
        elif self.inputs[KEY_LABEL_TYPE] in [CASL, PCASL]:
            # do GKM for CASL/pCASL
            print("General Kinetic Model for Continuous/pseudo-Continuous ASL")

    def _validate_inputs(self):
        """ Checks that the inputs meet their validation criteria
        'perfusion_rate' must be derived from BaseImageContainer and be >= 0
        'transit_time' must be derived from BaseImageContainer and be >= 0
        'm0' must be derived from BaseImageContainer and be >= 0
        'label_type' must be a string and equal to "CASL" OR "pCASL" OR "PASL"
        'label_duration" must be a float between 0 and 100
        'signal_time' must be a float between 0 and 100
        'label_efficiency' must be a float between 0 and 1
        'lambda_blood_brain' must be a float between 0 and 1
        't1_arterial_blood' must be a float between 0 and 100
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
            KEY_M0: (BaseImageContainer, 0, float("Inf")),
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
            if inputs_datavalid_dict[key][0] in (int, float):
                min_val = inputs_datavalid_dict[key][1]
                max_val = inputs_datavalid_dict[key][2]
                if input_value < min_val or input_value > max_val:
                    raise FilterInputValidationError(
                        f"{self} is not between {min_val} and {max_val} (is {input_value}"
                    )
            # BaseImageContainer derived
            if inputs_datavalid_dict[key][0] == BaseImageContainer:
                min_val = inputs_datavalid_dict[key][1]
                max_val = inputs_datavalid_dict[key][2]
                if (input_value.image < min_val).any() or (
                    input_value.image > max_val
                ).any():
                    raise FilterInputValidationError(
                        f"{self} has values outside of the range {min_val} to {max_val}"
                    )

            # string
            if inputs_datavalid_dict[key][0] == str:
                match_strings = inputs_datavalid_dict[key][1:]
                if input_value not in match_strings:
                    raise FilterInputValidationError(
                        f"{self} is not between {min_val} and {max_val} (is {input_value}"
                    )
