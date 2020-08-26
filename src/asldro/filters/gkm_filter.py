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

KEYS_DTYPE_DICT = {
    KEY_PERFUSION_RATE: BaseImageContainer,
    KEY_TRANSIT_TIME: BaseImageContainer,
    KEY_M0: BaseImageContainer,
    KEY_LABEL_TYPE: str,
    KEY_LABEL_DURATION: float,
    KEY_SIGNAL_TIME: float,
    KEY_LABEL_EFFICIENCY: float,
    KEY_LAMBDA_BLOOD_BRAIN: float,
    KEY_T1_ARTERIAL_BLOOD: float,
}

CASL = "CASL"
PCASL = "pCASL"
PASL = "PASL"


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
        # Loop over the keys
        for key in KEYS_TUPLE:
            # Check if key is present
            if key not in self.inputs:
                raise FilterInputValidationError(f"{self} does not have defined {key}")

            if not isinstance(self.inputs[key], KEYS_DTYPE_DICT[key]):
                raise FilterInputValidationError(
                    f"{self} is not a {KEYS_DTYPE_DICT[key]} (is {type(self.inputs[key])})"
                )
