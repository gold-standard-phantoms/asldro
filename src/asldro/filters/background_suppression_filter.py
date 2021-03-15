"""Background Suppression Filter"""

from re import M, error
import numpy as np
from scipy.optimize import minimize, OptimizeResult
import pdb
from asldro.containers.image import BaseImageContainer, COMPLEX_IMAGE_TYPE
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.validators.parameters import (
    from_list_validator,
    isinstance_validator,
    greater_than_equal_to_validator,
    greater_than_validator,
    for_each_validator,
    Parameter,
    ParameterValidator,
    range_inclusive_validator,
)


class BackgroundSuppressionFilter(BaseFilter):
    """A filter that simulates a background suppression
    pulse sequence on longitudinal magnetisation. It can either
    use explicitly supplied pulse timings, or calculate optimised
    pulse timings for specified T1s.


    **Inputs**

    Input Parameters are all keyword arguments for the
    :class:`BackgroundSuppressionFilter.add_inputs()` member function.
    They are also accessible via class constants,
    for example :class:`CombineTimeSeriesFilter.KEY_T1`.

    :param 'mag_z': Image of the initial longitudinal magnetisation.
      Image data must not be a complex data type.
    :type 'mag_z': BaseImageContainer
    :param 't1': Image of the longitudinal relaxation time. Image
      data must be greater than 0 and non-compex. Also its shape should
      match the shape of ``'mag_z'``.
    :type 't1': BaseImageContainer
    :param 'sat_pulse_time': The time, in seconds between the saturation
      pulse and the imaging excitation pulse. Must be greater than 0.
    :type 'sat_pulse_time': float
    :param 'inv_pulse_times': The inversion times for each inversion pulse,
      defined as the spacing between the inversion pulse and the imaging
      excitation pulse. Must be greater than 0. If omitted then optimal
      inversion times will be calculated for ``'num_inv'`` number
      of pulses, and the T1 times given by ``'t1_opt'``.
    :type 'inv_pulse_times': list[float], optional
    :param 't1_opt': T1 times, in seconds to optimise the pulse inversion
      times for. Each must be greater than 0, and this parameter must be
      present if ``'inv_pulse_times'`` is omitted.
    :type 't1_opt': list[float]
    :param 'mag_time': The time, in seconds after the saturation pulse to
      sample the longitudinal magnetisation. The output magnetisation will
      only reflect the pulses that will have run by this time. Must be 
      greater than 0. If omitted, defaults to the same value
      as ``'sat_pulse_time'``.
    :type 'mag_time': float
    :param 'num_inv_pulses': The number of inversion pulses to calculate 
      optimised timings for. Must be greater than 0, and this parameter
      must be present if ``'inv_pulse_times'`` is omitted.
    :type 'num_inv_pulses: int
    :param 'pulse_efficiency': Defines the efficiency of the inversion
      pulses. Can take the values:
      
        :'realistic': Pulse efficiencies are calculated according to a
          model based on the T1. See 
          :class:`BackgroundSuppressionFilter.calculate_pulse_efficiency`
          for details on implementation.
        :'ideal': Inversion pulses are 100% efficient.
        :-1 to 0: The efficiency is defined explicitly, with -1 being full
          inversion and 0 no inversion.

    :type 'pulse_efficiency': str or float

    **Outputs**

    Once run, the filter will populate the dictionary 
    :class:`BackgroundSuppressionFilter.outputs` with
    the following entries:

    :param 'mag_z': The longitudinal magnetisation at t=mag_time.
    :type 'mag_z': BaseImageContainer
    :param 'inv_pulse_times': The inversion pulse timings.
    :type 'inv_pulse_times': list[float]

    **Metadata**
    
    The following metadata entries will be appended to the metadata
    property of the output ``'mag_z'``:

        :background_suppression: ``True``
        :background_suppression_inv_pulse_timing: ``'inv_pulse_times'``
        :background_suppression_sat_pulse_timing: ``'sat_pulse_time'``
        :background_suppression_num_pulses: The number of inversion pulses.

    
    **Background Suppression Model**

    Details on the model implemented can be found in
    :class:`BackgroundSuppressionFilter.calculate_mz`

    Details on how the pulse timings are optimised can be found in
    :class:`BackgroundSuppressionFilter.optimise_inv_pulse_times`

    """

    KEY_MAG_Z = "mag_z"
    KEY_T1 = "t1"
    KEY_SAT_PULSE_TIME = "sat_pulse_time"
    KEY_INV_PULSE_TIMES = "inv_pulse_times"
    KEY_T1_OPT = "t1_opt"
    KEY_MAG_TIME = "mag_time"
    KEY_NUM_INV_PULSES = "num_inv_pulses"
    KEY_PULSE_EFFICIENCY = "pulse_efficiency"

    M_BACKGROUND_SUPPRESSION = "background_suppression"
    M_BSUP_INV_PULSE_TIMING = "background_suppression_inv_pulse_timing"
    M_BSUP_SAT_PULSE_TIMING = "background_suppression_sat_pulse_timing"
    M_BSUP_NUM_PULSES = "background_suppression_num_pulses"

    EFF_IDEAL = "ideal"
    EFF_REALISTIC = "realistic"

    def __init__(self):
        super().__init__(name="Background Suppression Filter")

    def _run(self):
        """Runs the filter
        """
        mag_z: BaseImageContainer = self.inputs[self.KEY_MAG_Z]
        t1: BaseImageContainer = self.inputs[self.KEY_T1]
        sat_pulse_time = self.inputs[self.KEY_SAT_PULSE_TIME]

        if self.outputs.get(self.KEY_MAG_TIME) is None:
            mag_time = sat_pulse_time
        else:
            mag_time = self.outputs.get(self.KEY_MAG_TIME)

        # determine the pulse efficiency mode
        if self.inputs[self.KEY_PULSE_EFFICIENCY] == self.EFF_IDEAL:
            inv_eff = -1.0
        elif self.inputs[self.KEY_PULSE_EFFICIENCY] == self.EFF_REALISTIC:
            # pulse efficiency calculation with static method
            inv_eff = self.calculate_pulse_efficiency(t1.image)
        elif isinstance(self.inputs[self.KEY_PULSE_EFFICIENCY], float):
            inv_eff = self.inputs[self.KEY_PULSE_EFFICIENCY]

        # determine whether the inversion pulse times have been provided
        # or if optimised times need to be calculated
        if self.inputs.get(self.KEY_INV_PULSE_TIMES) is None:
            # calculation required: minimise the least squares problem
            # argmin(||Mz(t=sat_pulse_time)||^2) to find the inversion pulse
            # times for the given T1's
            t1_opt = self.inputs[self.KEY_T1_OPT]
            num_inv_pulses = self.inputs[self.KEY_NUM_INV_PULSES]
            # if `pulse_efficiency` is 'realistic' then calculate the pulse
            # efficiencies for the t1's to optimise over
            if self.inputs[self.KEY_PULSE_EFFICIENCY] == self.EFF_REALISTIC:
                pulse_eff_opt = self.calculate_pulse_efficiency(t1_opt)
            else:
                # otherwise just use pulse_eff
                pulse_eff_opt = inv_eff

            result = self.optimise_inv_pulse_times(
                sat_pulse_time, t1_opt, pulse_eff_opt, num_inv_pulses
            )
            inv_pulse_times = result.x
        else:
            inv_pulse_times = self.inputs[self.KEY_INV_PULSE_TIMES]

        # calculate the longitudinal magnetisation at mag_time based on
        # the inversion pulse times
        self.outputs[self.KEY_MAG_Z] = mag_z.clone()
        self.outputs[self.KEY_MAG_Z].image = self.calculate_mz(
            self.outputs[self.KEY_MAG_Z].image,
            t1.image,
            inv_pulse_times,
            sat_pulse_time,
            mag_time,
            inv_eff,
        )
        metadata = {
            self.M_BACKGROUND_SUPPRESSION: True,
            self.M_BSUP_INV_PULSE_TIMING: inv_pulse_times,
            self.M_BSUP_SAT_PULSE_TIMING: mag_time,
            self.M_BSUP_NUM_PULSES: np.asarray(inv_pulse_times).size,
        }
        # merge the metadata
        self.outputs[self.KEY_MAG_Z].metadata = {
            **self.outputs[self.KEY_MAG_Z].metadata,
            **metadata,
        }

        self.outputs[self.KEY_INV_PULSE_TIMES] = inv_pulse_times

    def _validate_inputs(self):
        """Checks that the inputs meet their validation criteria
        'mag_z': BaseImageContainer and image_type != COMPLEX_IMAGE_TYPE
        't1': BaseImageContainer, >0 and image_type != COMPLEX_IMAGE_TYPE, shape should
        match
        'sat_pulse_time': float, >0
        'inv_pulse_times': list[float], each >0, optional,
        't1_opt': list[float], each >0,  must be present if
          'inv_pulse_times' is omitted
        'mag_time': float, >0, optional
        'num_inv_pulses': int, >0,  must be present if
            'inv_pulse_times' is omitted
        'pulse_efficiency': float or str, optional, default "ideal":
            str: "realistic" or "ideal"
            float: between -1 and 0 inclusive
        """

        input_validator = ParameterValidator(
            parameters={
                self.KEY_MAG_Z: Parameter(
                    validators=[isinstance_validator(BaseImageContainer)]
                ),
                self.KEY_T1: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                        greater_than_equal_to_validator(0),
                    ]
                ),
                self.KEY_SAT_PULSE_TIME: Parameter(
                    validators=[isinstance_validator(float), greater_than_validator(0),]
                ),
                self.KEY_INV_PULSE_TIMES: Parameter(
                    validators=[
                        for_each_validator(greater_than_validator(0)),
                        for_each_validator(isinstance_validator(float)),
                    ],
                    optional=True,
                ),
                self.KEY_MAG_TIME: Parameter(
                    validators=[
                        isinstance_validator(float),
                        greater_than_validator(0),
                    ],
                    optional=True,
                ),
                self.KEY_PULSE_EFFICIENCY: Parameter(
                    validators=[isinstance_validator((str, float))],
                    default_value=self.EFF_IDEAL,
                ),
            }
        )

        # check the images - shapes must match and the data cannot be complex
        new_params = input_validator.validate(
            self.inputs, error_type=FilterInputValidationError
        )

        # merge the updated parameters from the output with the input parameters
        self.inputs = {**self._i, **new_params}

        keys_of_images = [self.KEY_MAG_Z, self.KEY_T1]

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

        # validate 'pulse_efficiency' depending on whether it is a float or str
        if isinstance(self.inputs[self.KEY_PULSE_EFFICIENCY], str):
            pulse_eff_validator = ParameterValidator(
                parameters={
                    self.KEY_PULSE_EFFICIENCY: Parameter(
                        validators=[
                            from_list_validator([self.EFF_REALISTIC, self.EFF_IDEAL])
                        ]
                    )
                }
            )
            pulse_eff_validator.validate(
                self.inputs, error_type=FilterInputValidationError
            )
        elif isinstance(self.inputs[self.KEY_PULSE_EFFICIENCY], float):
            pulse_eff_validator = ParameterValidator(
                parameters={
                    self.KEY_PULSE_EFFICIENCY: Parameter(
                        validators=[range_inclusive_validator(-1, 0)]
                    )
                }
            )
            pulse_eff_validator.validate(
                self.inputs, error_type=FilterInputValidationError
            )

        # validate parameters for the case where 'inv_pulse_times' is not present,
        # therefore the optimal pulse times need to be calculated
        if self.inputs.get(self.KEY_INV_PULSE_TIMES) is None:
            calc_inv_times_validator = ParameterValidator(
                parameters={
                    self.KEY_T1_OPT: Parameter(
                        validators=[
                            for_each_validator(greater_than_validator(0)),
                            for_each_validator(isinstance_validator(float)),
                        ],
                    ),
                    self.KEY_NUM_INV_PULSES: Parameter(
                        validators=[
                            isinstance_validator(int),
                            greater_than_validator(0),
                        ],
                    ),
                },
            )
            calc_inv_times_validator.validate(
                self.inputs, error_type=FilterInputValidationError
            )

    @staticmethod
    def calculate_mz(
        initial_mz: np.ndarray,
        t1: np.ndarray,
        inv_pulse_times: list,
        sat_pulse_time: float,
        mag_time: float,
        inv_eff: np.ndarray,
        sat_eff: np.ndarray = 1.0,
    ) -> np.ndarray:
        """Calculates the longitudinal magnetisation after
        a sequence of background suppression pulses. Calculates
        according to the equation in:

            Maleki, N., Dai, W. & Alsop, D.C. Optimization of
            background suppression for arterial spin labeling
            perfusion imaging. Magn Reson Mater Phy 25,
            127–133 (2012). https://doi.org/10.1007/s10334-011-0286-3

        This is derived from the equations in the appendix of

            Mani, S., Pauly, J., Conolly, S., Meyer, C. and
            Nishimura, D. (1997), Background suppression with
            multiple inversion recovery nulling: Applications
            to projective angiography. Magn. Reson. Med., 37:
            898-905. https://doi.org/10.1002/mrm.1910370615

        :param initial_mz: The initial longitudinal magnetisation
        :type initial_mz: np.ndarray
        :param t1: The longitudinal relaxation time
        :type t1: np.ndarray
        :param inv_pulse_times: Inversion pulse times, with respect
          to the imaging excitation pulse.
        :type inv_pulse_times: list[float]
        :param mag_time: The time at which to calculate the 
          longitudinal magnetisation
        :param sat_pulse_time: The time between the saturation pulse
          and the imaging excitation pulse.
        :type sat_pulse_time:
        :type mag_time: float
        :param inv_eff: The efficiency of the inversion pulses,
          -1 is complete inversion. 
        :type inv_eff: np.ndarray
        :param sat_eff: The efficiency of the saturation pulses, 1 is
          full saturation.
        :type sat_eff: np.ndarray
        :return: The longitudinal magnetisation after the background
          suppression sequence
        :rtype: np.ndarray
        """
        # check that initial_mz, t1 and pulse_eff are broadcastable
        np.broadcast(initial_mz, t1, inv_eff, sat_eff)

        # sort the inversion pulse times into ascending order
        # inv_pulse_times = np.sort(inv_pulse_times)
        # pdb.set_trace()
        # determine the number of inversion pulses that will have played
        # out by t=mag_time
        inv_pulse_times = np.asarray(inv_pulse_times)
        inv_pulse_times = inv_pulse_times[mag_time > sat_pulse_time - inv_pulse_times]
        num_pulses = len(inv_pulse_times)

        return initial_mz * (
            1
            + ((1 - sat_eff) - 1)
            * inv_eff ** num_pulses
            * np.exp(-np.divide(mag_time, t1, out=np.zeros_like(t1), where=t1 != 0))
            + np.sum(
                [
                    ((inv_eff ** (m + 1)) - (inv_eff ** m))
                    * np.exp(-np.divide(tm, t1, out=np.zeros_like(t1), where=t1 != 0))
                    for m, tm in enumerate(inv_pulse_times)
                ],
                0,
            )
        )

    @staticmethod
    def calculate_pulse_efficiency(t1: np.ndarray) -> np.ndarray:
        r"""Calculates the pulse efficiency per t1 according to the
        polynomial described in:

            Maleki, N., Dai, W. & Alsop, D.C. Optimization of
            background suppression for arterial spin labeling
            perfusion imaging. Magn Reson Mater Phy 25,
            127–133 (2012). https://doi.org/10.1007/s10334-011-0286-3
        
        :param t1: t1 times to calculate the pulse efficiencies for, seconds.
        :type t1: np.ndarray
        :return: The pulse efficiencies, :math:`\chi`
        :rtype: np.ndarray
        """
        # convert t1 to a ndarray for consistency
        t1 = np.asarray(t1)
        pulse_eff = np.zeros_like(t1)
        t1 = t1 * 1000  # paper gives polynomial based on ms, so convert t1 to ms
        mid_t1 = (450.0 <= t1) & (t1 <= 2000.0)
        pulse_eff[(250.0 <= t1) & (t1 < 450.0)] = -0.998
        pulse_eff[mid_t1] = -(
            (-2.245e-15) * t1[mid_t1] ** 4
            + (2.378e-11) * t1[mid_t1] ** 3
            - (8.987e-8) * t1[mid_t1] ** 2
            + (1.442e-4) * t1[mid_t1]
            + (9.1555e-1)
        )
        pulse_eff[(2000.0 < t1) & (t1 <= 4200.0)] = -0.998
        return pulse_eff

    @staticmethod
    def optimise_inv_pulse_times(
        sat_time: float,
        t1: np.ndarray,
        pulse_eff: np.ndarray,
        num_pulses: int,
        method: str = "Nelder-Mead",
    ) -> OptimizeResult:
        """Calculates optimised inversion pulse times
        for a background suppression pulse sequence.

        :param sat_time: The time, in seconds between the saturation pulse and 
          the imaging excitation pulse.
        :type sat_time: float
        :param t1: The longitudinal relaxation times to optimise the pulses for
        :type t1: np.ndarray
        :param pulse_eff: The inversion pulse efficiency, corresponding to each
          ``t1`` entry.
        :type pulse_eff: np.ndarray
        :param num_pulses: The number of inversion pulses to optimise times for,
          must be greater than 0.
        :type num_pulses: int
        :param method: The optimisation method to use, see 
          :class:`scipy.optimize.minimize` for more details. Defaults to "Nelder-Mead".
        :type method: str, optional
        :raises ValueError: If the number of pulses is less than 1.
        :return: The result from the optimisation
        :rtype: OptimizeResult
        """
        if not num_pulses > 0:
            raise ValueError("num_pulses must be greater than 0")

        x0 = np.ones((num_pulses,))
        # create the objective function to optimise: ||Mz(t=sat_time)||^2
        fun = lambda x: np.sum(
            BackgroundSuppressionFilter.calculate_mz(
                np.ones_like(t1), t1, x, sat_time, sat_time, pulse_eff
            )
            ** 2
        )
        # perform the optimisation
        result = minimize(fun, x0, method=method)
        return result
