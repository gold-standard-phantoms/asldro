""" General Kinetic Model Filter """

import logging
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

logger = logging.getLogger(__name__)


class GkmFilter(BaseFilter):
    r"""
    A filter that generates the ASL signal using the General Kinetic Model.

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`GkmFilter.add_inputs()`
    member function. They are also accessible via class constants,
    for example :class:`GkmFilter.KEY_PERFUSION_RATE`

    :param 'perfusion_rate': Map of perfusion rate, in ml/100g/min (>=0)
    :type 'perfusion_rate': BaseImageContainer
    :param 'transit_time'  Map of the time taken for the labelled bolus
      to reach the voxel, seconds (>=0).
    :type 'transit_time': BaseImageContainer
    :param 'm0': The tissue equilibrium magnetisation, can be a map or single value (>=0).
    :type 'perfusion_rate': BaseImageContainer or float
    :param 'label_type': Determines which GKM equations to use:

      * "casl" OR "pcasl" (case insensitive) for the continuous model
      * "pasl" (case insensitive) for the pulsed model

    :type 'label_type': str
    :param 'label_duration': The length of the labelling pulse, seconds (0 to 100 inclusive)
    :type 'label_duration': float
    :param 'signal_time': The time after labelling commences to generate signal,
      seconds (0 to 100 inclusive)
    :type 'signal_time': float
    :param 'label_efficiency': The degree of inversion of the labelling (0 to 1 inclusive)
    :type 'label_efficiency': float
    :param 'lambda_blood_brain': The blood-brain-partition-coefficient (0 to 1 inclusive)
    :type 'lambda_blood_brain': float or BaseImageContainer
    :param 't1_arterial_blood': Longitudinal relaxation time of arterial blood,
        seconds (0 exclusive to 100 inclusive)
    :type 't1_arterial_blood': float
    :param 't1_tissue': Longitudinal relaxation time of the tissue,
        seconds (0 to 100 inclusive, however voxels with ``t1 = 0`` will have ``delta_m = 0``)
    :type 't1_tissue': BaseImageContainer
    :param 'model': The model to use to generate the perfusion signal:

      * "full" for the full "Buxton" General Kinetic Model :cite:p:`Buxton1998`
      * "whitepaper" for the simplified model, derived from the quantification
        equations the ASL Whitepaper consensus paper :cite:p:`Alsop2014`.

      Defaults to "full".

    :type 'model': str

    **Outputs**

    Once run, the filter will populate the dictionary :class:`GkmFilter.outputs`
    with the following entries

    :param 'delta_m': An image with synthetic ASL perfusion contrast. This will
      be the same class as the input 'perfusion_rate'
    :type 'delta_m': BaseImageContainer

    **Metadata**

    The following parameters are added to 
    :class:`GkmFilter.outputs["delta_m"].metadata`:

    * ``label_type``
    * ``label_duration`` (pcasl/casl only)
    * ``post_label_delay``
    * ``bolus_cut_off_flag`` (pasl only)
    * ``bolus_cut_off_delay_time`` (pasl only)
    * ``label_efficiency``
    * ``lambda_blood_brain`` (only if a single value is supplied)
    * ``t1_arterial_blood``
    * ``m0`` (only if a single value is supplied)
    * ``gkm_model`` = ``model``

    ``post_label_delay`` is calculated as ``signal_time - label_duration``

    ``bolus_cut_off_delay_time`` takes the value of the input
    ``label_duration``, this field is used for pasl in line with
    the BIDS specification.


    **Equations**

    The general kinetic model :cite:p:`Buxton1998` is the standard signal model 
    for ASL perfusion measurements. It considers the difference between the
    control and label conditions to be a deliverable tracer, referred to
    as :math:`\Delta M(t)`.

    The amount of :math:`\Delta M(t)` within a voxel at time :math:`t`
    depends on the history of:

    * delivery of magnetisation by arterial flow
    * clearance by venous flow
    * longitudinal relaxation

    These processes can be described by defining three functions of time:

    1. The delivery function :math:`c(t)` - the normalised arterial
       concentration of magnetisation arriving at the voxel
       at time :math:`t`.
    2. The residue function :math:`r(t,t')` - the fraction of tagged water
       molecules that arrive at time :math:`t'` and
       are still in the voxel at time :math:`t`.
    3. The magnetisation relaxation function :math:`m(t,t')` is the fraction
       of the original longitudinal magnetisation tag carried by the water
       molecules that arrived at time :math:`t'` that remains at time :math:`t`.

    Using these definitions :math:`\Delta M(t)` can be constructed as the sum
    over history of delivery of magnetisation to the tissue weighted with the
    fraction of that magnetisation that remains in the voxel:

    .. math::

        &\Delta M(t)=2\cdot M_{0,b}\cdot f\cdot\left\{ c(t)\ast\left[r(t)\cdot m(t)\right]\right\}\\
        &\text{where}\\
        &\ast=\text{convolution operator} \\
        &r(t)=\text{residue function}=e^{-\frac{ft}{\lambda}}\\
        &m(t)=e^{-\frac{t}{T_{1}}}\\
        &c(t)=\text{delivery function, defined as plug flow} = \begin{cases}
        0  &  0<t<\Delta t\\
        \alpha e^{-\frac{t}{T_{1,b}}}\,\text{(PASL)}  &  \Delta t<t<\Delta t+\tau\\
        \alpha e^{-\frac{\Delta t}{T_{1,b}}}\,\text{(CASL/pCASL)}\\
        0  &  t>\Delta t+\tau
        \end{cases}\\
        &\alpha=\text{labelling efficiency} \\
        &\tau=\text{label duration} \\
        &\Delta t=\text{initial transit delay, ATT} \\
        &M_{0,b} = \text{equilibrium magnetisation of arterial blood} = \frac{M_{0,\text{tissue}}}{\lambda} \\
        & f = \text{the perfusion rate, CBF}\\
        &\lambda = \text{blood brain partition coefficient}\\
        &T_{1,b} = \text{longitudinal relaxation time of arterial blood}\\
        &T_{1} = \text{longitudinal relaxation time of tissue}\\
        
    Note that all units are in SI, with :math:`f` having units :math:`s^{-1}`.
    Multiplying by 6000 gives units of :math:`ml/100g/min`.

    *Full Model*
    
    The full solutions to the GKM :cite:p:`Buxton1998` are used to calculate
    :math:`\Delta M(t)` when ``model=="full"``:

    *   (p)CASL:

        .. math::

            &\Delta M(t)=\begin{cases}
            0 & 0<t\leq\Delta t\\
            2M_{0,b}fT'_{1}\alpha e^{-\frac{\Delta t}{T_{1,b}}}q_{ss}(t) &
            \Delta t<t<\Delta t+\tau\\
            2M_{0,b}fT'_{1}\alpha e^{-\frac{\Delta t}{T_{1,b}}}
            e^{-\frac{t-\tau-\Delta t}{T'_{1}}}q_{ss}(t) & t\geq\Delta t+\tau
            \end{cases}\\
            &\text{where}\\
            &q_{ss}(t)=\begin{cases}
            1-e^{-\frac{t-\Delta t}{T'_{1}}} & \Delta t<t <\Delta t+\tau\\
            1-e^{-\frac{\tau}{T'_{1}}} & t\geq\Delta t+\tau
            \end{cases}\\
            &\frac{1}{T'_{1}}=\frac{1}{T_1} + \frac{f}{\lambda}\\

    *   PASL:

        .. math::

            &\Delta M(t)=\begin{cases}
            0 & 0<t\leq\Delta t\\
            2M_{0,b}f(t-\Delta t) \alpha e^{-\frac{t}{T_{1,b}}}q_{p}(t)
            & \Delta t < t < t\Delta t+\tau\\
            2M_{0,b}f\alpha \tau e^{-\frac{t}{T_{1,b}}}q_{p}(t)
            & t\geq\Delta t+\tau
            \end{cases}\\
            &\text{where}\\
            &q_{p}(t)=\begin{cases}
            \frac{e^{kt}(e^{-k \Delta t}-e^{-kt})}{k(t-\Delta t)}
            & \Delta t<t<\Delta t+\tau\\
            \frac{e^{kt}(e^{-k\Delta t}-e^{k(\tau + \Delta t)}}{k\tau}
            & t\geq\Delta t+\tau
            \end{cases}\\
            &\frac{1}{T'_{1}}=\frac{1}{T_1} + \frac{f}{\lambda}\\
            &k=\frac{1}{T_{1,b}}-\frac{1}{T'_1}

    *Simplified Model"

    The simplified model, derived from the single subtraction quantification
    equations (see :class:`.AslQuantificationFilter`) are used when
    ``model=="whitepaper"``:

    *   (p)CASL:

        .. math::

            &\Delta M(t) = \begin{cases}
            0 & 0<t\leq\Delta t + \tau\\
            {2  M_{0,b}  f  T_{1,b} \alpha 
            (1-e^{-\frac{\tau}{T_{1,b}}}) e^{-\frac{t-\tau}{T_{1,b}}}}
            & t > \Delta t + \tau\\
            \end{cases}\\

    *   PASL

        .. math::

            &\Delta M(t) = \begin{cases}
            0 & 0<t\leq\Delta t + \tau\\
            {2  M_{0,b}  f  \tau  \alpha  
            e^{-\frac{t}{T_{1,b}}}} & t > \Delta t + \tau\\
            \end{cases}

    """

    # Key constants
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
    KEY_MODEL = "model"
    M_POST_LABEL_DELAY = "post_label_delay"
    M_BOLUS_CUT_OFF_FLAG = "bolus_cut_off_flag"
    M_BOLUS_CUT_OFF_DELAY_TIME = "bolus_cut_off_delay_time"
    M_GKM_MODEL = "gkm_model"

    # Value constants
    CASL = "casl"
    PCASL = "pcasl"
    PASL = "pasl"

    MODEL_FULL = "full"
    MODEL_WP = "whitepaper"

    def __init__(self):
        super().__init__(name="General Kinetic Model")

    def _run(self):
        """Generates the delta_m signal based on the inputs"""

        perfusion_rate: np.ndarray = self.inputs[self.KEY_PERFUSION_RATE].image
        transit_time: np.ndarray = self.inputs[self.KEY_TRANSIT_TIME].image
        t1_tissue: np.ndarray = self.inputs[self.KEY_T1_TISSUE].image

        label_duration: float = self.inputs[self.KEY_LABEL_DURATION]
        signal_time: float = self.inputs[self.KEY_SIGNAL_TIME]
        label_efficiency: float = self.inputs[self.KEY_LABEL_EFFICIENCY]
        t1_arterial_blood: float = self.inputs[self.KEY_T1_ARTERIAL_BLOOD]
        model: str = self.inputs[self.KEY_MODEL]
        label_type = self.inputs[self.KEY_LABEL_TYPE].lower()

        # blank dictionary for metadata to add
        metadata = {}
        m0_tissue = GkmFilter.check_and_make_image_from_value(
            self.inputs[self.KEY_M0], perfusion_rate.shape, metadata, self.KEY_M0
        )
        lambda_blood_brain = GkmFilter.check_and_make_image_from_value(
            self.inputs[self.KEY_LAMBDA_BLOOD_BRAIN],
            perfusion_rate.shape,
            metadata,
            self.KEY_LAMBDA_BLOOD_BRAIN,
        )

        # REFACTOR FROM HERE
        if model == self.MODEL_FULL:
            # gkm function
            # logger.info(
            #     "Full General Kinetic Model for Continuous/pseudo-Continuous ASL"
            #     if label_type in [self.PCASL, self.CASL]
            #     else "Full General Kinetic Model for Pulsed ASL"
            # )
            delta_m = GkmFilter.calculate_delta_m_gkm(
                perfusion_rate,
                transit_time,
                m0_tissue,
                label_duration,
                signal_time,
                label_efficiency,
                lambda_blood_brain,
                t1_arterial_blood,
                t1_tissue,
                label_type,
            )

        elif model == self.MODEL_WP:
            # whitepaper function
            # logger.info(
            #     "Simplified Kinetic Model for Continuous/pseudo-Continuous ASL"
            #     if label_type in [self.PCASL, self.CASL]
            #     else "Simplified Kinetic Model for Pulsed ASL"
            # )
            delta_m = GkmFilter.calculate_delta_m_whitepaper(
                perfusion_rate,
                transit_time,
                m0_tissue,
                label_duration,
                signal_time,
                label_efficiency,
                lambda_blood_brain,
                t1_arterial_blood,
                label_type,
            )
        # add metadata depending on the label type
        if label_type == self.PASL:
            metadata[self.M_BOLUS_CUT_OFF_FLAG] = True
            metadata[self.M_BOLUS_CUT_OFF_DELAY_TIME] = label_duration
        elif label_type in [self.CASL, self.PCASL]:
            metadata[self.KEY_LABEL_DURATION] = label_duration

        # copy 'perfusion_rate' image container and set the image to delta_m
        self.outputs[self.KEY_DELTA_M] = self.inputs[self.KEY_PERFUSION_RATE].clone()
        # remove some metadata fields
        self.outputs[self.KEY_DELTA_M].metadata.pop("units", None)
        self.outputs[self.KEY_DELTA_M].metadata.pop("quantity", None)
        self.outputs[self.KEY_DELTA_M].image = delta_m

        # add common fields to metadata
        metadata = {
            **metadata,
            **{
                self.KEY_LABEL_TYPE: self.inputs[self.KEY_LABEL_TYPE].lower(),
                self.M_POST_LABEL_DELAY: (signal_time - label_duration),
                self.KEY_LABEL_EFFICIENCY: label_efficiency,
                self.KEY_T1_ARTERIAL_BLOOD: t1_arterial_blood,
                "image_flavour": "PERFUSION",
                self.M_GKM_MODEL: model,
            },
        }
        # merge this with the output image's metadata
        self.outputs[self.KEY_DELTA_M].metadata = {
            **self.outputs[self.KEY_DELTA_M].metadata,
            **metadata,
        }

    def _validate_inputs(self):
        """Checks that the inputs meet their validation criteria
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
                self.KEY_PERFUSION_RATE: Parameter(
                    validators=[
                        greater_than_equal_to_validator(0),
                        isinstance_validator(BaseImageContainer),
                    ]
                ),
                self.KEY_TRANSIT_TIME: Parameter(
                    validators=[
                        greater_than_equal_to_validator(0),
                        isinstance_validator(BaseImageContainer),
                    ]
                ),
                self.KEY_M0: Parameter(
                    validators=[
                        greater_than_equal_to_validator(0),
                        isinstance_validator((BaseImageContainer, float)),
                    ]
                ),
                self.KEY_T1_TISSUE: Parameter(
                    validators=[
                        range_inclusive_validator(0, 100),
                        isinstance_validator(BaseImageContainer),
                    ]
                ),
                self.KEY_LABEL_TYPE: Parameter(
                    validators=from_list_validator(
                        [self.CASL, self.PCASL, self.PASL], case_insensitive=True
                    )
                ),
                self.KEY_LABEL_DURATION: Parameter(
                    validators=[
                        range_inclusive_validator(0, 100),
                        isinstance_validator(float),
                    ]
                ),
                self.KEY_SIGNAL_TIME: Parameter(
                    validators=[
                        range_inclusive_validator(0, 100),
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
                        isinstance_validator((BaseImageContainer, float)),
                    ]
                ),
                self.KEY_T1_ARTERIAL_BLOOD: Parameter(
                    validators=[
                        range_inclusive_validator(0, 100),
                        isinstance_validator(float),
                    ]
                ),
                self.KEY_MODEL: Parameter(
                    validators=from_list_validator(
                        [self.MODEL_FULL, self.MODEL_WP], case_insensitive=True
                    ),
                    default_value=self.MODEL_FULL,
                ),
            }
        )

        new_params = input_validator.validate(
            self.inputs, error_type=FilterInputValidationError
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
        # merge the updated parameters from the output with the input parameters
        self.inputs = {**self._i, **new_params}

    @staticmethod
    def check_and_make_image_from_value(
        arg: float or BaseImageContainer,
        shape: tuple,
        metadata: dict,
        metadata_key: str,
    ) -> np.ndarray:
        """Checks the type of the input parameter to see if it is a float or a BaseImageContainer.
        If it is an image:

        * return the image ndarray
        * check if it has the same value everywhere (i.e. an image override), if it does then
          place the value into the `metadata` dict under the `metadata_key`

        If it is a float:
        * make a ndarray with the same value
        * place the value into the `metadata` dict under the `metadata_key`

        This makes calculations more straightforward as a ndarray can always be expected.

        **Arguments**

        :param arg: The input parameter to check
        :type arg: float or BaseImageContainer
        :param shape: The shape of the image to create
        :type shape: tuple
        :param metadata: metadata dict, which is updated by this function
        :type metadata: dict
        :param metadata_key: key to assign the value of arg (if a float or single value image) to
        :type metadata_key: str

        :return: image of the parameter
        :rype: np.ndarray

        """

        if isinstance(arg, BaseImageContainer):
            out_array: np.ndarray = arg.image
            # Get a flattened view of nD numpy array
            flatten_arr = np.ravel(out_array)
            # Check if all value in array are equal and update metadata if so
            if np.all(out_array == flatten_arr[0]):
                metadata[metadata_key] = flatten_arr[0]

        else:
            out_array: np.ndarray = arg * np.ones(shape)
            metadata[metadata_key] = arg
        return out_array

    @staticmethod
    def compute_arrival_state_masks(
        transit_time: np.ndarray, signal_time: float, label_duration: float,
    ) -> dict:
        """Creates boolean masks for each of the states of the delivery curve

        :param transit_time: map of the transit time
        :type transit_time: np.ndarray
        :param signal_time: the time to generate signal at
        :type signal_time: float
        :param label_duration: The duration of the labelling pulse
        :type label_duration: float
        :return: a dictionary with three entries, each a ndarray with shape
          the same as `transit_time`:

          :"not_arrived": voxels where the bolus has not reached yet
          :"arriving": voxels where the bolus has reached but not been completely
            delivered.
          :"arrived": voxels where the bolus has been completely delivered

        :rtype: dict
        """
        return {
            "not_arrived": 0 < signal_time <= transit_time,
            "arriving": (transit_time < signal_time)
            & (signal_time < transit_time + label_duration),
            "arrived": signal_time >= transit_time + label_duration,
        }

    @staticmethod
    def calculate_delta_m_gkm(
        perfusion_rate: np.ndarray,
        transit_time: np.ndarray,
        m0_tissue: np.ndarray,
        label_duration: float,
        signal_time: float,
        label_efficiency: float,
        partition_coefficient: np.ndarray,
        t1_arterial_blood: float,
        t1_tissue: np.ndarray,
        label_type: str,
    ) -> np.ndarray:
        """Calculates the difference in magnetisation between the control
        and label condition (:math:`\Delta M`) using the full solutions to the
        General Kinetic Model :cite:p:`Buxton1998`.

        :param perfusion_rate: Map of perfusion rate
        :type perfusion_rate: np.ndarray
        :param transit_time: Map of transit time
        :type transit_time: np.ndarray
        :param m0_tissue: The tissue equilibrium magnetisation
        :type m0_tissue: np.ndarray
        :param label_duration: The length of the labelling pulse
        :type label_duration: float
        :param signal_time: The time after the labelling pulse commences to generate signal.
        :type signal_time: float
        :param label_efficiency: The degree of inversion of the labelling pulse.
        :type label_efficiency: float
        :param partition_coefficient: The tissue-blood partition coefficient
        :type partition_coefficient: np.ndarray
        :param t1_arterial_blood: Longitudinal relaxation time of the arterial blood.
        :type t1_arterial_blood: float
        :param t1_tissue: Longitudinal relaxation time of the tissue
        :type t1_tissue: np.ndarray
        :param label_type: Determines the specific model to use: Pulsed ("pasl") or
          (pseudo)Continuous ("pcasl" or "casl") labelling
        :type label_type: str
        :return: the difference magnetisation, :math:`\Delta M`
        :rtype: np.ndarray

        """
        # divide perfusion_rate by 6000 to put into SI units
        perfusion_rate = np.asarray(perfusion_rate) / 6000

        # calculate M0b, handling runtime divide-by-zeros
        m0_arterial_blood = np.divide(
            m0_tissue,
            partition_coefficient,
            out=np.zeros_like(partition_coefficient),
            where=partition_coefficient != 0,
        )
        # calculate T1', handling runtime divide-by-zeros
        flow_over_lambda = np.divide(
            perfusion_rate,
            partition_coefficient,
            out=np.zeros_like(partition_coefficient),
            where=partition_coefficient != 0,
        )
        one_over_t1_tissue = np.divide(
            1, t1_tissue, out=np.zeros_like(t1_tissue), where=t1_tissue != 0
        )
        denominator = one_over_t1_tissue + flow_over_lambda
        t1_prime: np.ndarray = np.divide(
            1, denominator, out=np.zeros_like(denominator), where=denominator != 0
        )
        condition_masks = GkmFilter.compute_arrival_state_masks(
            transit_time, signal_time, label_duration
        )
        delta_m = np.zeros(perfusion_rate.shape)  # pre-allocate delta_m
        if label_type.lower() == GkmFilter.PASL:
            # do GKM for PASL
            k: np.ndarray = (
                1 / t1_arterial_blood if t1_arterial_blood != 0 else 0
            ) - np.divide(1, t1_prime, out=np.zeros_like(t1_prime), where=t1_prime != 0)
            # if transit_time == signal_time then there is a divide-by-zero condition.  Calculate
            # numerator and denominator separately for q_pasl_arriving
            numerator = np.exp(k * signal_time) * (
                np.exp(-k * transit_time) - np.exp(-k * signal_time)
            )
            denominator = k * (signal_time - transit_time)
            q_pasl_arriving = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator != 0,
            )
            numerator = np.exp(k * signal_time) * (
                np.exp(-k * transit_time) - np.exp(-k * (transit_time + label_duration))
            )
            denominator = k * label_duration
            q_pasl_arrived = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(denominator),
                where=denominator != 0,
            )
            delta_m_arriving = (
                2
                * m0_arterial_blood
                * perfusion_rate
                * (signal_time - transit_time)
                * label_efficiency
                * (
                    np.exp(-signal_time / t1_arterial_blood)
                    if t1_arterial_blood > 0
                    else 0
                )
                * q_pasl_arriving
            )
            delta_m_arrived = (
                2
                * m0_arterial_blood
                * perfusion_rate
                * label_efficiency
                * label_duration
                * (
                    np.exp(-signal_time / t1_arterial_blood)
                    if t1_arterial_blood > 0
                    else 0
                )
                * q_pasl_arrived
            )
        elif label_type.lower() in [GkmFilter.CASL, GkmFilter.PCASL]:
            # do GKM for CASL/pCASL
            q_ss_arriving = 1 - np.exp(
                -np.divide(
                    (signal_time - transit_time),
                    t1_prime,
                    out=np.zeros_like(t1_prime),
                    where=t1_prime != 0,
                )
            )
            q_ss_arrived = 1 - np.exp(
                -np.divide(
                    label_duration,
                    t1_prime,
                    out=np.zeros_like(t1_prime),
                    where=t1_prime != 0,
                )
            )
            delta_m_arriving = (
                2
                * m0_arterial_blood
                * perfusion_rate
                * t1_prime
                * label_efficiency
                * (
                    np.exp(-transit_time / t1_arterial_blood)
                    if t1_arterial_blood != 0
                    else np.zeros_like(transit_time)
                )
                * q_ss_arriving
            )
            delta_m_arrived = (
                2
                * m0_arterial_blood
                * perfusion_rate
                * t1_prime
                * label_efficiency
                * (
                    np.exp(-transit_time / t1_arterial_blood)
                    if t1_arterial_blood != 0
                    else np.zeros_like(transit_time)
                )
                * np.exp(
                    -np.divide(
                        (signal_time - label_duration - transit_time),
                        t1_prime,
                        out=np.zeros_like(t1_prime),
                        where=t1_prime != 0,
                    )
                )
                * q_ss_arrived
            )
        # combine the different arrival states into delta_m
        delta_m[condition_masks["not_arrived"]] = 0.0
        delta_m[condition_masks["arriving"]] = delta_m_arriving[
            condition_masks["arriving"]
        ]
        delta_m[condition_masks["arrived"]] = delta_m_arrived[
            condition_masks["arrived"]
        ]
        return delta_m

    @staticmethod
    def calculate_delta_m_whitepaper(
        perfusion_rate: np.ndarray,
        transit_time: np.ndarray,
        m0_tissue: np.ndarray,
        label_duration: float,
        signal_time: float,
        label_efficiency: float,
        partition_coefficient: np.ndarray,
        t1_arterial_blood: float,
        label_type: str,
    ) -> np.ndarray:
        """Calculates the difference in magnetisation between the control
        and label condition (:math:`\Delta M`) using the single
        subtraction simplification from the  ASL Whitepaper consensus paper
        :cite:p:`Alsop2014`.

        :param perfusion_rate: Map of perfusion rate
        :type perfusion_rate: np.ndarray
        :param transit_time: Map of transit time
        :type transit_time: np.ndarray
        :param m0_tissue: The tissue equilibrium magnetisation
        :type m0_tissue: np.ndarray
        :param label_duration: The length of the labelling pulse
        :type label_duration: float
        :param signal_time: The time after the labelling pulse commences to generate signal.
        :type signal_time: float
        :param label_efficiency: The degree of inversion of the labelling pulse.
        :type label_efficiency: float
        :param partition_coefficient: The tissue-blood partition coefficient
        :type partition_coefficient: np.ndarray
        :param t1_arterial_blood: Longitudinal relaxation time of the arterial blood.
        :type t1_arterial_blood: float
        :param t1_tissue: Longitudinal relaxation time of the tissue
        :type t1_tissue: np.ndarray
        :param label_type: Determines the specific model to use: Pulsed ("pasl") or
          (pseudo)Continuous ("pcasl" or "casl") labelling
        :type label_type: str
        :return: the difference magnetisation, :math:`\Delta M`
        :rtype: np.ndarray

        """
        # divide perfusion_rate by 6000 to put into SI units
        perfusion_rate = np.asarray(perfusion_rate) / 6000

        # calculate M0b, handling runtime divide-by-zeros
        m0_arterial_blood = np.divide(
            m0_tissue,
            partition_coefficient,
            out=np.zeros_like(partition_coefficient),
            where=partition_coefficient != 0,
        )
        condition_masks = GkmFilter.compute_arrival_state_masks(
            transit_time, signal_time, label_duration
        )
        delta_m = np.zeros(perfusion_rate.shape)  # pre-allocate delta_m

        if label_type.lower() == GkmFilter.PASL:
            delta_m_arriving = np.zeros_like(delta_m)
            # use simplified model for PASL
            delta_m_arrived = (
                2
                * m0_arterial_blood
                * perfusion_rate
                * label_duration
                * label_efficiency
                * np.exp(-signal_time / t1_arterial_blood)
                if t1_arterial_blood > 0
                else 0
            )

        elif label_type.lower() in [GkmFilter.PCASL, GkmFilter.CASL]:
            delta_m_arriving = np.zeros_like(delta_m)
            delta_m_arrived = (
                2
                * m0_arterial_blood
                * perfusion_rate
                * t1_arterial_blood
                * label_efficiency
                * np.exp(
                    -(signal_time - label_duration) / t1_arterial_blood
                    if t1_arterial_blood != 0
                    else 0
                )
                * (
                    1 - np.exp(-label_duration / t1_arterial_blood)
                    if t1_arterial_blood != 0
                    else 0
                )
            )

        # combine the different arrival states into delta_m
        delta_m[condition_masks["not_arrived"]] = 0.0
        delta_m[condition_masks["arriving"]] = delta_m_arriving[
            condition_masks["arriving"]
        ]
        delta_m[condition_masks["arrived"]] = delta_m_arrived[
            condition_masks["arrived"]
        ]
        return delta_m

