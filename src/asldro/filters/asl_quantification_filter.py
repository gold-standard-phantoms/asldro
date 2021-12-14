"""ASL quantification filter class"""
from typing import List
import numpy as np
from scipy.optimize import curve_fit
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.filters.gkm_filter import GkmFilter
from asldro.containers.image import BaseImageContainer
from asldro.validators.parameters import (
    Parameter,
    ParameterValidator,
    for_each_validator,
    isinstance_validator,
    greater_than_equal_to_validator,
    greater_than_validator,
    from_list_validator,
    range_inclusive_validator,
    and_validator,
    for_each_validator,
    shape_validator,
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
    :param 'label_type': the type of labelling used: "pasl" for pulsed ASL
      "pcasl" or "casl" for for continuous ASL.
    :type 'label_type': str
    :param 'lambda_blood_brain': The blood-brain-partition-coefficient (0 to 1 inclusive)
    :type 'lambda_blood_brain': float
    :param 'label_duration': The temporal duration of the labelled bolus, seconds
      (0 or greater). For PASL this is equivalent to :math:`\text{TI}_1`
    :type 'label_duration': float
    :param 'post_label_delay': The duration between the end of the labelling
        pulse and the imaging excitation pulse, seconds (0 or greater).
        For PASL this is equivalent to :math:`\text{TI}`.
        If ``'model'=='full'`` then this must be a list and the length of this
        must match the number of unique entries in ``'multiphase_index'``.
    :type 'post_label_delay': float or List[float]
    :param 'label_efficiency': The degree of inversion of the labelling
      (0 to 1 inclusive)
    :type 'label_efficiency': float
    :param 't1_arterial_blood': Longitudinal relaxation time of arterial blood,
        seconds (greater than 0)
    :type 't1_arterial_blood': float
    :param 't1_tissue': Longitudinal relaxation time of the tissue, seconds
        (greater than 0). Required if ``'model'=='full'``
    :type 't1_tissue': float or BaseImageContainer
    :param 'model': defines which model to use

        * 'whitepaper' uses the single-subtraction white paper equation
        * 'full' least square fitting to the full GKM.

    :type 'model': str
    :param 'multiphase_index': A list the same length as the fourth dimension
        of the label image that defines which phase each image belongs to,
        and is also the corresponding index in the ``'post_label_delay'`` list.
        Required if ``'model'=='full'``.

    **Outputs**

    :param 'perfusion_rate': map of the calculated perfusion rate
    :type 'perfusion_rate': BaseImageContainer

    If ``'model'=='full'`` the following are also output:

    :param 'transit_time': The estimated transit time in seconds.
    :type 'transit_time': BaseImageContainer
    :param 'std_error': The standard error of the estimate of the fit.
    :type 'std_error': BaseImageContainer
    :param 'perfusion_rate_err': One standard deviation error in the fitted
    perfusion rate.
    :type 'perfusion_rate_err': BaseImageContainer
    :param 'transit_time_err': One standard deviation error in the fitted
      transit time.
    :type 'transit_time_err': BaseImageContainer

    **Quantification Model**

    The following equations are used to calculate the perfusion rate, depending
    on the input ``model``:

    :'whitepaper': simplified single subtraction equations :cite:p:`Alsop2014`.

      * for pCASL/CASL see :class:`AslQuantificationFilter.asl_quant_wp_casl`
      * for PASL see :class:`AslQuantificationFilter.asl_quant_wp_pasl`.

    :'full': Lease squares fitting to the full General Kinetic Model :cite:p:`Buxton1998`.
      See :class:`AslQuantificationFilter.asl_quant_lsq_gkm`.

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
    KEY_POST_LABEL_DELAY = GkmFilter.M_POST_LABEL_DELAY
    KEY_MULTIPHASE_INDEX = "multiphase_index"
    KEY_T1_TISSUE = GkmFilter.KEY_T1_TISSUE
    KEY_TRANSIT_TIME = GkmFilter.KEY_TRANSIT_TIME
    KEY_PERFUSION_RATE_ERR = "perfusion_rate_err"
    KEY_TRANSIT_TIME_ERR = "transit_time_err"
    KEY_STD_ERROR = "std_error"

    WHITEPAPER = GkmFilter.MODEL_WP
    FULL = GkmFilter.MODEL_FULL
    M0_TOL = 1e-6

    FIT_IMAGE_NAME = {
        KEY_STD_ERROR: "FITErr",
        KEY_PERFUSION_RATE_ERR: "RCBFErr",
        KEY_TRANSIT_TIME: "ATT",
        KEY_TRANSIT_TIME_ERR: "ATTErr",
    }
    FIT_IMAGE_UNITS = {
        KEY_STD_ERROR: "a.u.",
        KEY_PERFUSION_RATE_ERR: "ml/100g/min",
        KEY_TRANSIT_TIME: "s",
        KEY_TRANSIT_TIME_ERR: "s",
    }
    ESTIMATION_ALGORITHM = {
        WHITEPAPER: """Calculated using the single subtraction simplified model for
CBF quantification from the ASL White Paper:

Alsop et. al., Recommended implementation of arterial
spin-labeled perfusion MRI for clinical applications:
a consensus of the ISMRM perfusion study group and the
european consortium for ASL in dementia. Magnetic Resonance
in Medicine, 73(1):102–116, apr 2014. doi:10.1002/mrm.25197
""",
        FULL: """Least Squares fit to the General Kinetic Model for
Arterial Spin Labelling:

Buxton et. al., A general
kinetic model for quantitative perfusion imaging with arterial
spin labeling. Magnetic Resonance in Medicine, 40(3):383–396,
sep 1998. doi:10.1002/mrm.1910400308.""",
    }

    def __init__(self):
        super().__init__(name="ASL Quantification")

    def _run(self):
        """Calculates the perfusion rate based on the inputs"""

        # create output image
        self.outputs[self.KEY_PERFUSION_RATE] = self.inputs[self.KEY_CONTROL].clone()
        # amend the metadata
        self.outputs[self.KEY_PERFUSION_RATE].metadata.pop("RepetitionTime", None)
        self.outputs[self.KEY_PERFUSION_RATE].metadata.pop(
            "RepetitionTimePreparation", None
        )
        self.outputs[self.KEY_PERFUSION_RATE].metadata.pop("EchoTime", None)
        self.outputs[self.KEY_PERFUSION_RATE].metadata.pop("M0Type", None)
        self.outputs[self.KEY_PERFUSION_RATE].metadata.pop("FlipAngle", None)
        self.outputs[self.KEY_PERFUSION_RATE].metadata["asl_context"] = "cbf"
        self.outputs[self.KEY_PERFUSION_RATE].metadata["Units"] = "ml/100g/min"
        self.outputs[self.KEY_PERFUSION_RATE].metadata["ImageType"] = [
            "DERIVED",
            "PRIMARY",
            "PERFUSION",
            "RCBF",
        ]

        if self.inputs[self.KEY_MODEL] == self.WHITEPAPER:
            # single subtraction quantification

            # if any of the images are 4D, take the average along the 4th dimension.
            images = {}
            for key in [self.KEY_M0, self.KEY_CONTROL, self.KEY_LABEL]:
                if len(self.inputs[key].shape) == 4:
                    # take the average along the 4th (time) dimension
                    images[key] = np.average(self.inputs[key].image, axis=3)
                else:
                    images[key] = self.inputs[key].image

            if self.inputs[self.KEY_LABEL_TYPE].lower() in [
                GkmFilter.CASL,
                GkmFilter.PCASL,
            ]:
                self.outputs[
                    self.KEY_PERFUSION_RATE
                ].image = AslQuantificationFilter.asl_quant_wp_casl(
                    control=images[self.KEY_CONTROL],
                    label=images[self.KEY_LABEL],
                    m0=images[self.KEY_M0],
                    lambda_blood_brain=self.inputs[self.KEY_LAMBDA_BLOOD_BRAIN],
                    label_duration=self.inputs[self.KEY_LABEL_DURATION],
                    post_label_delay=self.inputs[self.KEY_POST_LABEL_DELAY],
                    label_efficiency=self.inputs[self.KEY_LABEL_EFFICIENCY],
                    t1_arterial_blood=self.inputs[self.KEY_T1_ARTERIAL_BLOOD],
                )

            elif self.inputs[self.KEY_LABEL_TYPE].lower() in [GkmFilter.PASL]:
                self.outputs[
                    self.KEY_PERFUSION_RATE
                ].image = AslQuantificationFilter.asl_quant_wp_pasl(
                    control=images[self.KEY_CONTROL],
                    label=images[self.KEY_LABEL],
                    m0=images[self.KEY_M0],
                    lambda_blood_brain=self.inputs[self.KEY_LAMBDA_BLOOD_BRAIN],
                    bolus_duration=self.inputs[self.KEY_LABEL_DURATION],
                    inversion_time=self.inputs[self.KEY_POST_LABEL_DELAY],
                    label_efficiency=self.inputs[self.KEY_LABEL_EFFICIENCY],
                    t1_arterial_blood=self.inputs[self.KEY_T1_ARTERIAL_BLOOD],
                )
            self.outputs[self.KEY_PERFUSION_RATE].metadata[
                "EstimationAlgorithm"
            ] = self.ESTIMATION_ALGORITHM[self.WHITEPAPER]
        elif self.inputs[self.KEY_MODEL] == self.FULL:
            # fit multi PLD data to the General Kinetic Model
            # AslQuantificationFilter.asl_quant_lsq_gkm requires `t1_tissue` and
            # `lambda_blood_brain` to be np.ndarrays (same dimensions as m0), so
            # first create arrays of these if they are not
            shape = self.inputs[self.KEY_M0].shape
            t1_tissue = GkmFilter.check_and_make_image_from_value(
                self.inputs[self.KEY_T1_TISSUE], shape, {}, None
            )
            lambda_blood_brain = GkmFilter.check_and_make_image_from_value(
                self.inputs[self.KEY_LAMBDA_BLOOD_BRAIN], shape, {}, None
            )
            # The input `post_label_delay` is values of PLD's corresponding to
            # each multiphase index. The actual PLD array needs to be built
            # using this information.
            post_label_delays = [
                self.inputs[self.KEY_POST_LABEL_DELAY][i]
                for i in self.inputs[self.KEY_MULTIPHASE_INDEX]
            ]

            # compute `perfusion_rate` and `transit_time`
            results = AslQuantificationFilter.asl_quant_lsq_gkm(
                control=self.inputs[self.KEY_CONTROL].image,
                label=self.inputs[self.KEY_LABEL].image,
                m0_tissue=self.inputs[self.KEY_M0].image,
                lambda_blood_brain=lambda_blood_brain,
                label_duration=self.inputs[self.KEY_LABEL_DURATION],
                post_label_delay=post_label_delays,
                label_efficiency=self.inputs[self.KEY_LABEL_EFFICIENCY],
                t1_arterial_blood=self.inputs[self.KEY_T1_ARTERIAL_BLOOD],
                t1_tissue=t1_tissue,
                label_type=self.inputs[self.KEY_LABEL_TYPE].lower(),
            )
            self.outputs[self.KEY_PERFUSION_RATE].image = results[
                self.KEY_PERFUSION_RATE
            ]

            self.outputs[self.KEY_PERFUSION_RATE].metadata[
                "EstimationAlgorithm"
            ] = self.ESTIMATION_ALGORITHM[self.FULL]
            self.outputs[self.KEY_PERFUSION_RATE].metadata.pop("MultiphaseIndex", None)
            # when using the full model there are additional outputs
            for key in [
                self.KEY_PERFUSION_RATE_ERR,
                self.KEY_TRANSIT_TIME,
                self.KEY_TRANSIT_TIME_ERR,
                self.KEY_STD_ERROR,
            ]:
                self.outputs[key] = self.outputs[self.KEY_PERFUSION_RATE].clone()
                self.outputs[key].image = results[key]
                self.outputs[key].metadata["asl_context"] = self.FIT_IMAGE_NAME[
                    key
                ].lower()
                self.outputs[key].metadata["Units"] = self.FIT_IMAGE_UNITS[key]
                self.outputs[key].metadata["ImageType"] = [
                    "DERIVED",
                    "PRIMARY",
                    "PERFUSION",
                    self.FIT_IMAGE_NAME[key].upper(),
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
        't1_arterial_blood' must be a float and >0
        't1_tissue' must be a float or BaseImageContainer and >0
        'multiphase_index' must be a list of integers

        'multiphase_index' should match the length of the 4th dimension of
        the 'label' image.
        'multiphase_index' and 't1_tissue' are required if 'model' is 'full'.
        'control' and 'label' must have the same shape
        The shape of 'm0' must match the first 3 dimensions of 'label'

        """

        input_validator = {
            "common": ParameterValidator(
                parameters={
                    self.KEY_M0: Parameter(
                        validators=[isinstance_validator(BaseImageContainer),]
                    ),
                    self.KEY_CONTROL: Parameter(
                        validators=[isinstance_validator(BaseImageContainer),]
                    ),
                    self.KEY_LABEL: Parameter(
                        validators=[isinstance_validator(BaseImageContainer),]
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
                            greater_than_validator(0),
                            isinstance_validator(float),
                        ]
                    ),
                    self.KEY_MODEL: Parameter(
                        validators=from_list_validator(
                            [self.WHITEPAPER, self.FULL], case_insensitive=True,
                        )
                    ),
                },
                post_validators=[
                    shape_validator([self.KEY_CONTROL, self.KEY_LABEL, self.KEY_M0], 3)
                ],
            ),
            "full": ParameterValidator(
                parameters={
                    self.KEY_POST_LABEL_DELAY: Parameter(
                        validators=[
                            for_each_validator(
                                and_validator(
                                    [
                                        greater_than_equal_to_validator(0),
                                        isinstance_validator(float),
                                    ]
                                )
                            )
                        ]
                    ),
                    self.KEY_MULTIPHASE_INDEX: Parameter(
                        validators=[for_each_validator(isinstance_validator(int))]
                    ),
                    self.KEY_T1_TISSUE: Parameter(
                        validators=[
                            isinstance_validator((float, BaseImageContainer)),
                            greater_than_validator(0),
                        ]
                    ),
                },
                post_validators=[shape_validator([self.KEY_CONTROL, self.KEY_LABEL])],
            ),
            "whitepaper": ParameterValidator(
                parameters={
                    self.KEY_POST_LABEL_DELAY: Parameter(
                        validators=[
                            greater_than_equal_to_validator(0),
                            isinstance_validator(float),
                        ]
                    ),
                }
            ),
        }
        # validate the common parameters
        input_validator["common"].validate(
            self.inputs, error_type=FilterInputValidationError
        )
        # validate the model specific parameters
        input_validator[self.inputs[self.KEY_MODEL]].validate(
            self.inputs, error_type=FilterInputValidationError
        )

        # extra validation for the full GKM
        if self.inputs[self.KEY_MODEL] == self.FULL:
            label_shape = self.inputs[self.KEY_LABEL].shape
            # length of 'multiphase_index' should match the length of the label image
            # in the 4th dimension
            if not len(self.inputs[self.KEY_MULTIPHASE_INDEX]) == label_shape[3]:
                raise FilterInputValidationError(
                    "The length of 'multiphase_index' must be equal to the length"
                    "of the 'label' image in the 4th dimension "
                )

            if not len(self.inputs[self.KEY_POST_LABEL_DELAY]) == len(
                set(self.inputs[self.KEY_MULTIPHASE_INDEX])
            ):
                raise FilterInputValidationError(
                    "The length of 'post_label_delay' should be the same as the"
                    "number of unique entries in 'multiphase_index'"
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
        r"""
        Performs ASL quantification using the White Paper equation for
        (pseudo)continuous ASL :cite:p:`Alsop2014`.

        .. math::
            &f = \frac{6000 \cdot\ \lambda \cdot (\text{SI}_{\text{control}} - \text{SI}_{\text{label}}) \cdot
            e^{\frac{\text{PLD}}{T_{1,b}}}}{2 \cdot \alpha \cdot T_{1,b} \cdot \text{SI}_{\text{M0}}
            \cdot (1-e^{-\frac{\tau}{T_{1,b}}})}\\
            \text{where,}\\
            &f = \text{perfusion rate in ml/100g/min}\\
            &\text{SI}_{\text{control}} = \text{control image signal}\\
            &\text{SI}_{\text{label}} = \text{label image signal}\\
            &\text{SI}_{\text{M0}} = \text{equilibrium magnetision signal}\\
            &\tau = \text{label duration}\\
            &\text{PLD} = \text{Post Label Delay}\\
            &T_{1,b} = \text{longitudinal relaxation time of arterial blood}\\
            &\alpha = \text{labelling efficiency}\\
            &\lambda = \text{blood-brain partition coefficient}\\

        :param control: control image, :math:`\text{SI}_{\text{control}}`
        :type control: np.ndarray
        :param label: label image :math:`\text{SI}_{\text{label}}`
        :type label: np.ndarray
        :param m0: equilibrium magnetisation image, :math:`\text{SI}_{\text{M0}}`
        :type m0: np.ndarray
        :param lambda_blood_brain: blood-brain partition coefficient in ml/g, :math:`\lambda`
        :type lambda_blood_brain: float
        :param label_duration: label duration in seconds, :math:`\tau`
        :type label_duration: float
        :param post_label_delay: duration between the end of the label pulse
          and the start of the image acquisition in seconds, :math:`\text{PLD}`
        :type post_label_delay: float
        :param label_efficiency: labelling efficiency, :math:`\alpha`
        :type label_efficiency: float
        :param t1_arterial_blood: longitudinal relaxation time of arterial 
          blood in seconds, :math:`T_{1,b}`
        :type t1_arterial_blood: float
        :return: the perfusion rate in ml/100g/min, :math:`f`
        :rtype: np.ndarray
        """
        control = np.asarray(control)
        label = np.asarray(label)
        m0 = np.asarray(m0)
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
            where=np.abs(m0) >= AslQuantificationFilter.M0_TOL,
        )

    @staticmethod
    def asl_quant_wp_pasl(
        control: np.ndarray,
        label: np.ndarray,
        m0: np.ndarray,
        lambda_blood_brain: float,
        bolus_duration: float,
        inversion_time: float,
        label_efficiency: float,
        t1_arterial_blood: float,
    ) -> np.ndarray:
        r"""
        Performs ASL quantification using the White Paper equation for
        pulsed ASL :cite:p:`Alsop2014`.

        .. math::
            &f = \frac{6000 \cdot\ \lambda \cdot (\text{SI}_{\text{control}}
            - \text{SI}_{\text{label}}) \cdot
            e^{\frac{\text{TI}}{T_{1,b}}}}{2 \cdot \alpha \cdot \text{TI}_1
            \cdot \text{SI}_{\text{M0}}}\\
            \text{where,}\\
            &f = \text{perfusion rate in ml/100g/min}\\
            &\text{SI}_{\text{control}} = \text{control image signal}\\
            &\text{SI}_{\text{label}} = \text{label image signal}\\
            &\text{SI}_{\text{M0}} = \text{equilibrium magnetision signal}\\
            &\text{TI} = \text{inversion time}\\
            &\text{TI}_1 = \text{bolus duration}\\
            &T_{1,b} = \text{longitudinal relaxation time of arterial blood}\\
            &\alpha = \text{labelling efficiency}\\
            &\lambda = \text{blood-brain partition coefficient}\\

        :param control: control image, :math:`\text{SI}_{\text{control}}`
        :type control: np.ndarray
        :param label: label image, :math:`\text{SI}_{\text{label}}`
        :type label: np.ndarray
        :param m0: equilibrium magnetisation image, :math:`\text{SI}_{\text{M0}}`
        :type m0: np.ndarray
        :param lambda_blood_brain: blood-brain partition coefficient in ml/g,
          :math:`\lambda`
        :type lambda_blood_brain: float
        :param inversion_time: time between the inversion pulse and the start
          of the image acquisition in seconds, :math:`\text{TI}`
        :type inversion_time: float
        :param bolus_duration: temporal duration of the labelled bolus in
          seconds, defined as the duration between the inversion pulse and
          the start of the bolus cutoff pulses (QUIPPSS, Q2-TIPS etc),
          :math:`\text{TI}_1`
        :type bolus_duration: float
        :param label_efficiency: labelling efficiency, :math:`\alpha`
        :type label_efficiency: float
        :param t1_arterial_blood: longitudinal relaxation time of arterial 
          blood in seconds, :math:`T_{1,b}`
        :type t1_arterial_blood: float
        :return: the perfusion rate in ml/100g/min, :math:`f`
        :rtype: np.ndarray
        """
        control = np.asarray(control)
        label = np.asarray(label)
        m0 = np.asarray(m0)
        return np.divide(
            6000
            * lambda_blood_brain
            * (control - label)
            * np.exp(inversion_time / t1_arterial_blood),
            2 * label_efficiency * bolus_duration * m0,
            out=np.zeros_like(m0),
            where=np.abs(m0) >= AslQuantificationFilter.M0_TOL,
        )

    @staticmethod
    def asl_quant_lsq_gkm(
        control: np.ndarray,
        label: np.ndarray,
        m0_tissue: np.ndarray,
        lambda_blood_brain: np.ndarray,
        label_duration: float,
        post_label_delay: np.ndarray or List[float],
        label_efficiency: float,
        t1_arterial_blood: float,
        t1_tissue: np.ndarray,
        label_type: str,
    ) -> dict:
        """Calculates the perfusion and transit time by least-squares
        fitting to the ASL General Kinetic Model :cite:p:`Buxton1998`.

        Fitting is performed using :class:`scipy.optimize.curve_fit`.

        See :class:`.GkmFilter` and :class:`.GkmFilter.calculate_delta_m_gkm` for
        implementation details of the GKM function.

        :param control: control signal, must be 4D with signal for each
            post labelling delay on the 4th axis. Must have same dimensions as ``label``.
        :type control: np.ndarray
        :param label: label signal, must be 4D with signal for each post
            labelling delay on the 4th axis. Must have same dimensions as ``control``.
        :type label: np.ndarray
        :param m0_tissue: equilibrium magnetisation of the tissue.
        :type m0_tissue: np.ndarray
        :param lambda_blood_brain: tissue partition coefficient in g/ml.
        :type lambda_blood_brain: np.ndarray
        :param label_duration: duration of the labelling pulse in seconds.
        :type label_duration: float
        :param post_label_delay: array of post label delays, must be equal in
            length to the number of 3D volumes in ``control`` and ``label``.
        :type post_label_delay: np.ndarray
        :param label_efficiency: The degree of inversion of the labelling pulse.
        :type label_efficiency: float
        :param t1_arterial_blood: Longitudinal relaxation time of the arterial
            blood in seconds.
        :type t1_arterial_blood: float
        :param t1_tissue: Longitudinal relaxation time of the tissue in seconds.
        :type t1_tissue: np.ndarray
        :param label_type: The type of labelling: pulsed ('pasl') or continuous
            ('casl' or 'pcasl').
        :type label_type: str
        :return: A dictionary containing the following np.ndarrays:

            :'perfusion_rate': The estimated perfusion rate in ml/100g/min.
            :'transit_time': The estimated transit time in seconds.
            :'std_error': The standard error of the estimate of the fit.
            :'perfusion_rate_err': One standard deviation error in the fitted
              perfusion rate.
            :'transit_time_err': One standard deviation error in the fitted
              transit time.

        :rtype: dict

        ``control``, ``label``, ``m0_tissue``, ``t1_tissue`` and 
        ``lambda_blood_brain`` must all have
        the same dimensions for the first 3 dimensions.


        """
        np.broadcast(m0_tissue, t1_tissue, lambda_blood_brain)
        np.broadcast(control, label)

        post_label_delay = np.asarray(post_label_delay)

        # subtract to get delta_m
        delta_m = control - label
        I, J, K = delta_m.shape[:3]
        perfusion_rate = np.zeros((I, J, K))
        transit_time = np.zeros((I, J, K))
        std_error = np.zeros((I, J, K))
        perfusion_rate_err = np.zeros((I, J, K))
        transit_time_err = np.zeros((I, J, K))
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    # create an anonymous version of the function to solve
                    func = lambda plds, perf, att: np.array(
                        [
                            GkmFilter.calculate_delta_m_gkm(
                                perf,
                                att,
                                m0_tissue[i, j, k]
                                if isinstance(m0_tissue, np.ndarray)
                                else m0_tissue,
                                label_duration,
                                label_duration + pld,
                                label_efficiency,
                                lambda_blood_brain[i, j, k]
                                if isinstance(lambda_blood_brain, np.ndarray)
                                else lambda_blood_brain,
                                t1_arterial_blood,
                                t1_tissue[i, j, k]
                                if isinstance(t1_tissue, np.ndarray)
                                else t1_tissue,
                                label_type,
                            )
                            for pld in plds
                        ]
                    )
                    # fit for the perfusion rate and transit time
                    obs = delta_m[i, j, k, :]
                    popt, pcov = curve_fit(func, post_label_delay, obs)
                    perfusion_rate[i, j, k] = popt[0]
                    transit_time[i, j, k] = popt[1]
                    std_error[i, j, k] = np.sqrt(
                        np.sum((obs - func(post_label_delay, *popt)) ** 2)
                        / post_label_delay.size
                    )
                    # compute one standard deviation errors of the parameters
                    perr = np.sqrt(np.diag(pcov))
                    perfusion_rate_err = perr[0]
                    transit_time_err = perr[1]

        return {
            "perfusion_rate": perfusion_rate,
            "transit_time": transit_time,
            "perfusion_rate_err": perfusion_rate_err,
            "transit_time_err": transit_time_err,
            "std_error": std_error,
        }
