""" PhaseMagnitudeFilter Class"""

import numpy as np
from asldro.containers.image import (
    BaseImageContainer,
    COMPLEX_IMAGE_TYPE,
    REAL_IMAGE_TYPE,
    IMAGINARY_IMAGE_TYPE,
    MAGNITUDE_IMAGE_TYPE,
    PHASE_IMAGE_TYPE,
)
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError

from asldro.validators.parameters import (
    ParameterValidator,
    Parameter,
    isinstance_validator,
    has_attribute_value_validator,
)


class PhaseMagnitudeFilter(BaseFilter):
    r"""A filter block that will take complex image data and convert it into
    its Phase and Magnitude components. Typically, this will be used after
    a :class:`.AcquireMriImageFilter` which contains real and imaginary
    components

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`PhaseMagnitudeFilter.add_inputs()`
    member function. They are also accessible via class constants,
    for example :class:`AcquireMriImageFilter.KEY_IMAGE`

    :param 'image': The input data image, cannot be a phase image
    :type 'image': BaseImageContainer

    **Outputs**

    :param 'phase': Phase image (will have `image_type`==PHASE_IMAGE_TYPE)
    :type 'phase': BaseImageContainer

    :param 'magnitude': Magnitude image (will have `image_type`==MAGNITUDE_IMAGE_TYPE)
    :type 'magnitude': BaseImageContainer
    """

    # KEY_CONSTANTS
    KEY_IMAGE = "image"
    KEY_PHASE = "phase"
    KEY_MAGNITUDE = "magnitude"

    def __init__(self):
        super().__init__(name="Phase-Magnitude Image Filter")

    def _run(self):
        """Calculate the phase and magnitude components from the
        input complex data and return them as separate image containers
        """
        image: BaseImageContainer = self.inputs["image"]
        phase = image.clone()
        phase.image_type = PHASE_IMAGE_TYPE
        magnitude = image.clone()
        magnitude.image_type = MAGNITUDE_IMAGE_TYPE

        # do the same for both complex and real `image_type`
        if image.image_type in [COMPLEX_IMAGE_TYPE, REAL_IMAGE_TYPE]:
            phase.image = np.angle(image.image)
            magnitude.image = np.absolute(image.image)

        # apply a phase shift of 90° when calculating the phase
        elif image.image_type == IMAGINARY_IMAGE_TYPE:
            phase.image = np.angle(image.image * np.exp(1j * np.pi / 2))
            magnitude.image = np.absolute(image.image)

        # phase is undefined if only magnitude is supplied
        elif image.image_type == MAGNITUDE_IMAGE_TYPE:
            magnitude.image = np.absolute(image.image)
            phase = None

        self.outputs[self.KEY_PHASE] = phase
        self.outputs[self.KEY_MAGNITUDE] = magnitude

    def _validate_inputs(self):
        """Validate the inputs.
        'image' must be derived from BaseImageContainer and have
        a COMPLEX `image_type` instance variable.
        """
        input_validator = ParameterValidator(
            parameters={
                self.KEY_IMAGE: Parameter(
                    validators=[isinstance_validator(BaseImageContainer),]
                )
            }
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)

        # raise an error if the input image is a phase image.
        if self.inputs[self.KEY_IMAGE].image_type == PHASE_IMAGE_TYPE:
            raise FilterInputValidationError(
                "input 'image' has attribute 'image_type' with value 'PHASE_IMAGE_TYPE', this is"
                "not supported"
            )
