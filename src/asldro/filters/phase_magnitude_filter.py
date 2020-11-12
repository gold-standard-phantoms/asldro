""" PhaseMagnitudeFilter Class"""

import numpy as np
from asldro.containers.image import (
    BaseImageContainer,
    COMPLEX_IMAGE_TYPE,
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

    :param 'image': The complex input image data
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
        phase.image = np.angle(phase.image)
        phase.image_type = PHASE_IMAGE_TYPE

        magnitude = image.clone()
        magnitude.image = np.absolute(magnitude.image)
        magnitude.image_type = MAGNITUDE_IMAGE_TYPE

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
                    validators=[
                        isinstance_validator(BaseImageContainer),
                        has_attribute_value_validator("image_type", COMPLEX_IMAGE_TYPE),
                    ]
                )
            }
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)
