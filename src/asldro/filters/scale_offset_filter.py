""" ScaleOffsetFilter Class"""

import numpy as np
from asldro.containers.image import BaseImageContainer
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError

from asldro.validators.parameters import (
    ParameterValidator,
    Parameter,
    isinstance_validator,
)


class ScaleOffsetFilter(BaseFilter):
    r"""A filter that will take image data and apply a scale and/or offset according to the
    equation:

    .. math::
        I_{output} = I_{input} * m + b

    where 'm' is the scale and 'b' is the offset (scale first then offset)

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`ScaleOffsetFilter.add_inputs()`
    member function. They are also accessible via class constants,
    for example :class:`ScaleOffsetFilter.KEY_IMAGE`

    :param 'image': The input image
    :type 'image': BaseImageContainer
    :param 'scale': (optional) a scale to apply
    :type 'scale': float / int
    :param 'offset': (optional) an offset to apply
    :type 'offset': float / int

    **Outputs**

    :param 'image': The output image
    :type 'image': BaseImageContainer
    """

    # KEY_CONSTANTS
    KEY_IMAGE = "image"
    KEY_SCALE = "scale"
    KEY_OFFSET = "offset"

    def __init__(self):
        super().__init__(name="Scale/offset filter")

    def _run(self):
        """Apply the scaling and offset to the image"""
        image: BaseImageContainer = self.inputs["image"]

        output = image.clone()
        if self.KEY_SCALE in self.inputs:
            # Allow unsafe casting (allow data-type conversion)
            output.image = np.multiply(
                output.image, self.inputs[self.KEY_SCALE], casting="unsafe"
            )
        if self.KEY_OFFSET in self.inputs:
            # Allow unsafe casting (allow data-type conversion)
            output.image = np.add(
                output.image, self.inputs[self.KEY_OFFSET], casting="unsafe"
            )

        self.outputs[self.KEY_IMAGE] = output

    def _validate_inputs(self):
        """Validate the inputs.
        'image' must be derived from BaseImageContainer
        'scale' (optional) must be a float or integer
        'offset' (optional) must be a float or integer
        """
        input_validator = ParameterValidator(
            parameters={
                self.KEY_IMAGE: Parameter(
                    validators=[
                        isinstance_validator(BaseImageContainer),
                    ]
                ),
                self.KEY_SCALE: Parameter(
                    validators=[isinstance_validator((int, float))], optional=True
                ),
                self.KEY_OFFSET: Parameter(
                    validators=[isinstance_validator((int, float))], optional=True
                ),
            }
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)
