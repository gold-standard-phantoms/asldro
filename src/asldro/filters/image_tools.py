"""Filters for basic image container manipulation and maths"""
import numpy as np

from asldro.containers.image import BaseImageContainer
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.validators.parameters import (
    Parameter,
    ParameterValidator,
    from_list_validator,
    isinstance_validator,
)


class FloatToIntImageFilter(BaseFilter):
    """ A filter which converts image data from float to integer.

    **Inputs**

    Input Parameters are all keyword arguments for the 
    :class:`FloatToIntImageFilter.add_input()`
    member function. They are also accessible via class constants,
    for example :class:`FloatToIntImageFilter.KEY_IMAGE`

    :param 'image': Image to convert from float to integer. The dtype of
      the image data must be float.
    :type 'image': BaseImageContainer
    :param 'method': Defines which method to use for conversion:

        * "round": returns the nearest integer
        * "floor": returns the largest integer that is less than the input value.
        * "ceil":  returns the smallest integer that is greater than the input value.
        * "truncate": Removes the decimal portion of the number. This will round
          down for positive numbers and up for negative.
    
    **Outputs** 

    Once run, the filter will populate the dictionary
    :class:`FloatToIntImageFilter.outputs` with the following entries

    :param 'image': The input image, with the image data as integer type.
    :type 'image': BaseImageContainer
    """

    KEY_IMAGE = "image"
    KEY_METHOD = "method"

    ROUND = "round"
    FLOOR = "floor"
    CEIL = "ceil"
    TRUNCATE = "truncate"
    METHODS = [ROUND, FLOOR, CEIL, TRUNCATE]

    def __init__(self):
        super().__init__("FloatToIntImageFilter")

    def _run(self):
        """ Convert the image's data from float to integer """
        image: BaseImageContainer = self.inputs[self.KEY_IMAGE]
        method = self.inputs[self.KEY_METHOD]
        self.outputs[self.KEY_IMAGE] = image.clone()

        # only convert if the input image is a float
        if self.inputs[self.KEY_IMAGE].image.dtype.kind == "f":

            data_type = ""
            # determine the best type to use
            if np.any(image.image < 0):
                data_type = np.int16
                if np.amax(np.absolute(image.image)) > 2 ** 15 - 1:
                    data_type = np.int32
                    if np.amax(np.absolute(image.image)) > 2 ** 31 - 1:
                        data_type = np.int64
            else:
                data_type = np.uint16
                if np.amax(np.absolute(image.image)) > 2 ** 16 - 1:
                    data_type = np.uint32
                    if np.amax(np.absolute(image.image)) > 2 ** 32 - 1:
                        data_type = np.uint64

            if method == self.ROUND:
                self.outputs[self.KEY_IMAGE].image = np.rint(
                    self.outputs[self.KEY_IMAGE].image
                ).astype(data_type)
            elif method == self.CEIL:
                self.outputs[self.KEY_IMAGE].image = np.ceil(
                    self.outputs[self.KEY_IMAGE].image
                ).astype(data_type)
            elif method == self.FLOOR:
                self.outputs[self.KEY_IMAGE].image = np.floor(
                    self.outputs[self.KEY_IMAGE].image
                ).astype(data_type)
            elif method == self.TRUNCATE:
                self.outputs[self.KEY_IMAGE].image = self.outputs[
                    self.KEY_IMAGE
                ].image.astype(data_type)

    def _validate_inputs(self):
        """Checks that the inputs meet their validation criteria
        'image' must be derived from BaseImageContainer and have image data
        that is a float
        'method' must be 
        
        """

        input_validator = ParameterValidator(
            parameters={
                self.KEY_IMAGE: Parameter(
                    validators=isinstance_validator(BaseImageContainer)
                ),
                self.KEY_METHOD: Parameter(
                    validators=from_list_validator(self.METHODS),
                    default_value=self.ROUND,
                ),
            }
        )

        new_params = input_validator.validate(
            self.inputs, error_type=FilterInputValidationError
        )

        if not issubclass(
            self.inputs[self.KEY_IMAGE].image.dtype.type, (np.floating, np.integer)
        ):
            raise FilterInputValidationError(
                "the dtype of the input 'image' must be floating point"
                f"data type is {self.inputs[self.KEY_IMAGE].image.dtype}"
            )

        # merge the updated parameters from the output with the input parameters
        self.inputs = {**self._i, **new_params}
