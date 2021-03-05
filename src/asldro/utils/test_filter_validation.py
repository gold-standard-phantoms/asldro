"""tests for filter_validation.py"""

import pytest
import numpy as np
from asldro.utils.filter_validation import validate_filter_inputs
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.validators.parameters import (
    Parameter,
    ParameterValidator,
    isinstance_validator,
)


class ProductFilter(BaseFilter):
    """A filter that multiplies its inputs together
    
    input1: float
    input2: float
    input3: float, optional

    the output is called `product`
    """

    def __init__(self):
        super().__init__(name="ProductFilter")

    def _run(self):
        """Multiplies all inputs and creates an `output` with the result"""
        self.outputs["product"] = np.prod(self.inputs.values())

    def _validate_inputs(self):
        input_validator = ParameterValidator(
            parameters={
                "input1": Parameter(validators=isinstance_validator(float)),
                "input2": Parameter(validators=isinstance_validator(float)),
                "input3": Parameter(
                    validators=isinstance_validator(float), optional=True
                ),
            }
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)


def test_validate_filter_inputs_function():
    validation_data = {
        "input1": [False, 10.0, 20, "str"],
        "input2": [False, 20.0, 20, "str"],
        "input3": [True, 15.0, 15, "str"],
    }

    validate_filter_inputs(ProductFilter, validation_data)
