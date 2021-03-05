"""Utility functions for testing filters"""
from copy import deepcopy
import pytest
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError


def validate_filter_inputs(filter_to_test: BaseFilter, validation_data: dict):
    """Tests a filter with a validation data dictionary.  Checks that FilterInputValidationErrors
    are raised when data is missing or incorrect.

    :param filter_to_test: the class of the filter to test
    :type filter_to_test: BaseFilter
    :param validation_data: A dictionary, where each key is an input parameter
      for the filter, and the value is a list/tuple where:
        
        :[0]: is_optional
        :[1]: a value that should pass
        :[2:end]: values that should fail

    :type validation_data: dict
    """
    test_filter = filter_to_test()
    test_data = deepcopy(validation_data)
    # check with inputs that should pass
    for data_key in test_data:
        test_filter.add_input(data_key, test_data[data_key][1])
    test_filter.run()

    for inputs_key in validation_data:
        test_data = deepcopy(validation_data)
        test_filter = filter_to_test()
        is_optional: bool = test_data[inputs_key][0]

        # remove key
        test_data.pop(inputs_key)
        for data_key in test_data:
            test_filter.add_input(data_key, test_data[data_key][1])

        # optional inputs should run without issue
        if is_optional:
            test_filter.run()
        else:
            with pytest.raises(FilterInputValidationError):
                test_filter.run()

        # Try data that should fail
        for test_value in validation_data[inputs_key][2:]:
            test_filter = filter_to_test()
            for data_key in test_data:
                test_filter.add_input(data_key, test_data[data_key][1])
            test_filter.add_input(inputs_key, test_value)

            with pytest.raises(FilterInputValidationError):
                test_filter.run()
