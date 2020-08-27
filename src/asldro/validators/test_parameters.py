""" parameters.py tests """
import pytest
from asldro.validators.parameters import (
    Parameter,
    ParameterValidator,
    ValidationError,
    range_exclusive_validator,
    range_inclusive_validator,
    from_list_validator,
    list_of_type_validator,
    non_empty_list_validator,
    regex_validator,
    reserved_string_list_validator,
    greater_than_validator,
    greater_than_equal_to_validator,
    for_each_validator,
)


def test_range_inclusive_validator_creator():
    """ Check the inclusive validator creator raises
    errors when start >end """

    with pytest.raises(ValueError):
        range_inclusive_validator(2, 1)


def test_range_inclusive_validator():
    """ Test the inclusive validator with some values """
    validator = range_inclusive_validator(-1, 1)
    assert str(validator) == "Value must be between -1 and 1 (inclusive)"
    assert validator(-0.99)
    assert validator(0.99)
    assert validator(-1)
    assert validator(1)
    assert not validator(-1.01)
    assert not validator(1.01)
    assert not validator("not a number")


def test_range_exclusive_validator_creator():
    """ Check the exclusive validator creator raises
    errors when start >= end """
    with pytest.raises(ValueError):
        range_exclusive_validator(1, 1)
    with pytest.raises(ValueError):
        range_exclusive_validator(2, 1)


def test_range_exclusive_validator():
    """ Test the exclusive validator with some values """
    validator = range_exclusive_validator(-1, 1)
    assert str(validator) == "Value must be between -1 and 1 (exclusive)"
    assert validator(-0.99)
    assert validator(0.99)
    assert not validator(-1)
    assert not validator(1)
    assert not validator("not a number")


def test_greater_than_validator_creator():
    """ Check the greater_than_validator creator raises
    errors when start is not a number type """
    with pytest.raises(TypeError):
        greater_than_validator("str")
    with pytest.raises(TypeError):
        greater_than_validator([])


def test_greater_than_validator():
    """ Test the greater_than_validator with some values """
    validator = greater_than_validator(100)
    assert str(validator) == "Value must be greater than 100"
    assert validator(101)
    assert validator(1000)
    assert validator(float("inf"))
    assert not validator(99)
    assert not validator(100)
    assert not validator(float("-inf"))
    assert not validator("not a number")


def test_greater_than_equal_to_validator_creator():
    """ Check the greater_than_equal_to_validator creator raises
    errors when start is not a number type """
    with pytest.raises(TypeError):
        greater_than_equal_to_validator("str")
    with pytest.raises(TypeError):
        greater_than_equal_to_validator([])


def test_greater_than_equal_to_validator():
    """ Test the greater_than_equal_to_validator with some values """
    validator = greater_than_equal_to_validator(100)
    assert str(validator) == "Value must be greater than or equal to 100"
    assert validator(101)
    assert validator(1000)
    assert validator(float("inf"))
    assert not validator(99)
    assert validator(100)
    assert not validator(float("-inf"))
    assert not validator("not a number")


def test_from_list_validator_creator():
    """ The the from list validator creation"""
    with pytest.raises(TypeError):
        from_list_validator("foo")
    with pytest.raises(TypeError):
        from_list_validator(1)
    with pytest.raises(TypeError):
        from_list_validator({})


def test_from_list_validator():
    """ Test the from list validator """
    validator = from_list_validator(["FOO", "BAR", "foo"])
    assert str(validator) == "Value must be in ['FOO', 'BAR', 'foo']"
    assert validator("FOO")
    assert validator("BAR")
    assert validator("foo")
    assert not validator("bar")
    assert not validator(["FOO"])
    assert not validator({})
    assert not validator(1)


def test_list_of_type_validator_creator():
    """ Test the list of validator creation"""
    with pytest.raises(TypeError):
        list_of_type_validator(1)
    with pytest.raises(TypeError):
        list_of_type_validator("foo")
    with pytest.raises(TypeError):
        list_of_type_validator([])
    with pytest.raises(TypeError):
        list_of_type_validator({})


def test_list_of_type_validator():
    """ Test the list of type validator """
    validator = list_of_type_validator(str)
    assert str(validator) == "Value must be a list of type str"
    assert validator([])
    assert validator(["foo"])
    assert validator(["foo", "bar"])
    assert not validator([1])
    assert not validator([1, 2])

    validator = list_of_type_validator(int)
    assert str(validator) == "Value must be a list of type int"
    assert validator([])
    assert validator([1])
    assert validator([1, 2])
    assert not validator(["foo"])
    assert not validator(["foo", "bar"])
    assert not validator([1.0, 2.0])

    validator = list_of_type_validator(float)
    assert str(validator) == "Value must be a list of type float"
    assert validator([])
    assert validator([1.0])
    assert validator([1.0, 2.0])
    assert not validator(["foo"])
    assert not validator(["foo", "bar"])
    assert not validator([1, 2])
    assert not validator([1])

    validator = list_of_type_validator((float, int))
    assert str(validator) == "Value must be a list of type float or int"
    assert validator([])
    assert validator([1.0])
    assert validator([1.0, 2.0])
    assert validator([1, 2])
    assert validator([1])
    assert not validator(["foo"])
    assert not validator(["foo", "bar"])


def test_non_empty_list_validator():
    """ Test the non-empty list validator """
    validator = non_empty_list_validator()
    assert str(validator) == "Value must be a non-empty list"
    assert validator([1, 2, 3])
    assert validator(["foo", "bar"])
    assert not validator(1)
    assert not validator("foo")
    assert not validator([])


def test_regex_validator_creator():
    """ Test the regex validator creator """
    with pytest.raises(ValueError):
        regex_validator("[")  # bad regex


def test_regex_validator():
    """ Test the regex validator """
    validator = regex_validator(r"^M*(s|t)$")
    assert str(validator) == "Value must match pattern ^M*(s|t)$"
    assert validator("s")
    assert validator("t")
    assert validator("MMMMMMt")
    assert validator("Ms")
    assert not validator("foo")
    assert not validator("")


def test_reserved_string_list_validator_creator():
    """ Test the reserved string list validator creator """
    with pytest.raises(TypeError):
        reserved_string_list_validator(1)

    with pytest.raises(ValueError):
        reserved_string_list_validator([])

    with pytest.raises(ValueError):
        reserved_string_list_validator(["foo", 1])


def test_reserved_string_list_validator():
    """ Test the reserved string list validator """
    validator = reserved_string_list_validator(
        strings=["M0", "CONTROL", "LABEL"], delimiter="_"
    )
    assert (
        str(validator)
        == "Value must be a string combination of ['M0', 'CONTROL', 'LABEL'] separated by '_'"
    )
    assert validator("M0")
    assert validator("CONTROL")
    assert validator("LABEL_CONTROL")
    assert validator("M0_LABEL_CONTROL")
    assert not validator("foo")
    assert not validator("M0_foo")


def test_for_each_validator_creator():
    """ Test the for each validator creator """
    with pytest.raises(TypeError):
        for_each_validator(item_validator="not a validator")

    # An uninitialised validator
    with pytest.raises(TypeError):
        for_each_validator(item_validator=range_inclusive_validator)


def test_for_each_validator():
    """ Test the for each validator """

    validator = for_each_validator(greater_than_validator(0.5))
    assert (
        str(validator)
        == "Must be a list and for each value in the list: Value must be greater than 0.5"
    )
    assert validator([0.6, 0.7, 0.8])
    assert not validator([0.5, 0.6, 0.7])
    assert not validator([0.1, 100, 0.9])
    assert not validator("not a list")


def test_parameter_validator_valid():
    """ Test the parameter validator with some valid example data """
    parameter_validator = ParameterValidator(
        {
            "foo": Parameter(reserved_string_list_validator(["foo", "bar"])),
            "bar": Parameter(non_empty_list_validator(), default_value=[1, 2, 3]),
        }
    )
    assert parameter_validator.validate({"foo": "bar foo bar"}) == {
        "foo": "bar foo bar",
        "bar": [1, 2, 3],
    }  # no ValidationError raised


def test_parameter_validator_valid_with_optional_parameters():
    """ Test the parameter validator with some valid example data
    including a (missing) optional parameter """
    parameter_validator = ParameterValidator(
        {
            "foo": Parameter(reserved_string_list_validator(["foo", "bar"])),
            "bar": Parameter(non_empty_list_validator(), optional=True),
        }
    )
    assert parameter_validator.validate({"foo": "bar foo bar"}) == {
        "foo": "bar foo bar"
    }  # no ValidationError raised


def test_parameter_validator_missing_required():
    """ Test the parameter validator with some valid example data """
    parameter_validator = ParameterValidator(
        {
            "foo": Parameter(reserved_string_list_validator(["foo", "bar"])),
            "bar": Parameter(non_empty_list_validator()),
        }
    )

    with pytest.raises(
        ValidationError,
        match="bar is a required parameter and is not in the input dictionary",
    ):
        parameter_validator.validate({"foo": "bar foo bar"})


def test_parameter_validator_multiple_validators():
    """ Test the parameter validator with multiple validators """
    parameter_validator = ParameterValidator(
        {
            "a_number": Parameter(
                [range_inclusive_validator(1, 2), from_list_validator([1.5, 1.6])]
            )
        }
    )
    parameter_validator.validate({"a_number": 1.5})  # OK

    with pytest.raises(
        ValidationError,
        match=r"Parameter a_number with value 1.7 does not meet the following criterion: Value must be in \[1.5, 1.6\]",
    ):
        parameter_validator.validate({"a_number": 1.7})


def test_parameter_validator_multiple_errors():
    """ Test that multiple errors are correctly reported by the validator """
    parameter_validator = ParameterValidator(
        {
            "a_number": Parameter(
                [range_inclusive_validator(1, 2), from_list_validator([1.5, 1.6])]
            ),
            "b_number": Parameter(list_of_type_validator(str)),
        }
    )
    parameter_validator.validate({"a_number": 1.5, "b_number": ["foo", "bar"]})  # OK

    with pytest.raises(
        ValidationError,
        match=r"^Parameter a_number with value 0.9 does not meet the following criterion: Value must be between 1 and 2 \(inclusive\)\. Parameter a_number with value 0.9 does not meet the following criterion: Value must be in \[1.5, 1.6\]\. Parameter b_number with value \[1, 2\] does not meet the following criterion: Value must be a list of type str$",
    ):
        parameter_validator.validate({"a_number": 0.9, "b_number": [1, 2]})

