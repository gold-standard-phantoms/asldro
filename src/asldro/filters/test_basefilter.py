""" BaseFilter tests """

from unittest.mock import call
import pytest
from asldro.filters.basefilter import (
    BaseFilter,
    FilterInputKeyError,
    FilterInputValidationError,
    FilterLoopError,
)


class BaseFilterTester(BaseFilter):
    """
    We shouldn't instantiate an abstact class
    """

    def __init__(self):
        super().__init__(name="BaseFilterTester")

    def _run(self):
        """ Dummy run """

    def _validate_inputs(self):
        """ Dummy run """


def test_filter_name():
    """ Test the name of the filter is set """
    base_filter = BaseFilterTester()
    assert base_filter.name == "BaseFilterTester"


def test_add_same_input():
    """
    Check that adding two inputs with the same key raises and error
    """
    base_filter = BaseFilterTester()
    base_filter.add_input("key", 1)

    with pytest.raises(FilterInputKeyError):
        base_filter.add_input("key", 2)


class SumFilter(BaseFilter):
    """A simple adder filter which create a single output called
    `sum` and adds all of the inputs
    """

    def __init__(self):
        super().__init__(name="SumFilter")

    def _run(self):
        """Adds all inputs and creates an `output` with the result"""
        self.outputs["sum"] = sum(self.inputs.values())

    def _validate_inputs(self):
        """ All inputs must be integers or floats """
        for input_key, input_value in self.inputs.items():
            if not isinstance(input_value, (int, float)):
                raise FilterInputValidationError(
                    f"Input {input_key} is not a float or int (is {type(input_value)})"
                )


def test_validate_inputs():
    """ Filter should only allow int or float inputs """
    sum_filter = SumFilter()
    sum_filter.add_input("input1", 5)
    sum_filter.run()
    sum_filter.add_input("input2", 10.5)
    sum_filter.run()
    with pytest.raises(FilterInputValidationError):
        sum_filter.add_input("input3", "str")
        sum_filter.run()


def test_simple_sum_filter():
    """ Filter should add all inputs """
    filter_a = SumFilter()
    filter_a.add_input("input_a", 5)
    filter_a.add_input("input_b", 10)
    filter_a.add_input("input_c", 3)
    filter_a.run()
    assert filter_a.outputs == {"sum": 18}


def test_input_input_filter_key_clash_error():
    """A FilterInputKeyError should be raised when an output is mapped to an input
    using the same name as an existing input"""
    filter_a = SumFilter()
    filter_a.add_input("a", 1)
    filter_b = SumFilter()
    filter_b.add_input("a", 1)
    filter_b.add_parent_filter(parent=filter_a, io_map={"sum": "a"})
    with pytest.raises(FilterInputKeyError):
        filter_b.run()


def test_input_filter_input_filter_key_clash_error():
    """A FilterInputKeyError should be raised when an output is mapped to an input
    filter using the same name as an input filter"""
    filter_a = SumFilter()
    filter_a.add_input("a", 1)
    filter_b = SumFilter()
    filter_b.add_input("a", 1)
    filter_c = SumFilter()
    filter_c.add_parent_filter(parent=filter_a, io_map={"sum": "input"})
    filter_c.add_parent_filter(parent=filter_b, io_map={"sum": "input"})
    with pytest.raises(FilterInputKeyError):
        filter_c.run()


def test_chained_sum_filter():
    """
    Test a more complex chain of sum filters
    A---+----+
        ˅    ˅
        C--->D
        ^
    B---+
    """
    filter_a = SumFilter()
    filter_a.name = "filter_a"
    filter_a.add_input("a", 1)
    filter_a.add_input("b", 2)
    filter_a.add_input("c", 3)

    filter_b = SumFilter()
    filter_b.name = "filter_b"
    filter_b.add_input("d", 4)
    filter_b.add_input("e", 5)
    filter_b.add_input("f", 6)

    filter_c = SumFilter()
    filter_c.name = "filter_c"
    filter_c.add_input("g", 7)

    filter_d = SumFilter()
    filter_d.name = "filter_d"
    filter_d.add_input("h", 8)

    filter_c.add_parent_filter(filter_a, {"sum": "input_sum_a"})
    filter_c.add_parent_filter(filter_b, {"sum": "input_sum_b"})

    filter_c.add_child_filter(filter_d, {"sum": "input_sum_c"})
    filter_a.add_child_filter(filter_d, {"sum": "input_sum_a"})

    # Running filter_d should run all filters
    filter_d.run()

    assert filter_a.outputs == {"sum": 6}
    assert filter_b.outputs == {"sum": 15}
    assert filter_c.outputs == {"sum": 28}
    assert filter_d.outputs == {"sum": 42}


def test_loop_handling():
    """ If the filters are chained in a loop, check this is managed gracefully """
    filter_a = SumFilter()
    filter_a.add_input("a", 1)
    filter_b = SumFilter()
    filter_b.add_input("b", 1)

    filter_a.add_parent_filter(parent=filter_b)
    filter_b.add_parent_filter(parent=filter_a)

    with pytest.raises(FilterLoopError):
        filter_b.run()


def test_basefilter_add_inputs(mocker):
    """ Test the add_inputs function """
    mocker.patch.object(SumFilter, "add_input")
    filter_a = SumFilter()
    filter_a.add_inputs({"one": "two", "three": "four"})
    calls = [call("one", "two"), call("three", "four")]
    filter_a.add_input.assert_has_calls(calls=calls, any_order=False)
    assert filter_a.add_input.call_count == 2


def test_basefilter_add_inputs_with_non_optional_iomap(mocker):
    """Test the add_inputs function with an non-optional io_map.
    A KeyError should be raised if the key doesn't exist."""
    mocker.patch.object(SumFilter, "add_input")
    filter_a = SumFilter()
    with pytest.raises(KeyError):
        filter_a.add_inputs(
            {"one": "two", "three": "four"},
            io_map={"one": "eno", "thiskeydoesntexist": "OK"},
        )


def test_basefilter_add_inputs_with_iomap(mocker):
    """ Test the add_inputs function with an optional io_map """
    mocker.patch.object(SumFilter, "add_input")
    filter_a = SumFilter()
    filter_a.add_inputs(
        {"one": "two", "three": "four"},
        io_map={"one": "eno", "thiskeydoesntexist": "OK"},
        io_map_optional=True,
    )
    calls = [call("eno", "two")]
    filter_a.add_input.assert_has_calls(calls=calls, any_order=False)
    assert filter_a.add_input.call_count == 1
