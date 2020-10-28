""" BaseFilter classes and exception handling """

import copy
import logging
from typing import List, Mapping, Any
from abc import ABC, abstractmethod

from asldro.utils.general import map_dict

logger = logging.getLogger(__name__)


class BaseFilterException(Exception):
    """ Exceptions for this modules """

    def __init__(self, msg: str):
        """ :param msg: The message to display """
        super().__init__()
        self.msg = msg

    def __str__(self):
        return self.msg


class FilterInputKeyError(Exception):
    """ Used to show an error with a filter's input keys
    e.g. multiple values have been assigned to the same input """


class FilterInputValidationError(Exception):
    """ Used to show an error when running the validation on the filter's inputs
    i.e. when running _validate_inputs() """


class FilterLoopError(Exception):
    """ Used when a loop is detected in the filter chain """

    def __init__(self):
        super().__init__("A loop has been detected in the filters")


FILTER = "filter"
IO_MAP = "io_map"


class BaseFilter(ABC):
    """
    An abstract base class for filters. All filters should inherit this class
    """

    def __init__(self, name: str = "Unknown"):
        self.name = name
        self.inputs = {}
        self.outputs = {}

        # A placeholder for inputs before they are bound
        self._i = {}

        # Parent filters (a list of dict {FILTER: Filter, IO_MAP: dict/None})
        self.parent_dict_list = []  # type: List[Dict]

        # Needs to be run (inputs have changed)
        self.needs_run = False

    def __str__(self):
        return self.name

    def add_input(self, key: str, value):
        """
        Adds an input with a given key and value.
        If the key is already in the inputs, an RuntimeError is raised
        """
        # Mark this filter as needing to be run
        self.needs_run = True

        if key in self._i:
            raise FilterInputKeyError(
                f"Key: {key} already existing in inputs for {self.name} filter."
            )
        self._i[key] = value

    def add_inputs(
        self,
        input_dict: Mapping[str, Any],
        io_map: Mapping[str, str] = None,
        io_map_optional: bool = False,
    ):
        """
        Adds multiple inputs via a dictionary. Optionally, maps the dictionary onto
        different input keys using an io_map.
        :param input_dict: The input dictionary
        :param io_map: The dictionary used to perform the mapping. All keys
        and values must be strings. For example:
        As an example:
        {
            "one": "two",
            "three": "four"
        }
        will map inputs keys of "one" to "two" AND "three" to "four". If io_map is None,
        no mapping with be performed.
        :param io_map_optional: If this is False, a KeyError will be raised
        if the keys in the io_map are not found in the input_dict.
        :raises KeyError: if keys required in the mapping are not found in the input_dict
        """
        mapped_inputs = {}
        if io_map is None:
            mapped_inputs = {**input_dict}
        else:
            mapped_inputs = map_dict(
                input_dict=input_dict, io_map=io_map, io_map_optional=io_map_optional
            )
        for key, value in mapped_inputs.items():
            self.add_input(key, value)

    def add_child_filter(self, child: "BaseFilter", io_map: Mapping[str, str] = None):
        """ See documentation for `add_parent_filter` """
        child.add_parent_filter(parent=self, io_map=io_map)

    def add_parent_filter(self, parent: "BaseFilter", io_map: Mapping[str, str] = None):
        """
        Add a parent filter (the inputs of this filter will be connected
        to the output of the parents).
        By default, the ALL outputs of the parent will be directly mapped to the
        inputs of this filter using the same KEY.
        This can be overridden by supplying io_map. e.g.
        io_map = {
            "output_key1":"input_key1",
            "output_key2":"input_key2",
            ...
            }
        will map the output of the parent filter with a key of "output_key1" to the
        input of this filter with a key of "input_key1" etc.
        If `io_map` is defined ONLY those keys which are explicitly listed are mapped
        (the others are ignored)
        """
        # Mark this filter as needing to be run
        self.needs_run = True

        # Search the parents to see if this parent already exists. If so, update it.
        new_parent_dict = {FILTER: parent, IO_MAP: io_map}
        for i, old_parent_dict in enumerate(self.parent_dict_list):
            if old_parent_dict[FILTER] == parent:
                self.parent_dict_list[i] = new_parent_dict
                return

        # Otherwise, add parent as a new parent
        self.parent_dict_list.append(new_parent_dict)

    def run(self, history=None):
        """
        Calls the _run class on all parents (recursively) to make sure they are up-to-date.
        Then maps the parents' outputs to inputs for this filter.
        Then calls the _run method on this filter.
        """
        # Don't run anything if the inputs haven't changed
        if not self.needs_run:
            return

        if history is None:
            history = []

        # Check we haven't already been to this filter (recursion == bad)
        if self in history:
            raise FilterLoopError()
        history.append(self)

        # Run all of the parent filters
        for parent_dict in self.parent_dict_list:
            parent_dict[FILTER].run(history=history)

        logger.info("Running %s", self)
        # Populate all of the inputs to this filter
        self.inputs = {}

        # Shallow copy the inputs added with `add_input`
        self.inputs = copy.copy(self._i)

        # Map all inputs from parent filters
        for parent_dict in self.parent_dict_list:

            for output_key in parent_dict[FILTER].outputs:
                if parent_dict[IO_MAP] is None:
                    # Directly map the parent outputs to inputs
                    input_key = output_key
                else:
                    # Use the io_map lookup to check the output_key exists
                    if output_key not in parent_dict[IO_MAP]:
                        input_key = None
                    else:
                        input_key = parent_dict[IO_MAP][output_key]

                if input_key is not None:
                    # We have a mapping from output to input
                    # Check the input_key does not already exist
                    if input_key in self.inputs:
                        raise FilterInputKeyError(
                            f"A mapping is defined: "
                            f"from filter \"{parent_dict['filter']}\" "
                            f'with output key "{output_key}" '
                            f'to filter "{self}" '
                            f'with input key "{input_key}". '
                            "However, this input has already been defined."
                        )
                    self.inputs[input_key] = parent_dict[FILTER].outputs[output_key]

        # Validate the inputs to this filter
        self._validate_inputs()

        # Run this filter
        self._run()

        # Mark this filter as not needing to be run
        self.needs_run = False

    @abstractmethod
    def _run(self):
        """
        Takes all of the inputs and creates the outputs.
        THIS SHOULD BE OVERWRITTEN IN THE SUBCLASS
        """

    @abstractmethod
    def _validate_inputs(self):
        """
        Validates all of the inputs. Should raise a FilterInputValidationError
        if the validation is not passed.
        THIS SHOULD BE OVERWRITTEN IN THE SUBCLASS
        """
