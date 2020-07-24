import copy
from typing import Tuple, List
from abc import ABC, abstractmethod


class FilterInputKeyError(Exception):
    def __init__(self, msg: str):
        super().__init__(str)


class FilterInputValidationError(Exception):
    def __init__(self, msg: str):
        super().__init__(str)


class FilterLoopError(Exception):
    def __init__(self):
        super().__init__("A loop has been detected in the filters")


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

        # Parent filters (a list of dict {"filter": Filter, "io_map": dict/None})
        self.parents = []  # type: List[Dict]

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

    def add_child_filter(self, child: "BaseFilter", io_map: dict = None):
        """ See documentation for `add_parent_filter` """
        child.add_parent_filter(parent=self, io_map=io_map)

    def add_parent_filter(self, parent: "BaseFilter", io_map: dict = None):
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
        will map the input of this filter with a key of "input_key1" to the
        output of the parent filter with a key of "output_key1", etc.
        If `io_map` is defined ONLY those keys which are explicitly listed are mapped
        (the others are ignored)
        """
        # Mark this filter as needing to be run
        self.needs_run = True

        # Search the parents to see if this parent already exists. If so, update it.
        parent_dict = {"filter": parent, "io_map": io_map}
        for i, p in enumerate(self.parents):
            if p["filter"] == parent:
                self.parents[i] = parent_dict
                return

        # Otherwise, add parent as a new parent
        self.parents.append(parent_dict)

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
        for p in self.parents:
            p["filter"].run(history=history)

        # Populate all of the inputs to this filter
        self.inputs = {}

        # Shallow copy the inputs added with `add_input`
        self.inputs = copy.copy(self._i)

        # Map all inputs from parent filters
        for p in self.parents:

            for output_key in p["filter"].outputs:
                if p["io_map"] is None:
                    # Directly map the parent outputs to inputs
                    input_key = output_key
                else:
                    # Use the io_map lookup to check the output_key exists
                    if output_key not in p["io_map"]:
                        input_key = None

                    input_key = p["io_map"][output_key]

                if input_key is not None:
                    # We have a mapping from output to input
                    # Check the input_key does not already exist
                    if input_key in self.inputs:
                        raise FilterInputKeyError(
                            f"A mapping is defined: "
                            f"from filter \"{p['filter']}\" "
                            f'with output key "{output_key}" '
                            f'to filter "{self}" '
                            f'with input key "{input_key}". '
                            "However, this input has already been defined."
                        )
                    self.inputs[input_key] = p["filter"].outputs[output_key]

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
        pass

    @abstractmethod
    def _validate_inputs(self):
        """
        Validates all of the inputs. Should raise a FilterInputValidationError
        if the validation is not passed.
        THIS SHOULD BE OVERWRITTEN IN THE SUBCLASS
        """
        pass
