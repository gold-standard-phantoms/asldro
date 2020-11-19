""" AppendMetadataFilter """

from asldro.containers.image import BaseImageContainer
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.validators.parameters import (
    Parameter,
    ParameterValidator,
    isinstance_validator,
)


class AppendMetadataFilter(BaseFilter):
    """A filter that can add key-value pairs to the metadata dictionary property of an
    image container.  If the supplied key already exists the old value will be overwritten
    with the new value.  The input image container is modified and a reference passed
    to the output, i.e. no copy is made.

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`AppendMetadataFilter.add_inputs()`
    member function. They are also accessible via class constants,
    for example :class:`AppendMetadataFilter.KEY_METADATA`

    :param 'image': The input image to append the metadata to
    :type 'image': BaseImageContainer
    :param 'metadata': dictionary of key-value pars to append to the metadata property of the
        input image.
    :type 'metadata': dict

    **Outputs**

    Once run, the filter will populate the dictionary :class:`AppendMetadataFilter.outputs` with the
    following entries

    :param 'image': The input image, with the input metadata merged.
    :type 'image: BaseImageContainer

    """

    # Key constants
    KEY_IMAGE = "image"
    KEY_METADATA = "metadata"

    def __init__(self):
        super().__init__(name="Append Meta Data")

    def _run(self):
        """ appends the input image with the supplied metadata"""
        # copy the reference to the input image to outputs
        self.outputs[self.KEY_IMAGE] = self.inputs[self.KEY_IMAGE]
        # merge the input metadata with the existing metadata
        self.outputs[self.KEY_IMAGE].metadata = {
            **self.outputs[self.KEY_IMAGE].metadata,
            **self.inputs[self.KEY_METADATA],
        }

    def _validate_inputs(self):
        """Checks that the inputs meet their validation criteria
        'image' must be derived from BaseImageContainer
        'metadata' must be a dict
        """

        input_validator = ParameterValidator(
            parameters={
                self.KEY_IMAGE: Parameter(
                    validators=isinstance_validator(BaseImageContainer)
                ),
                self.KEY_METADATA: Parameter(validators=isinstance_validator(dict)),
            }
        )

        input_validator.validate(self.inputs, error_type=FilterInputValidationError)
