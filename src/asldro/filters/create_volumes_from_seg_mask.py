"""Create volumes from segmentation mask filter"""

import jsonschema
import numpy as np

from asldro.containers.image import BaseImageContainer
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.validators.parameters import (
    Parameter,
    ParameterValidator,
    isinstance_validator,
    for_each_validator,
)


class CreateVolumesFromSegMask(BaseFilter):
    """A filter for assigning values to regions defined
    by a segmentation mask, then concatenating these images
    into a 5D image

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`CreateVolumesFromSegMask.add_input()`
    member function. They are also accessible via class constants,
    for example :class:`GroundTruthLoaderFilter.KEY_IMAGE`

    :param 'seg_mask': segmentation mask image comprising integer values for each region. dtype of
      the image data must be an unsigned or signed integer type.
    :type 'seg_mask': BaseImageContainer
    :param 'label_values': List of the integer values in ``'seg_mask'``, must not have any duplicate
      values. The order of the integers in this list must be matched by the input ``'label_names'``,
      and the lists with quantity values for each quantity in the input ``'quantitites'``.
    :type 'label_values': list[int]
    :param 'label_names': List of strings defining the names of each region defined in
      ``'seg_mask'``. The order matches the order given in ``'label_values'``.
    :type 'label_names': list[str]
    :param 'quantities': Dictionary, containing key/value pairs where the key name defines a
      a quantity, and the value is an array of floats that define the value to assign to each 
      region. The order of these floats matches the order given in ``'label_values'``.
    :param 'units': List of strings defining the units that correspond with each quantity given
      in the dictionary ``'quantities'``, as given by the order defined in that dictionary.
    :type 'units': list[str]

    **Outputs**

    Once run, the filter will populate the dictionary :class:`CreateVolumesFromSegMask.outputs` with the
    following entries

    :param 'image': The combined 5D image, with volumes where the values for each quantity have been
      assigned to the regions defined in ``'seg_mask'``. The final entry in the 5th dimension is
      a copy of the image ``'seg_mask'``.
    :type 'image': BaseImageContainer
    :param 'image_info': A dictionary describing the regions, quantities and units in the outpu
      ``'image'``. This is of the same format as the ground truth JSON file, however there is no
      'parameters' object.  See :ref:`custom-ground-truth`
      for more information on this format.
    :type 'image_info': dict



    """

    KEY_IMAGE = "image"
    KEY_SEG_MASK = "seg_mask"
    KEY_LABEL_NAMES = "label_names"
    KEY_LABEL_VALUES = "label_values"
    KEY_QUANTITIES = "quantities"
    KEY_UNITS = "units"
    KEY_IMAGE_INFO = "image_info"

    def __init__(self):
        super().__init__("CreateVolumesFromSegMask")

    def _run(self):
        """runs the filter"""
        seg_mask: BaseImageContainer = self.inputs[self.KEY_SEG_MASK].clone()
        label_values: list[int] = self.inputs[self.KEY_LABEL_VALUES]
        label_names: list[str] = self.inputs[self.KEY_LABEL_NAMES]
        quantities: dict = self.inputs[self.KEY_QUANTITIES]
        units: list[str] = self.inputs[self.KEY_UNITS]

        num_quantities = len(units)
        num_labels = len(label_values)

        # seg_mask should be expanded to 4D (if it is not already)
        if seg_mask.image.ndim < 4:
            seg_mask.image = np.expand_dims(np.atleast_3d(seg_mask.image), axis=3)

        image_shape = seg_mask.shape
        new_image_shape = [1] * 5  # pre-allocate with 5 entries
        for i, x in enumerate(image_shape):
            new_image_shape[i] = x
        new_image_shape[4] = num_quantities + 1

        image_data_5d = np.zeros(new_image_shape)
        image_info = {
            "quantities": list(quantities.keys()) + ["seg_label"],
            "units": units + [""],
            "segmentation": {
                label_names[i]: label_values[i] for i in range(num_labels)
            },
        }

        for i, quantity in enumerate(quantities.keys()):
            temp_image = np.zeros(image_shape)
            # image_info["quantities"][i] = quantity
            for j, region_value in enumerate(label_values):
                temp_image[seg_mask.image == region_value] = quantities[quantity][j]
            image_data_5d[:, :, :, :, i] = temp_image

        # put the seg_mask volume last
        image_data_5d[:, :, :, :, -1] = seg_mask.image.astype(image_data_5d.dtype)

        image = seg_mask.clone()
        image.image = image_data_5d

        self.outputs[self.KEY_IMAGE_INFO] = image_info
        self.outputs[self.KEY_IMAGE] = image

    def _validate_inputs(self):
        """Checks that the inputs meet their validation criteria
        'seg_mask' BaseImageContainer, dtype dtype==np.uint16, ndim <= 4
        'label_names', list[str], must be same length as 'label_values'
        'label_values', list[int],all values must be unique, must match the number of
        unique values in 'seg_mask'
        'quantities', dict, each of the keys must be a list the length of 
        'label_values' and the values a float or int type.
        'units', list[str], must be the same length as the number of keys in 'quantities'
        """

        input_validator = ParameterValidator(
            parameters={
                self.KEY_SEG_MASK: Parameter(
                    validators=isinstance_validator(BaseImageContainer)
                ),
                self.KEY_LABEL_VALUES: Parameter(
                    validators=[for_each_validator(isinstance_validator(int)),]
                ),
                self.KEY_LABEL_NAMES: Parameter(
                    validators=[for_each_validator(isinstance_validator(str)),]
                ),
                self.KEY_QUANTITIES: Parameter(validators=isinstance_validator(dict)),
                self.KEY_UNITS: Parameter(
                    validators=[for_each_validator(isinstance_validator(str)),]
                ),
            },
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)

        # label_values should only contain unique values
        label_values = self.inputs[self.KEY_LABEL_VALUES]
        if len(label_values) != len(set(label_values)):
            raise FilterInputValidationError("'label_values' has non-unique entries")

        # label_names should be the same length as label_values
        if len(self.inputs[self.KEY_LABEL_NAMES]) != len(label_values):
            raise FilterInputValidationError(
                f"'label_names' must be the same length as 'label_values'"
            )

        seg_mask: BaseImageContainer = self.inputs[self.KEY_SEG_MASK]
        # The data type of the image data in 'seg_mask' should be a signed or unsigned integer
        if not issubclass(seg_mask.image.dtype.type, np.integer):
            raise FilterInputValidationError(
                f"{self} filter requires the image data in the input 'seg_mask'"
                "to be either signed or unsigned integer type"
                f"type is {seg_mask.image.dtype}"
            )

        # the number of data dimensions should be 4 or less
        if np.ndim(seg_mask.image) > 4:
            raise FilterInputValidationError(
                f"{self} filter requires the image data in the input 'seg_mask'"
                "to have 4 dimensions or less"
                f"number of dimensions is {np.ndim(seg_mask.image)}"
            )

        # the number of unique values in the image data of 'seg_mask' must match the
        # number of values in 'label_values'
        if len(np.unique(seg_mask.image)) != len(label_values):
            raise FilterInputValidationError(
                f"{self} filter requires the number of unique values in the image data"
                "of the input 'seg_mask' to be the same as the number of values in"
                "'label_values"
            )
        # The values in 'label_values' must match the unique values in 'seg_mask'
        if (
            np.unique(seg_mask.image) != sorted(self.inputs[self.KEY_LABEL_VALUES])
        ).all():
            raise FilterInputValidationError(
                f"{self} filter requires the values in the input 'label_values' to match"
                "the unique values in the input 'seg_mask'"
            )

        # for each key in 'quantities', the value must be a list of length 'label_values'
        # and must be a float
        quantities: dict = self.inputs[self.KEY_QUANTITIES]
        for key in quantities.keys():
            # check is a list
            if not isinstance(quantities[key], list):
                raise FilterInputValidationError(
                    f"value for {key} in 'quantities' must be a list"
                )
            # check all entries are floats
            if not all(isinstance(x, float) for x in quantities[key]):
                raise FilterInputValidationError(
                    f"all values in list for {key} in 'quantities' must be of type float"
                )
            # check length of list is the same as 'label_values'
            if not len(quantities[key]) == len(self.inputs[self.KEY_LABEL_VALUES]):
                raise FilterInputValidationError(
                    f"number of entries for {key} in 'quantities' must be"
                    f"{len(self.inputs[self.KEY_LABEL_VALUES])}"
                )

        # units should be the same length as the keys in 'quantities'
        if len(self.inputs[self.KEY_UNITS]) != len(quantities.keys()):
            raise FilterInputValidationError(
                f"'units' must be the same length as number of keys in 'quantities'"
            )

