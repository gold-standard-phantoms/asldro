"""Combined Fuzzy Masks Filter"""

import numpy as np


from asldro.containers.image import BaseImageContainer
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.validators.parameters import (
    Parameter,
    ParameterValidator,
    greater_than_validator,
    range_inclusive_validator,
    for_each_validator,
    isinstance_validator,
)


class CombineFuzzyMasksFilter(BaseFilter):
    """
    A filter for creating a segmentation mask based on one or more 'fuzzy' masks.

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`CombineFuzzyMasksFilter.add_input()`
    member function. They are also accessible via class constants,
    for example :class:`CombineFuzzyMasksFilter.KEY_THRESHOLD`

    :param 'fuzzy_mask': Fuzzy mask images to combine. Each mask should have voxel values between
      0 and 1, defining the fraction of that region in each voxel.
    :type 'fuzzy_mask': BaseImageContainer or list[BaseImageContainer]
    :param 'region_values': A list of values to assign to each region in the output 'seg_mask'. The
      order corresponds with the order of the masks in 'fuzzy_mask'.
    :type 'region_values': int or list[int]
    :param 'region_priority': A list of priority order for the regions, 1 being the highest
      priority. The order corresponds with the order of the masks in 'fuzzy_mask', and all values
      must be unique. If 'fuzzy_mask' is a single image then this input can be omitted.
    :type 'region_priority': list[int] or int
    :param 'threshold': The threshold value, below which a region's contributions to a voxel are 
      ignored. Must be between 0 and 1.0. Defaults to 0.05.
    :type 'threshold': float, optional


    **Outputs**

    Once run, the filter will populate the dictionary :class:`CombineFuzzyMasksFilter.outputs`
    with the following entries

    :param 'seg_label': A segmentation mask image constructed from the inputs, defining exclusive 
      regions (one region per voxel). The image data type is numpy.int16.
    :type 'seg_label': BaseImageContainer

    """

    KEY_FUZZY_MASK = "fuzzy_mask"
    KEY_REGION_VALUES = "region_values"
    KEY_REGION_PRIORITY = "region_priority"
    KEY_THRESHOLD = "threshold"
    KEY_SEG_MASK = "seg_mask"

    def __init__(self):
        super().__init__("CombineFuzzyMasksFilter")

    def _run(self):
        """runs the filter"""

        # determine number of fuzzy_masks

        if isinstance(self.inputs[self.KEY_FUZZY_MASK], BaseImageContainer):
            fuzzy_masks = [
                self.inputs[self.KEY_FUZZY_MASK],
            ]
            number_masks = 1
            region_values = [
                self.inputs[self.KEY_REGION_VALUES],
            ]
            region_priorities = [
                self.inputs[self.KEY_REGION_PRIORITY],
            ]
        else:
            fuzzy_masks = self.inputs[self.KEY_FUZZY_MASK]
            number_masks = len(fuzzy_masks)
            region_values = self.inputs[self.KEY_REGION_VALUES]
            region_priorities = self.inputs[self.KEY_REGION_PRIORITY]

        seg_mask: BaseImageContainer = fuzzy_masks[0].clone()
        seg_mask.image = np.zeros_like(seg_mask.image, dtype=np.int16)

        masks_range = list(range(number_masks))

        for i in masks_range:
            # create a binary image with voxels where the fraction in the ith image is
            # higher than the jth image, or if it is a tie with another mask for
            # first place, then if the region's priority is higher
            region_mask = np.ones_like(seg_mask.image, dtype=bool)
            for j in masks_range[:i] + masks_range[i + 1 :]:
                # flip to false any voxels where the region's fraction is lower than the jth
                # mask
                #
                region_mask[fuzzy_masks[i].image < fuzzy_masks[j].image] = False

                # if the region priority value is higher (1 = highest priority) than for the jth
                # image then set any voxels where the fractions are equal to false
                if region_priorities[i] > region_priorities[j]:
                    region_mask[
                        (fuzzy_masks[i].image == fuzzy_masks[j].image)
                        & (fuzzy_masks[i].image > 0)
                    ] = False

            # use the threshold as a final test: only use voxels that are above the fraction as well
            # as all previous conditions
            region_mask = region_mask & (
                fuzzy_masks[i].image > self.inputs[self.KEY_THRESHOLD]
            )

            seg_mask.image[region_mask] = region_values[i]

        self.outputs[self.KEY_SEG_MASK] = seg_mask

    def _validate_inputs(self):
        """Checks that the inputs meet their validation criteria
        'fuzzy_mask' must be a BaseImageContainer, or list of BaseImageContainers. For each
          image the voxel values should be between 0 and 1 inclusive.
          All image shapes must match
          All image affine matrices must be the same.
        'region_values' must be a list of integers. The length must be the same as the length
          of 'fuzzy_mask'.
        'region_priority' must be a list of integers, values greater than 0. The length must
           be the same as the length of 'fuzzy_mask'. All values must be unique. Optional for
           the case where fuzzy_mask is a single image.
        threshold.
        'threshold' must be a float, between 0 and 1 inclusive. It is optional with default 0.05.
        """

        input_validator = ParameterValidator(
            parameters={
                self.KEY_FUZZY_MASK: Parameter(
                    validators=[isinstance_validator((list, BaseImageContainer))]
                ),
                self.KEY_REGION_VALUES: Parameter(
                    validators=[isinstance_validator((list, int))]
                ),
                self.KEY_REGION_PRIORITY: Parameter(
                    validators=[isinstance_validator((list, int))], optional=True
                ),
                self.KEY_THRESHOLD: Parameter(
                    validators=[
                        isinstance_validator(float),
                        range_inclusive_validator(0.0, 1.0),
                    ],
                    optional=True,
                    default_value=0.05,
                ),
            }
        )
        new_params = input_validator.validate(
            self.inputs, error_type=FilterInputValidationError
        )

        # check all the fuzzy mask images have the same shape and affine
        number_masks = 1  # if there's just one image don't need to check

        if isinstance(self.inputs[self.KEY_FUZZY_MASK], list):

            # first check against a validator
            fuzzy_mask: list[BaseImageContainer] = self.inputs[self.KEY_FUZZY_MASK]
            fuzzy_mask_list_validator = ParameterValidator(
                parameters={
                    self.KEY_FUZZY_MASK: Parameter(
                        validators=[
                            for_each_validator(
                                isinstance_validator(BaseImageContainer)
                            ),
                            for_each_validator(range_inclusive_validator(0.0, 1.0)),
                        ]
                    ),
                    self.KEY_REGION_VALUES: Parameter(
                        validators=[for_each_validator(isinstance_validator(int)),]
                    ),
                    self.KEY_REGION_PRIORITY: Parameter(
                        validators=[
                            for_each_validator(isinstance_validator(int)),
                            for_each_validator(greater_than_validator(0)),
                        ]
                    ),
                }
            )
            fuzzy_mask_list_validator.validate(
                self.inputs, error_type=FilterInputValidationError
            )
            number_masks = len(fuzzy_mask)
            image_shapes = [fuzzy_mask[i].shape for i in range(number_masks)]
            if image_shapes.count(image_shapes[0]) != number_masks:
                raise FilterInputValidationError(
                    [
                        "shapes of the images in input 'fuzzy_masks' do not match",
                        [
                            f"image #{i}: {image_shapes[i]}, "
                            for i in range(number_masks)
                        ],
                    ]
                )
            image_affines = [fuzzy_mask[i].affine for i in range(number_masks)]
            if not (image_affines == image_affines[0]).all():
                raise FilterInputValidationError(
                    [
                        "affines of the images in input 'fuzzy_masks' do not match",
                        [
                            f"image #{i}: {image_affines[i]}, "
                            for i in range(number_masks)
                        ],
                    ]
                )
            # 'region_values' must have length == number_masks
            if len(self.inputs[self.KEY_REGION_VALUES]) != number_masks:
                raise FilterInputValidationError(
                    f"'region_values' must be the same length as 'fuzzy_masks'"
                )

            if len(self.inputs[self.KEY_REGION_PRIORITY]) != len(
                np.unique(self.inputs[self.KEY_REGION_PRIORITY])
            ):
                raise FilterInputValidationError(
                    f"'region_priority' must not have any repeated values"
                )

            # 'region_priority' must have length == number_masks
            if len(self.inputs[self.KEY_REGION_PRIORITY]) != number_masks:
                raise FilterInputValidationError(
                    f"'region_priority' must be the same length as 'fuzzy_masks'"
                )
        else:
            fuzzy_mask: BaseImageContainer = self.inputs[self.KEY_FUZZY_MASK]
            fuzzy_mask_image_validator = ParameterValidator(
                parameters={
                    self.KEY_FUZZY_MASK: Parameter(
                        validators=[range_inclusive_validator(0.0, 1.0),]
                    ),
                    self.KEY_REGION_PRIORITY: Parameter(
                        validators=[greater_than_validator(0)],
                        optional=True,
                        default_value=1,
                    ),
                }
            )
            fuzzy_mask_image_params = fuzzy_mask_image_validator.validate(
                self.inputs, error_type=FilterInputValidationError
            )
            # merge these with new_params
            new_params = {**new_params, **fuzzy_mask_image_params}

        # merge the updated parameters from the output with the input parameters
        self.inputs = {**self._i, **new_params}
