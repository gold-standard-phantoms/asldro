""" Combine Time Series Filter """

import re
from typing import List, Tuple

import numpy as np
import nibabel as nib
from asldro.containers.image import (
    BaseImageContainer,
    COMPLEX_IMAGE_TYPE,
    NiftiImageContainer,
)
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError


class CombineTimeSeriesFilter(BaseFilter):
    """A filter that takes, as input, as set of ImageContainers. These should each
    represent a single time point in a time series acquisition. As an output, these
    ImageContainers will be concatenated across the 4th (time) dimension and their
    metadata combined with the following rules:
    - if all values of a given field are the same, for all time-series, use that
    value in the output metadata; else,
    - concatenate the values in a list.
    Instance variables of the BaseImageContainer such as ``image_flavour``, will all
    be checked for consistency and copied across to the output image.

    **Inputs**

    Input Parameters are all keyword arguments for the
    :class:`CombineTimeSeriesFilter.add_inputs()` member function.
    They are also accessible via class constants,
    for example :class:`CombineTimeSeriesFilter.KEY_T1`.

    :param 'image_NNNNN': A time-series image. The order of these time series will be
    determined by the NNNNN component, which shall be a positive integer. Any number of
    digits can be used in combination in NNNNN. For example, as sequence, `image_0000`,
    `image_1`, `image_002`, `image_03` is valid.
    NOTE: the indices MUST start from 0 and increment by 1, and have no missing or duplicate
    indices. This is to help prevent accidentally missing/adding an index value.
    :type 'image_NNNNN': BaseImageContainer

    **Outputs**

    Once run, the filter will populate the dictionary :class:`MriSignalFilter.outputs` with the
    following entries

    :param 'image': A 4D image of the combined time series.
    :type 'image': BaseImageContainer
    """

    # Key constants
    KEY_IMAGE = "image"
    INPUT_IMAGE_REGEX_OBJ = re.compile(r"^image_(?P<index>[0-9]+)$")

    def __init__(self):
        super().__init__(name="Combine Time-Series")

    def _run(self):
        indexed_containers = self._get_input_images()
        containers = [c[0] for c in indexed_containers]

        dataobj = np.stack([container.image for container in containers], axis=3)

        output_container = NiftiImageContainer(
            nifti_img=nib.Nifti1Image(dataobj=dataobj, affine=containers[0].affine)
        )
        output_container.data_domain = containers[0].data_domain
        # TODO: We could set output_container.time_step_seconds here

        # Create the metadata
        output_container.metadata = {}
        # Find all of the keys in all of the metadata
        all_keys = {
            key for container in containers for key in container.metadata.keys()
        }

        for key in all_keys:
            present = [key in container.metadata for container in containers]
            if sum(present) == 1:
                # The key appears in one container only, so copy it to the output
                output_container.metadata[key] = containers[
                    present.index(True)
                ].metadata[key]
            else:
                # The key appears in more than one containers
                all_values = [
                    container.metadata[key] if key in container.metadata else None
                    for container in containers
                ]
                # find values that are not None
                values_not_none = [val for val in all_values if val is not None]
                if all_values.count(all_values[0]) == len(all_values):
                    # All values are the same - output that value for the given key
                    output_container.metadata[key] = containers[0].metadata[key]
                elif all_values.count(values_not_none[0]) == len(values_not_none):
                    # All values that are not None are the same, output just that value for
                    # given key
                    output_container.metadata[key] = values_not_none[0]
                else:
                    # The values are not the same - concatenate them into a list
                    output_container.metadata[key] = all_values

        self.outputs[self.KEY_IMAGE] = output_container

    def _get_input_images(self) -> List[Tuple[BaseImageContainer, int]]:
        """Based on the naming rule for input images:
        `image_NNNNN`, where N is any positive integer, extract all of the
        input images, in ascending index order, to a list
        :return: A list of tuples where the first element is a list
        of :class:`BaseImageContainer`. The second element is the
        extracted integer image index.
        """

        containers = []
        indices = []

        # Extract the input images
        for key, value in self.inputs.items():
            match_obj = self.INPUT_IMAGE_REGEX_OBJ.match(key)
            if match_obj is not None:
                containers.append(value)
                indices.append(int(match_obj.group("index")))

        return sorted(zip(containers, indices), key=lambda x: x[1])

    def _validate_inputs(self):
        """Checks that the inputs meet their validation critera:
        There must be one or more input image.
        Once parsed, it must be the case that there are no duplicate indices
        for the input images (for example: `image_001` and `image_01`)
        All input image indices must start from 0 and increment by one each time.
        All input images must be derived from BaseImageContainer, and non-complex.
        All input images must have the same dimensions.
        All input images must have three dimensions.
        """

        indexed_containers = self._get_input_images()
        containers = [c[0] for c in indexed_containers]
        indices = [c[1] for c in indexed_containers]

        # The indices must increase 0,1,2,3,4, etc
        if indices != list(range(len(indices))):
            raise FilterInputValidationError(
                "Input image indices are not in "
                f"a valid input order. Must increase 0,1,2,etc. Indices are: {indices}"
            )

        # There must be one or more input image
        if len(indexed_containers) == 0:
            raise FilterInputValidationError("No input images found")

        # Check there are no duplicate indices in the input images
        if list(set(indices)) != indices:
            raise FilterInputValidationError(
                f"There are duplicate indices in the input images: {indices}"
            )

        # Check all of the input_nnnnn are actually image containers
        for container, index in indexed_containers:
            if not isinstance(container, BaseImageContainer):
                raise FilterInputValidationError(
                    f"Input with index {index} is not an image container, "
                    f"is {type(container)}"
                )

        # Check that all the input images are all the same dimensions
        # Skip checking the first element as we use that to compare
        for container, index in list(indexed_containers)[1:]:
            if container.image.shape != containers[0].image.shape:
                raise FilterInputValidationError(
                    f"Input image with index {index} has shape {container.image.shape} "
                    f"which does not match the other images with shape "
                    f"{containers[0].image.shape}"
                )
        # Check that all input images are 3d
        if len(container.image.shape) != 3:
            raise FilterInputValidationError("Input images must be 3D")

        # Check that all the input images are not of image_type == "COMPLEX_IMAGE_TYPE"
        for container, index in indexed_containers:
            if container.image_type == COMPLEX_IMAGE_TYPE:
                raise FilterInputValidationError(
                    f"Input image with index {index} has image type {COMPLEX_IMAGE_TYPE}, "
                    "this is not supported"
                )
