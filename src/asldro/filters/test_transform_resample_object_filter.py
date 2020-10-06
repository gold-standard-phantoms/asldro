""" Transform Resample Object Filter Tests """

from copy import deepcopy
import pytest
import numpy as np
import numpy.testing
import nibabel as nib
import matplotlib.pylab as plt
from nilearn.plotting import show

from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.containers.image import NiftiImageContainer, NumpyImageContainer
from asldro.filters.transform_resample_object_filter import (
    TransformResampleObjectFilter,
)

TEST_VOLUME_DIMENSIONS = (32, 32, 32)
TEST_NIFTI_ONES = NiftiImageContainer(
    nifti_img=nib.Nifti2Image(np.zeros(TEST_VOLUME_DIMENSIONS), affine=np.eye(4))
)

INPUT_VALIDATION_DICTIONARY = {
    "object": (TEST_NIFTI_ONES, np.ones(TEST_VOLUME_DIMENSIONS), "str", 1.0),
    "acquisition_shape": (
        TEST_VOLUME_DIMENSIONS,
        (16.0, 16.0, 16.0),
        (1, 2, 3, 4),
        "str",
        1,
        [32, 32, 32],
    ),
    "rotation": (
        (0.0, 0.0, 0.0),
        1.0,
        (181.0, -181.0, 234.2),
        (0.0, 0.0, 0.0, 0.0),
        "str",
    ),
    "rotation_origin": ((1.0, 2.0, 3.0), 1.0, (int(1), int(2), int(3)), "str"),
    "translation": (
        (1.0, 2.0, 3.0),
        1.0,
        (int(1), int(2), int(3)),
        (0.0, 0.0, 0.0, 0.0),
        "str",
    ),
}


def add_multiple_inputs_to_filter(input_filter: BaseFilter, input_data: dict):
    """ Adds the data held within the input_data dictionary to the filter's inputs """
    for key in input_data:
        input_filter.add_input(key, input_data[key])

    return input_filter


@pytest.mark.parametrize("validation_data", [INPUT_VALIDATION_DICTIONARY])
def test_transform_resample_object_filter_validate_inputs(validation_data: dict):
    """ Check a FilterInputValidationError is raised when the
    inputs to the TransformResampleObjectFilter are incorrect or missing
    """
    # Check with all data that should pass
    xr_obj_filter = TransformResampleObjectFilter()
    test_data = deepcopy(validation_data)

    for data_key in test_data:
        xr_obj_filter.add_input(data_key, test_data[data_key][0])

    xr_obj_filter.run()

    for inputs_key in validation_data:
        xr_obj_filter = TransformResampleObjectFilter()
        test_data = deepcopy(validation_data)

        # remove key
        test_data.pop(inputs_key)
        for data_key in test_data:
            xr_obj_filter.add_input(data_key, test_data[data_key][0])

        # "object" is not optional so should raise an error
        # all the others are so should run with no issues
        if inputs_key == "object":
            with pytest.raises(FilterInputValidationError):
                xr_obj_filter.run()
        else:
            xr_obj_filter.run()

        # Try data that should fail
        for test_value in validation_data[inputs_key][1:]:
            xr_obj_filter = TransformResampleObjectFilter()
            for data_key in test_data:
                xr_obj_filter.add_input(data_key, test_data[data_key][0])
            xr_obj_filter.add_input(inputs_key, test_value)
            with pytest.raises(FilterInputValidationError):
                xr_obj_filter.run()


def test_transform_resample_object_filter_mock_data():
    """ Test the transform_resampe_object_filter with some mock data """

    # Create some synthetic data
    grid = np.mgrid[0:128, 0:128]
    circle = (
        np.sum((grid - np.array([32, 32])[:, np.newaxis, np.newaxis]) ** 2, axis=0)
        < 256
    )
    diamond = (
        np.sum(np.abs(grid - np.array([75, 32])[:, np.newaxis, np.newaxis]), axis=0)
        < 16
    )
    rectangle = (
        np.max(np.abs(grid - np.array([64, 64])[:, np.newaxis, np.newaxis]), axis=0)
        < 16
    )
    image = np.zeros_like(circle)
    image = image + circle + 2.0 * rectangle + 3.0 * diamond + np.eye(128)
    image = numpy.expand_dims(image, axis=2)

    # define world coordinate origin (x,y,z) = (0,0,0) at (i,j,k) = (64,64,1)
    # and 1 voxel == 1mm isotropically
    # therefore according to RAS+:
    source_affine = np.array(
        ((1, 0, 0, -64), (0, 1, 0, -64), (0, 0, 1, 0), (0, 0, 0, 1),)
    )

    nifti_image = nib.Nifti2Image(image, affine=source_affine)

    # create NiftiImageContainer of this image
    nifti_image_container = NiftiImageContainer(nifti_image)

    xr_obj_filter = TransformResampleObjectFilter()
    xr_obj_filter.add_input(
        TransformResampleObjectFilter.KEY_OBJECT, nifti_image_container
    )
    xr_obj_filter.add_input(
        TransformResampleObjectFilter.KEY_ROTATION, (0.0, 0.0, 10.0)
    )
    xr_obj_filter.add_input(
        TransformResampleObjectFilter.KEY_TRANSLATION, (0.0, 0.0, 0.0)
    )
    xr_obj_filter.add_input(
        TransformResampleObjectFilter.KEY_ACQUISITION_SHAPE, (64, 64, 1)
    )

    xr_obj_filter.run()
    new_nifti_container: NiftiImageContainer = xr_obj_filter.outputs[
        TransformResampleObjectFilter.KEY_IMAGE
    ]

    # visually check

    # plt.figure()
    # plt.imshow(nifti_image_container.image)
    # plt.title("original image")
    # plt.figure()
    # plt.imshow(new_nifti_container.image)
    # plt.title("transformed and resampled")
    # show()

