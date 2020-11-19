""" Resample Filter Tests """
# pylint: disable=duplicate-code

from copy import deepcopy
import pytest
import numpy as np
import numpy.testing
import nibabel as nib
import nilearn as nil
from asldro.filters.basefilter import FilterInputValidationError
from asldro.containers.image import NiftiImageContainer, NumpyImageContainer
from asldro.filters.resample_filter import ResampleFilter
from asldro.filters.affine_matrix_filter import AffineMatrixFilter

TEST_VOLUME_DIMENSIONS = (32, 32, 32)
TEST_NIFTI_ONES = NiftiImageContainer(
    nifti_img=nib.Nifti2Image(np.zeros(TEST_VOLUME_DIMENSIONS), affine=np.eye(4))
)

# input validation dictionary, [0] in each tuple passes, after that should fail validation
INPUT_VALIDATION_DICTIONARY = {
    "image": (TEST_NIFTI_ONES, np.ones(TEST_VOLUME_DIMENSIONS), "str", 1.0),
    "affine": (np.eye(4), np.eye(3), np.eye(1, 4), 1.0, "str"),
    "shape": (
        TEST_VOLUME_DIMENSIONS,
        (1, 3, 4, 5),
        (32.0, 32.0, 32.0),
        "str",
        1,
        [1, 2, 3],
    ),
}


@pytest.mark.parametrize("validation_data", [INPUT_VALIDATION_DICTIONARY])
def test_resample_filter_validate_inputs(validation_data: dict):
    """Check a FilterInputValidationError is raised when the
    inputs to the ResampleFilter are incorrect or missing
    """
    # Check with all data that should pass
    resample_filter = ResampleFilter()
    test_data = deepcopy(validation_data)
    for data_key in test_data:
        resample_filter.add_input(data_key, test_data[data_key][0])
    # should run with no issues
    resample_filter.run()

    for inputs_key in validation_data:
        resample_filter = ResampleFilter()
        test_data = deepcopy(validation_data)
        # remove the corresponding key from test_data
        test_data.pop(inputs_key)

        for data_key in test_data:
            resample_filter.add_input(data_key, test_data[data_key][0])

        # Key not defined

        with pytest.raises(FilterInputValidationError):
            resample_filter.run()

        # Key has wrong data type
        resample_filter.add_input(inputs_key, None)
        with pytest.raises(FilterInputValidationError):
            resample_filter.run()

        # Data not in the valid range
        for test_value in validation_data[inputs_key][1:]:
            # re-initialise filter
            resample_filter = ResampleFilter()

            # add valid inputs
            for data_key in test_data:
                resample_filter.add_input(data_key, test_data[data_key][0])

            # add invalid input and check a FilterInputValidationError is raised
            resample_filter.add_input(inputs_key, test_value)
            with pytest.raises(FilterInputValidationError):
                resample_filter.run()


def test_resample_filter_mock_data():
    """ Test the resample_filter with some mock data """

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
        np.max(np.abs(grid - np.array([64, 96])[:, np.newaxis, np.newaxis]), axis=0)
        < 16
    )
    image = np.zeros_like(circle)
    image = image + circle + 2.0 * rectangle + 3.0 * diamond
    image = numpy.expand_dims(image, axis=2)
    source_affine = np.eye(4)

    nifti_image = nib.Nifti2Image(image, affine=source_affine)

    # create NumpyImageContainer and NiftiImageContainer of this image
    numpy_image_container = NumpyImageContainer(image=image, affine=source_affine)
    nifti_image_container = NiftiImageContainer(nifti_image)

    # Create affine matrix for a translation of +10 along the x-axis
    affine_translate = AffineMatrixFilter()
    affine_translate.add_input(AffineMatrixFilter.KEY_TRANSLATION, (10.0, 0.0, 0.0))
    affine_translate.run()
    target_affine = affine_translate.outputs["affine"]
    target_shape = (128, 128, 1)

    # resample image without using the filter
    resampled_image = nil.image.resample_img(
        nifti_image, target_affine=target_affine, target_shape=target_shape
    )

    # run for the NumpyImageContainer
    resample_translate_filter = ResampleFilter()
    resample_translate_filter.add_input("affine", target_affine)
    resample_translate_filter.add_input("image", numpy_image_container)
    resample_translate_filter.add_input("shape", target_shape)
    resample_translate_filter.run()
    resampled_numpy_container = resample_translate_filter.outputs["image"].clone()

    # run for the NiftiImageContainer
    resample_translate_filter = ResampleFilter()
    resample_translate_filter.add_input("affine", target_affine)
    resample_translate_filter.add_input("image", nifti_image_container)
    resample_translate_filter.add_input("shape", target_shape)
    resample_translate_filter.run()
    resampled_nifti_container = resample_translate_filter.outputs["image"].clone()

    # compare outputs: image data
    numpy.testing.assert_array_equal(
        np.asanyarray(resampled_image.dataobj), resampled_numpy_container.image,
    )
    numpy.testing.assert_array_equal(
        np.asanyarray(resampled_image.dataobj), resampled_nifti_container.image,
    )

    # compare outputs: affine
    numpy.testing.assert_array_equal(
        resampled_image.affine, resampled_numpy_container.affine,
    )
    numpy.testing.assert_array_equal(
        resampled_image.affine, resampled_nifti_container.affine,
    )


# mock data: rotation, rotation_origin, translation, scale, expected_signal
MOCK_DATA = (
    (
        (0.0, 90.0, 0.0),  # rotate 90 degrees about y
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (100, 100, 100),
        ([40], [60], [60]),  # i, j, k coordinates where there is signal
    ),
    (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),  # translate +10 along x
        (1.0, 1.0, 1.0),
        (100, 100, 100),
        ([50], [60], [60]),
    ),
    (
        (0.0, 45.0, 0.0),  # rotate 45 degrees about y
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (100, 100, 100),
        ([50, 50], [60, 60], [64, 65]),
    ),
    (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (2.0, 2.0, 2.0),  # scale by factor 2 isotropically
        (100, 100, 100),
        ([55], [55], [55]),
    ),
    (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.5, 1.0, 1.0),  # scale by factor 0.5 along x
        (100, 100, 100),
        ([69, 70, 71], [60, 60, 60], [60, 60, 60]),
    ),
    (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (25.0, 25.0, 25.0),  # translate by +25 along each axis to centre the scaling
        (2.0, 2.0, 2.0),  # scale by factor 2.0 along x
        (50, 50, 50),  # resize in x direction to 50 voxels
        ([30], [30], [30]),
    ),
)


@pytest.mark.parametrize(
    "rotation, rotation_origin, translation, scale, target_shape, expected_signal",
    MOCK_DATA,
)
def test_resample_filter_single_point_transformations(
    rotation: float,
    rotation_origin: float,
    translation: float,
    scale: float,
    target_shape: tuple,
    expected_signal: list,
):
    # pylint: disable=too-many-locals, too-many-arguments
    """Tests the ResampleFilter by resampling an image comprising of points based on affines
    that are generated with rotation, translation, and scale parameters.  After resampling,
    `expected_signal` is a list of tuples defining voxel coordinates where signal should be,
    and `expected_background` is a list of tuples defining voxel coordinates where signal shouldn't
    be.

    :param rotation: Tuple of floats describing rotation angles about x, y, z
    :type rotation: float
    :param rotation_offset: Tuple of floats describing the coordinates of the rotation origin
    :type rotation_offset: float
    :param translation: Tuple of floats describing a translation vector
    :type translation: float
    :param scale: Tuple of floats describing scaling along x, y, z
    :type scale: Tuple[float, float, float]
    :param target_shape: Target shape after resampling
    :type target_shape: Tuple[float, float, float]
    :param expected_signal: List of [x,y,z] coordinates where signal should be expected
    :type expected_signal: list
    :param expected_background: List of [x,y,z] coordinates where background should be expected
    :type expected_background: list
    """
    # create blank image
    n_i = 100
    n_j = 100
    n_k = 100
    image = np.zeros((n_i, n_j, n_k))

    # define world coordinate origin (x,y,z) = (0,0,0) at (i,j,k) = (50,50,50)
    # and 1 voxel == 1mm isotropically
    # therefore according to RAS+:
    affine = np.array(((1, 0, 0, -50), (0, 1, 0, -50), (0, 0, 1, -50), (0, 0, 0, 1),))

    # define a vector in this space: world coords (10, 10, 10)
    vector_image_coords = np.rint(
        nil.image.coord_transform(10, 10, 10, np.linalg.inv(affine))
    ).astype(np.int32)

    # assign this point as signal (== 1)
    image[vector_image_coords[0], vector_image_coords[1], vector_image_coords[2]] = 1

    nii = nib.Nifti2Image(image, affine=affine)

    original_image_container = NiftiImageContainer(nifti_img=nii)

    # Create transformation affine.
    affine_filter = AffineMatrixFilter()
    affine_filter.add_input("affine", affine)
    affine_filter.add_input("rotation", rotation)
    affine_filter.add_input("rotation_origin", rotation_origin)
    affine_filter.add_input("translation", translation)
    affine_filter.add_input("scale", scale)

    # ResampleFilter
    resample_filter = ResampleFilter()
    resample_filter.add_input("image", original_image_container)
    resample_filter.add_input("shape", target_shape)
    resample_filter.add_parent_filter(affine_filter)

    resample_filter.run()
    transformed_image = resample_filter.outputs["image"]

    # Compare voxels where signal is above a threshold - the resampling will not preserve
    # exact values
    signal_threshold = 0.1

    signal_detected = transformed_image.image > signal_threshold
    signal_should_be_present = np.zeros_like(signal_detected, dtype=bool)
    signal_should_be_present[expected_signal] = True

    numpy.testing.assert_array_equal(signal_detected, signal_should_be_present)


def test_resample_filter_metadata():
    """ Tests the metadata output of the resample filter """
    test_image = TEST_NIFTI_ONES.clone()
    # add some meta data
    test_image.metadata = {
        "key1": 1,
        "key2": "two",
        "key3": [2, 4, 5],
    }
    resample_affine = 2 * np.eye(4)
    resample_filter = ResampleFilter()
    resample_filter.add_input("affine", resample_affine)
    resample_filter.add_input("image", test_image)
    resample_filter.add_input("shape", TEST_VOLUME_DIMENSIONS)
    resample_filter.run()

    valid_dict = {
        "key1": 1,
        "key2": "two",
        "key3": [2, 4, 5],
        "voxel_size": list(
            nib.affines.voxel_sizes(resample_filter.outputs["image"].affine)
        ),
    }
    assert resample_filter.outputs["image"].metadata == valid_dict
