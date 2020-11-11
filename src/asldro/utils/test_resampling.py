""" Tests for the utility functions in resampling.py """
# pylint: disable=duplicate-code

from typing import Tuple, Union
import pytest
import numpy as np
import numpy.testing
import nibabel as nib
import nilearn as nil
import asldro.utils.resampling as rs
from asldro.containers.image import (
    NiftiImageContainer,
    BaseImageContainer,
    NumpyImageContainer,
)

import asldro.data.affine_test_data as atd

TEST_VOLUME_DIMENSIONS = (32, 32, 32)
TEST_NIFTI_ONES = nib.Nifti2Image(np.zeros(TEST_VOLUME_DIMENSIONS), affine=np.eye(4))
TEST_NIFTI_CONTAINER_ONES = NiftiImageContainer(nifti_img=TEST_NIFTI_ONES)
TEST_NUMPY_CONTAINER_ONES = NumpyImageContainer(
    np.zeros(TEST_VOLUME_DIMENSIONS), affine=np.eye(4)
)

# mock data: image, translation, rotation, rotation_origin,
# target_shape, expected_target_affine, expected_resampled_affine
MOCK_DATA = (
    (
        TEST_NIFTI_ONES,  # image
        (0.0, 0.0, 0.0),  # translation
        (0.0, 0.0, 0.0),  # rotation
        (0.0, 0.0, 0.0),  # rotation_origin
        (32, 32, 32),  # target_shape
        (
            10.0,
            10.0,
            10.0,
            1.0,
        ),  # expected_target
        (
            10.0,
            10.0,
            10.0,
            1.0,
        ),  # expected_resampled
    ),
    (
        TEST_NIFTI_ONES,  # image
        (0.0, 0.0, 0.0),  # translation
        (0.0, 0.0, 0.0),  # rotation
        (0.0, 0.0, 0.0),  # rotation_origin
        (16, 16, 16),  # target_shape
        (
            20.0,
            20.0,
            20.0,
            1.0,
        ),  # expected_target
        (
            20.0,
            20.0,
            20.0,
            1.0,
        ),  # expected_resampled
    ),
    (
        TEST_NIFTI_ONES,  # image
        (90.0, 0.0, 0.0),  # translation
        (0.0, 0.0, 0.0),  # rotation
        (0.0, 0.0, 0.0),  # rotation_origin
        (16, 16, 16),  # target_shape
        (
            -70.0,
            20.0,
            20.0,
            1.0,
        ),  # expected_target
        (
            20.0,
            20.0,
            20.0,
            1.0,
        ),  # expected_resampled
    ),
    (
        TEST_NIFTI_CONTAINER_ONES,  # image
        (0.0, 0.0, 0.0),  # translation
        (0.0, 45.0, 0.0),  # rotation
        (0.0, 0.0, 0.0),  # rotation_origin
        (16, 16, 16),  # target_shape
        (
            -0.000000000,
            20.000000000,
            28.284271247,
            1.000000000,
        ),  # expected_target
        (
            20.0,
            20.0,
            20.0,
            1.0,
        ),  # expected_resampled
    ),
    (
        TEST_NUMPY_CONTAINER_ONES,  # image
        (30.0, 0.0, 0.0),  # translation
        (0.0, 45.0, 0.0),  # rotation
        (0.0, 0.0, 5.0),  # rotation_origin
        (16, 16, 16),  # target_shape
        (
            -17.677669530,
            20.000000000,
            8.535533906,
            1.000000000,
        ),  # expected_target
        (
            20.0,
            20.0,
            20.0,
            1.0,
        ),  # expected_resampled
    ),
)


@pytest.mark.parametrize(
    "image, translation, rotation, rotation_origin, target_shape,"
    "expected_target, expected_resampled",
    MOCK_DATA,
)
def test_transform_resample_affine(
    image: Union[nib.Nifti1Image, nib.Nifti2Image, BaseImageContainer],
    translation: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    rotation_origin: Tuple[float, float, float],
    target_shape: Tuple[int, int, int],
    expected_target: Tuple[float, float, float, float],
    expected_resampled: Tuple[float, float, float, float],
):
    # pylint: disable=too-many-arguments
    """test the function transform_resample_affine
    To simplify inputs, the resultant affines are multiplied by the vector (10.0, 10.0, 10.0, 1.0)
    and this value compared withe expected values


    :param image: The input image
    :type image: Union[nib.Nifti1Image, nib.Nifti2Image, BaseImageContainer]
    :param translation: vector to translate in the image by in world space
    :type translation: Tuple[float, float, float]
    :param rotation: angles to rotate the object by in world space
    :type rotation: Tuple[float, float, float]
    :param rotation_origin: coordinates for the centre point of rotation in world space
    :type rotation_origin: Tuple[float, float, float]
    :param target_shape: target shape for the resampled image
    :type target_shape: Tuple[int, int, int]
    :param expected_target: expected value of target_affine @ vector (homogeneous coords)
    :type expected_target: Tuple[float, float, float, float]
    :param expected_resampled: expected value of target_affine @ vector (homogeneous coords)
    :type expected_resampled: Tuple[float, float, float, float]

    """
    target_affine, resampled_affine = rs.transform_resample_affine(
        image, translation, rotation, rotation_origin, target_shape
    )

    vector = (10.0, 10.0, 10.0, 1.0)

    numpy.testing.assert_array_almost_equal(expected_target, target_affine @ vector)
    numpy.testing.assert_array_almost_equal(
        expected_resampled, resampled_affine @ vector
    )


def test_transform_resample_image_mock_data():
    """ Test the transform_resample_image function with mock data"""
    # Create some synthetic data

    rotation = (0.0, 0.0, 45.0)
    translation = (0.0, 10.0, 0.0)
    target_shape = (64, 64, 1)
    (nifti_image, rotation_origin) = create_test_image()

    # use transform_resample_affine to obtain the affine
    target_affine_1, resampled_affine = rs.transform_resample_affine(
        nifti_image, translation, rotation, rotation_origin, target_shape
    )

    # resample using nilearn function
    resampled_nifti_1 = nil.image.resample_img(
        nifti_image, target_affine=target_affine_1, target_shape=target_shape
    )

    # transform and resample with the function
    resampled_nifti_2, target_affine_2 = rs.transform_resample_image(
        nifti_image, translation, rotation, rotation_origin, target_shape
    )
    # the images should be identical
    numpy.testing.assert_array_equal(
        resampled_nifti_1.dataobj, resampled_nifti_2.dataobj
    )
    # target affines should be identical
    numpy.testing.assert_array_equal(target_affine_1, target_affine_2)
    # resampled affine, and the affine of the image that was resampled with the function should
    # be identical
    numpy.testing.assert_array_equal(resampled_affine, resampled_nifti_2.affine)


def create_test_image() -> Tuple[NiftiImageContainer, Tuple[float, float, float]]:
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

    image = np.expand_dims(image, axis=2)

    # define world coordinate origin (x,y,z) = (0,0,0) at (i,j,k) = (64,64,1)
    # and 1 voxel == 1mm isotropically
    # therefore according to RAS+:
    source_affine = np.array(
        ((1, 0, 0, -64), (0, 1, 0, -64), (0, 0, 1, -0.5), (0, 0, 0, 1))
    )

    rotation_origin = tuple(
        np.array(nil.image.coord_transform(75, 32, 0, source_affine)).astype(float)
    )

    image[
        tuple(
            np.rint(
                nil.image.coord_transform(
                    rotation_origin[0],
                    rotation_origin[1],
                    rotation_origin[2],
                    np.linalg.inv(source_affine),
                )
            ).astype(np.int32)
        )
    ] = 5.0

    image[
        tuple(
            np.rint(
                nil.image.coord_transform(
                    0,
                    0,
                    0,
                    np.linalg.inv(source_affine),
                )
            ).astype(np.int32)
        )
    ] = 6.0
    return (nib.Nifti2Image(image, affine=source_affine), rotation_origin)


@pytest.mark.parametrize(
    "theta, expected",
    atd.ROT_X_TEST_DATA,
)
def test_rot_x_mat(theta: float, expected: np.ndarray):
    """ Tests rot_x_mat with some angles, comparing against expected values """
    numpy.testing.assert_array_almost_equal(rs.rot_x_mat(theta), expected, 6)


@pytest.mark.parametrize(
    "theta, expected",
    atd.ROT_Y_TEST_DATA,
)
def test_rot_y_mat(theta: float, expected: np.ndarray):
    """ Tests rot_y_mat with some angles, comparing against expected values """
    numpy.testing.assert_array_almost_equal(rs.rot_y_mat(theta), expected, 6)


@pytest.mark.parametrize(
    "theta, expected",
    atd.ROT_Z_TEST_DATA,
)
def test_rot_z_mat(theta: float, expected: np.ndarray):
    """ Tests rot_z_mat with some angles, comparing against expected values """
    numpy.testing.assert_array_almost_equal(rs.rot_z_mat(theta), expected, 6)


@pytest.mark.parametrize(
    "vector, expected",
    atd.TRANSLATE_TEST_DATA,
)
def test_translate_mat(vector: Tuple[float, float, float], expected: np.ndarray):
    """ Tests translate_mat with some translation vectors, comparing against expected values """
    numpy.testing.assert_array_almost_equal(rs.translate_mat(vector), expected, 6)


@pytest.mark.parametrize(
    "scale, expected",
    atd.SCALE_TEST_DATA,
)
def test_scale_mat(scale: Tuple[float, float, float], expected: np.ndarray):
    """ Tests scale_mat with some scale factors, comparing against expected values """
    numpy.testing.assert_array_almost_equal(rs.scale_mat(scale), expected, 6)
