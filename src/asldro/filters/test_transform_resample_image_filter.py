""" Transform Resample image Filter Tests """

from copy import deepcopy
from typing import Tuple
import logging
import pytest
import numpy as np
import numpy.testing

from numpy.linalg import inv
import nibabel as nib
import nilearn as nil

from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.containers.image import NiftiImageContainer
from asldro.filters.transform_resample_image_filter import TransformResampleImageFilter

# import matplotlib.pyplot as plt
# from nilearn.plotting import show

logger = logging.getLogger(__name__)

TEST_VOLUME_DIMENSIONS = (32, 32, 32)
TEST_NIFTI_ONES = NiftiImageContainer(
    nifti_img=nib.Nifti2Image(np.zeros(TEST_VOLUME_DIMENSIONS), affine=np.eye(4))
)

INPUT_VALIDATION_DICTIONARY = {
    "image": (TEST_NIFTI_ONES, np.ones(TEST_VOLUME_DIMENSIONS), "str", 1.0),
    "target_shape": (
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
def test_transform_resample_image_filter_validate_inputs(validation_data: dict):
    """ Check a FilterInputValidationError is raised when the
    inputs to the TransformResampleImageFilter are incorrect or missing
    """
    # Check with all data that should pass
    xr_obj_filter = TransformResampleImageFilter()
    test_data = deepcopy(validation_data)

    for data_key in test_data:
        xr_obj_filter.add_input(data_key, test_data[data_key][0])

    xr_obj_filter.run()

    for inputs_key in validation_data:
        xr_obj_filter = TransformResampleImageFilter()
        test_data = deepcopy(validation_data)

        # remove key
        test_data.pop(inputs_key)
        for data_key in test_data:
            xr_obj_filter.add_input(data_key, test_data[data_key][0])

        # "image" is not optional so should raise an error
        # all the others are so should run with no issues
        if inputs_key == "image":
            with pytest.raises(FilterInputValidationError):
                xr_obj_filter.run()
        else:
            xr_obj_filter.run()

        # Try data that should fail
        for test_value in validation_data[inputs_key][1:]:
            xr_obj_filter = TransformResampleImageFilter()
            for data_key in test_data:
                xr_obj_filter.add_input(data_key, test_data[data_key][0])
            xr_obj_filter.add_input(inputs_key, test_value)
            with pytest.raises(FilterInputValidationError):
                xr_obj_filter.run()


def transform_resample_image_function(
    image: nib.Nifti2Image,
    translation: tuple,
    rotation: tuple,
    rotation_origin: tuple,
    target_shape: tuple,
) -> Tuple[nib.Nifti2Image, nib.Nifti2Image, nib.Nifti2Image, np.array]:
    """[summary]

    :param image: [description]
    :type image: BaseImageContainer
    :param translation: [description]
    :type translation: tuple
    :param rotation: [description]
    :type rotation: tuple
    :param rotation_origin: [description]
    :type rotation_origin: tuple
    :param target_shape: [description]
    :type target_shape: tuple
    :return: [description]
    :rtype: BaseImageContainer
    """
    scale = np.array(image.shape) / np.array(target_shape)
    output_voxel_size = scale  # Perhaps `output_voxel_size` should be a parameter?
    source_image_to_world = image.affine

    # The affine matrices used in rotation and translation
    # for the motion model
    motion_model_rotation_affine = (
        rot_x_mat(rotation[0]) @ rot_y_mat(rotation[1]) @ rot_z_mat(rotation[2])
    )
    inverse_motion_model_rotation_affine = np.linalg.inv(motion_model_rotation_affine)
    motion_model_translation_affine = translate_mat(translation)
    rotation_centre_translation_affine = translate_mat(rotation_origin)
    inverse_rotation_centre_translation_affine = translate_mat(
        (-rotation_origin[0], -rotation_origin[1], -rotation_origin[2])
    )

    # Assuming coordinates are in world space, will perform the rotation component
    # of the motion model
    world_space_rotation_affine = (
        rotation_centre_translation_affine
        @ motion_model_rotation_affine
        @ inverse_rotation_centre_translation_affine
    )

    # Assuming coordinates are in rotated world space, will perform the translation
    # component of the motion model
    rotated_world_space_translation_affine = (
        motion_model_rotation_affine
        @ motion_model_translation_affine
        @ inverse_motion_model_rotation_affine
    )

    # Assuming coordinates are in world space, will perform the rotation and
    # translation components of the motion model
    translated_rotated_world_space_affine = (
        rotated_world_space_translation_affine @ world_space_rotation_affine
    )

    # Will perform a voxel sampling as per the desired output_voxel_size (must be in
    # some form of world space coordinates system)
    resampling_affine = scale_mat(1 / output_voxel_size)

    # The full transformation from world space to target image space, in order:
    # - the motion model is applied
    # - the resampling is applied
    # - the FOV is created at the correct location by image-space translation
    world_to_target_affine = (
        translate_mat(
            np.array(target_shape) / 2
        )  # Set the target origin to the image centre (in image space)
        @ resampling_affine
        @ translated_rotated_world_space_affine
    )
    # Invert to get target to world space, as per the `affine` nifti specification
    target_affine = inv(world_to_target_affine)

    sampled_image: nib.Nifti2Image = nil.image.resample_img(
        image, target_affine=target_affine, target_shape=target_shape
    )
    # Calculate the new affine, which only takes into account the
    sampled_image_affine = inv(resampling_affine @ inv(image.affine))
    sampled_image.set_sform(sampled_image_affine)
    return (sampled_image, target_affine)


def test_transform_resample_image_filter_mock_data():
    """ Test the transform_resampe_image_filter with some mock data """

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
        ((1, 0, 0, -64), (0, 1, 0, -64), (0, 0, 1, -0.5), (0, 0, 0, 1))
    )

    nifti_image = nib.Nifti2Image(image, affine=source_affine)

    rotation = (0.0, 0.0, 45.0)
    rotation_origin = tuple(
        np.array(nil.image.coord_transform(30, 30, 0, source_affine)).astype(float)
    )
    # rotation_origin = (0.0, 0.0, 0.0)
    translation = (0.0, 10.0, 0.0)
    target_shape = (64, 64, 1)

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

    # create NiftiImageContainer of this image
    nifti_image_container = NiftiImageContainer(nifti_image)

    xr_obj_filter = TransformResampleImageFilter()
    xr_obj_filter.add_input(
        TransformResampleImageFilter.KEY_IMAGE, nifti_image_container
    )
    xr_obj_filter.add_input(TransformResampleImageFilter.KEY_ROTATION, rotation)
    xr_obj_filter.add_input(
        TransformResampleImageFilter.KEY_ROTATION_ORIGIN, rotation_origin
    )
    xr_obj_filter.add_input(TransformResampleImageFilter.KEY_TRANSLATION, translation)
    xr_obj_filter.add_input(TransformResampleImageFilter.KEY_TARGET_SHAPE, target_shape)

    xr_obj_filter.run()
    new_nifti_container: NiftiImageContainer = xr_obj_filter.outputs[
        TransformResampleImageFilter.KEY_IMAGE
    ]

    ### function called here
    str_nifti, target_affine = transform_resample_image_function(
        nifti_image, translation, rotation, rotation_origin, target_shape
    )
    # visually check
    """
    plt.figure()
    plt.imshow(np.fliplr(np.rot90(nifti_image_container.image, axes=(1, 0))))
    plt.title("original image")
    plt.axis("image")

    plt.figure()
    plt.imshow(np.fliplr(np.rot90(new_nifti_container.image, axes=(1, 0))))
    plt.title("transformed and resampled with filter")
    plt.axis("image")
    plt.text(
        0,
        new_nifti_container.shape[1],
        f"rotation={rotation}"
        "\n"
        f"rotation origin-{rotation_origin}"
        "\n"
        f"translation={translation}"
        "\n"
        f"shape = {new_nifti_container.shape}"
        "\n",
        {"color": "white"},
    )

    plt.figure()
    plt.imshow(np.fliplr(np.rot90(str_nifti.dataobj, axes=(1, 0))))
    plt.title("transformed and resampled with function")
    plt.axis("image")
    plt.text(
        0,
        str_nifti.dataobj.shape[1],
        f"rotation={rotation}"
        "\n"
        f"rotation origin-{rotation_origin}"
        "\n"
        f"translation={translation}"
        "\n"
        f"shape = {str_nifti.dataobj.shape}"
        "\n",
        {"color": "white"},
    )
    plt.show()
    """
    numpy.testing.assert_array_equal(str_nifti.dataobj, new_nifti_container.image)
    # Affines do not yet match because extra functionality is required
    # numpy.testing.assert_array_equal(str_nifti.affine, new_nifti_container.affine)


def rot_x_mat(theta: float) -> np.array:
    """creates a 4x4 affine performing rotations about x

    :param theta: angle to rotate about x in degrees
    :type theta: float
    :return: 4x4 affine for rotating about x
    :rtype: np.array
    """
    theta = np.radians(theta)
    return np.array(
        (
            (1, 0, 0, 0),
            (0, np.cos(theta), -np.sin(theta), 0),
            (0, np.sin(theta), np.cos(theta), 0),
            (0, 0, 0, 1),
        )
    )


def rot_y_mat(theta: float) -> np.array:
    """creates a 4x4 affine performing rotations about y

    :param theta: angle to rotate about y in degrees
    :type theta: float
    :return: 4x4 affine for rotating about y
    :rtype: np.array
    """
    theta = np.radians(theta)
    return np.array(
        (
            (np.cos(theta), 0, np.sin(theta), 0),
            (0, 1, 0, 0),
            (-np.sin(theta), 0, np.cos(theta), 0),
            (0, 0, 0, 1),
        )
    )


def rot_z_mat(theta: float) -> np.array:
    """creates a 4x4 affine performing rotations about z

    :param theta: angle to rotate about z in degrees
    :type theta: float
    :return: 4x4 affine for rotating about z
    :rtype: np.array
    """
    theta = np.radians(theta)
    return np.array(
        (
            (np.cos(theta), -np.sin(theta), 0, 0),
            (np.sin(theta), np.cos(theta), 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    )


def translate_mat(translation: Tuple[float, float, float]) -> np.array:
    """Creates a 4x4 affine performing translations

    :param vector: describes (x, y, z) to translate along respective axes 
    :type vector: Tuple[float, float, float]
    :return: 4x4 affine for translation
    :rtype: np.array
    """
    return np.array(
        (
            (1, 0, 0, translation[0]),
            (0, 1, 0, translation[1]),
            (0, 0, 1, translation[2]),
            (0, 0, 0, 1),
        )
    )


def scale_mat(scale: Tuple[float, float, float]) -> np.array:
    """Creates a 4x4 affine performing scaling

    :param vector: describes (sx, sy, sz) scaling factors 
    :type vector: Tuple[float, float, float]
    :return: 4x4 affine for scaling
    :rtype: np.array
    """
    return np.array(
        ((scale[0], 0, 0, 0), (0, scale[1], 0, 0), (0, 0, scale[2], 0), (0, 0, 0, 1))
    )


def transform_resample_image_function_reverse(
    image: nib.Nifti2Image,
    translation: tuple,
    rotation: tuple,
    rotation_origin: tuple,
    target_shape: tuple,
) -> Tuple[nib.Nifti2Image, np.array]:
    """[summary]

    :param image: [description]
    :type image: BaseImageContainer
    :param translation: [description]
    :type translation: tuple
    :param rotation: [description]
    :type rotation: tuple
    :param rotation_origin: [description]
    :type rotation_origin: tuple
    :param target_shape: [description]
    :type target_shape: tuple
    :return: [description]
    :rtype: BaseImageContainer
    """
    scale = np.array(image.shape) / np.array(target_shape)
    affine_1 = image.affine
    affine_2 = (
        translate_mat(translation)
        @ translate_mat(tuple(1 * np.array(rotation_origin)))
        @ rot_x_mat(rotation[0])
        @ rot_y_mat(rotation[1])
        @ rot_z_mat(rotation[2])
        @ translate_mat(tuple(-1 * np.array(rotation_origin)))
        @ affine_1
    )

    affine_3 = affine_2 @ translate_mat(tuple(image.affine[:3, 3])) @ scale_mat(scale)

    return (
        nil.image.resample_img(
            image, target_affine=affine_3, target_shape=target_shape
        ),
        affine_3,
    )
