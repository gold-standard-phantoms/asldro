""" Utility Functions for the ASL DRO """

from typing import Tuple, Union
import numpy as np
from numpy.linalg import inv
import nibabel as nib
import nilearn as nil

from asldro.containers.image import BaseImageContainer


def transform_resample_image(
    image: Union[nib.Nifti1Image, nib.Nifti2Image, BaseImageContainer],
    translation: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    rotation_origin: Tuple[float, float, float],
    target_shape: Tuple[int, int, int],
) -> Tuple[Union[nib.Nifti2Image, nib.Nifti2Image], np.ndarray]:
    """
    Transforms and resamples a nibabel NIFTI image in world-space

    :param image: The input image
    :type image: Union[nib.Nifti1Image, nib.Nifti2Image]
    :param translation: vector to translate in the image by in world space
    :type translation: Tuple[float, float, float]
    :param rotation: angles to rotate the object by in world space
    :type rotation: Tuple[float, float, float]
    :param rotation_origin: coordinates for the centre point of rotation in world space
    :type rotation_origin: Tuple[float, float, float]
    :param target_shape: target shape for the resampled image
    :type target_shape: Tuple[int, int, int]
    :return: [`resampled_image`, `target_affine`]. `resampled_image` is the input image with the
        transformation and resampling applied. `target_affine` is the affine that was used to
        resample the image. Note `resampled_image` has an affine that only corresponds to voxel
        scaling and not motion, i.e. the image FOV is the same as the input image FOV.
    :rtype: Tuple[Union[nib.Nifti2Image, nib.Nifti2Image], np.array]
    """
    target_affine_with_motion, target_affine_no_motion = transform_resample_affine(
        image, translation, rotation, rotation_origin, target_shape
    )
    resampled_image: nib.Nifti2Image = nil.image.resample_img(
        image, target_affine=target_affine_with_motion, target_shape=target_shape
    )
    resampled_image.set_sform(target_affine_no_motion)
    return resampled_image, target_affine_with_motion


def transform_resample_affine(
    image: Union[nib.Nifti1Image, nib.Nifti2Image, BaseImageContainer],
    translation: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    rotation_origin: Tuple[float, float, float],
    target_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the affine matrices that transform and resample an image in world-space. Note that
    while an image (NIFTI or BaseImageContainer derived) is accepted an an argument, the image
    is not acutally resampled.

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
    :return: [`target_affine_with_motion`, `target_affine_no_motion`]. `target_affine_with_motion`
        is the affine to supply to a resampling function.  After resampling theresampled image's
        affine should be set to to `target_affine_no_motion` so that it only has the
        image formation (scaling) operation performed (not the motion)
    :rtype: Tuple[np.array, np.array]
    """
    scale = np.array(image.shape) / np.array(target_shape)
    output_voxel_size = scale  # Perhaps `output_voxel_size` should be a parameter?
    source_image_to_world = image.affine

    # The affine matrices used in rotation and translation
    # for the motion model
    motion_model_rotation_affine = (
        rot_x_mat(rotation[0]) @ rot_y_mat(rotation[1]) @ rot_z_mat(rotation[2])
    )
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

    # Assuming coordinates are in world space, will perform the rotation and
    # translation components of the motion model
    translated_rotated_world_space_affine = (
        motion_model_translation_affine @ world_space_rotation_affine
    )

    # Will perform a voxel sampling as per the desired output_voxel_size (must be in
    # some form of world space coordinates system)
    resampling_affine = scale_mat(1 / output_voxel_size)

    # The full transformation from world space to target image space, in order:
    # - the motion model is applied
    # - the resampling is applied
    # - the FOV is created at the correct location by image-space translation

    # set the target origin to the input image origin - this needs to account
    # for the image scaling so divide by the output voxel size
    resampling_translate_affine = inv(
        translate_mat(source_image_to_world[:3, 3] / output_voxel_size)
    )

    world_to_target_affine = (
        resampling_translate_affine
        @ resampling_affine
        @ translated_rotated_world_space_affine
    )
    # Invert to get target to world space, as per the `affine` nifti specification
    target_affine_with_motion = inv(world_to_target_affine)
    target_affine_no_motion = inv(resampling_translate_affine @ resampling_affine)

    return (target_affine_with_motion, target_affine_no_motion)


def rot_x_mat(theta: float) -> np.ndarray:
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


def rot_y_mat(theta: float) -> np.ndarray:
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


def rot_z_mat(theta: float) -> np.ndarray:
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


def translate_mat(
    translation: Union[Tuple[float, float, float], np.ndarray]
) -> np.ndarray:
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


def scale_mat(scale: Union[Tuple[float, float, float], np.ndarray]) -> np.ndarray:
    """Creates a 4x4 affine performing scaling

    :param vector: describes (sx, sy, sz) scaling factors
    :type vector: Tuple[float, float, float]
    :return: 4x4 affine for scaling
    :rtype: np.array
    """
    return np.array(
        ((scale[0], 0, 0, 0), (0, scale[1], 0, 0), (0, 0, scale[2], 0), (0, 0, 0, 1))
    )
