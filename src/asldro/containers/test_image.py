""" ImageContainer tests """
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import os
from typing import List

import nibabel as nib
import numpy as np
import numpy.testing
import pytest

from asldro import definitions
from asldro.containers.image import (
    BaseImageContainer,
    NiftiImageContainer,
    NumpyImageContainer,
    UNITS_METERS,
    UNITS_MICRONS,
    UNITS_SECONDS,
    UNITS_MILLIMETERS,
    UNITS_MILLISECONDS,
    UNITS_MICROSECONDS,
    SPATIAL_DOMAIN,
    INVERSE_DOMAIN,
)


@pytest.fixture
def icbm_v1_nifti():
    """ Returns the ICBM v1 test nifti data """
    return NiftiImageContainer(
        nifti_img=nib.load(
            os.path.join(definitions.ROOT_DIR, "data", "hrgt_ICBM_2009a_NLS_v1.nii.gz")
        )
    )


# NiftiImageContainer TESTS

NIFTI_HEADER = {
    "dim_info": 57,
    "dim": [5, 128, 96, 24, 2, 5, 1, 1],
    "intent_p1": 0.0,
    "intent_p2": 0.0,
    "intent_p3": 0.0,
    "datatype": 4,  # 16bit INT
    "bitpix": 16,
    "slice_start": 0,
    "pixdim": [-1.0, 2.0, 2.0, 2.2, 2000.0, 1.0, 1.0, 1.0],
    "vox_offset": 0.0,
    "slice_end": 23,
    "xyzt_units": 10,
    "cal_max": 1162.0,
    "cal_min": 0.0,
    "slice_duration": 0.0,
    "toffset": 0.0,
    "glmax": 0,
    "glmin": 0,
    "descrip": b"FSL3.3\x00 v2.25 NIfTI-1 Single file format",
    "aux_file": b"",
    "qform_code": 1,
    "sform_code": 1,
    "quatern_b": -1.94510681403e-26,
    "quatern_c": -0.996708512306,
    "quatern_d": -0.081068739295,
    "qoffset_x": 117.855102539,
    "qoffset_y": -35.7229423523,
    "qoffset_z": -7.24879837036,
    "srow_x": [-2.0, 0.0, 0.0, 117.86],
    "srow_y": [-0.0, 1.97, -0.36, -35.72],
    "srow_z": [0.0, 0.32, 2.17, -7.25],
    "intent_name": b"",
}

NIFTI2_HEADER_EXCLUDES = ["glmax", "glmin"]
NIFTI_AFFINE = np.array(
    [
        NIFTI_HEADER["srow_x"],
        NIFTI_HEADER["srow_y"],
        NIFTI_HEADER["srow_z"],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


@pytest.fixture
def nifti_image_containers_a() -> List[NiftiImageContainer]:
    """ Returns a list of NiftiImageContainers. The first items uses
    a Nifti1Image. The second item uses a Nifti2Image """

    containers = []
    for image_type in [nib.Nifti1Image, nib.Nifti2Image]:
        temp_img = image_type(np.zeros((1, 1, 1)), affine=np.eye(4))
        header = temp_img.header

        for key, value in NIFTI_HEADER.items():
            # Skip adding headers that don't exist in a NIFTI2 header
            if image_type == nib.Nifti2Image and key in NIFTI2_HEADER_EXCLUDES:
                continue

            header[key] = value

        image_container = NiftiImageContainer(
            nifti_img=image_type(
                np.zeros(NIFTI_HEADER["dim"][1 : 1 + NIFTI_HEADER["dim"][0]]),
                affine=NIFTI_AFFINE,
                header=header,
            )
        )
        containers.append(image_container)
    return containers


def test_nifti_image_container_image(
    nifti_image_containers_a: List[NiftiImageContainer]
):
    """ Test that the image is returned from a NiftiImageContainer """

    for image_container in nifti_image_containers_a:
        assert (image_container.image == np.zeros((128, 96, 24, 2, 5))).all()
        assert image_container.image.shape == np.zeros((128, 96, 24, 2, 5)).shape


def test_nifti_image_container_shape(
    nifti_image_containers_a: List[NiftiImageContainer]
):
    """ Test that the shape is correctly returned from a NiftiImageContainer """

    for image_container in nifti_image_containers_a:
        assert image_container.shape == (128, 96, 24, 2, 5)


def test_nifti_image_container_affine(
    nifti_image_containers_a: List[NiftiImageContainer]
):
    """ Test that the image is returned from a NiftiImageContainer """

    for image_container in nifti_image_containers_a:
        assert (
            image_container.affine
            == np.array(
                [
                    [-2.0, 0.0, 0.0, 117.86],
                    [-0.0, 1.97, -0.36, -35.72],
                    [0.0, 0.32, 2.17, -7.25],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        ).all()


def test_nifti_image_container_time_units(
    nifti_image_containers_a: List[NiftiImageContainer]
):
    """ Test that the correct time units are returned from a NiftiImageContainer """

    for image_container in nifti_image_containers_a:
        assert image_container.time_units == UNITS_SECONDS


def test_nifti_image_container_space_units(
    nifti_image_containers_a: List[NiftiImageContainer]
):
    """ Test that the correct space_units are returned from a NiftiImageContainer """

    for image_container in nifti_image_containers_a:
        assert image_container.space_units == UNITS_MILLIMETERS


def test_nifti_image_xyzt_units_reading(
    nifti_image_containers_a: List[NiftiImageContainer]
):
    """ Test the xyzt_units are interpreted properly """
    for image_container in nifti_image_containers_a:

        image_container.header["xyzt_units"] = 1 | 8  # meters and seconds
        assert image_container.space_units == UNITS_METERS
        assert image_container.time_units == UNITS_SECONDS

        image_container.header["xyzt_units"] = 2 | 16  # mm and msec
        assert image_container.space_units == UNITS_MILLIMETERS
        assert image_container.time_units == UNITS_MILLISECONDS

        image_container.header["xyzt_units"] = 3 | 24  # mm and usec
        assert image_container.space_units == UNITS_MICRONS
        assert image_container.time_units == UNITS_MICROSECONDS


def test_nifti_image_container_voxel_size_mm(
    nifti_image_containers_a: List[NiftiImageContainer]
):
    """ Test that the correct voxel size is returned from a NiftiImageContainer """

    for image_container in nifti_image_containers_a:
        # NIFTI header has mm in xyzt_units
        np.testing.assert_array_almost_equal(
            image_container.voxel_size_mm, np.array([2.0, 2.0, 2.2])
        )
        image_container.space_units = UNITS_METERS
        np.testing.assert_array_almost_equal(
            image_container.voxel_size_mm, np.array([2000.0, 2000.0, 2200.0])
        )
        image_container.space_units = UNITS_MICRONS
        np.testing.assert_array_almost_equal(
            image_container.voxel_size_mm, np.array([2.0e-3, 2.0e-3, 2.2e-3])
        )


def test_nifti_image_container_time_step_seconds(
    nifti_image_containers_a: List[NiftiImageContainer]
):
    """ Test that the correct time step is returned from a NiftiImageContainer """

    for image_container in nifti_image_containers_a:
        # NIFTI header has seconds in xytz_units
        assert image_container.time_step_seconds == 2000.0
        image_container.time_units = UNITS_MILLISECONDS
        assert image_container.time_step_seconds == 2e6
        image_container.time_units = UNITS_MICROSECONDS
        assert image_container.time_step_seconds == 2e9


def test_nifti_image_container_has_header(
    nifti_image_containers_a: List[NiftiImageContainer]
):
    """ Test that the nifti header is returned from a NiftiImageContainer"""

    for image_container in nifti_image_containers_a:
        assert image_container.header is not None


# NumpyImageContainer TESTS


@pytest.fixture
def numpy_image_container() -> NumpyImageContainer:
    """ Creates and returns a NumpyImageContainer for testing """
    return NumpyImageContainer(
        image=np.ones((2, 3, 4, 5, 6)),
        affine=np.array(
            [
                [-2.0, 0.0, 0.0, 117.86],
                [-0.0, 1.97, -0.36, -35.72],
                [0.0, 0.32, 2.17, -7.25],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        space_units=UNITS_MILLIMETERS,
        time_units=UNITS_SECONDS,
        voxel_size=[2.0, 2.0, 2.2],
        time_step=2000.0,
    )


def test_numpy_image_container_image(numpy_image_container):
    """ Test that the image is returned from a NumpyImageContainer """
    np.testing.assert_array_equal(numpy_image_container.image, np.ones((2, 3, 4, 5, 6)))


def test_numpy_image_container_shape(numpy_image_container):
    """ Test that the shape is returned from a NumpyImageContainer """
    assert numpy_image_container.shape == (2, 3, 4, 5, 6)


def test_numpy_image_container_affine(numpy_image_container):
    """ Test that the affine is returned from a NumpyImageContainer """
    assert (
        numpy_image_container.affine
        == np.array(
            [
                [-2.0, 0.0, 0.0, 117.86],
                [-0.0, 1.97, -0.36, -35.72],
                [0.0, 0.32, 2.17, -7.25],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    ).all()


def test_numpy_image_container_time_units(numpy_image_container):
    """ Test that the correct time units are returned from a NumpyImageContainer """
    assert numpy_image_container.time_units == UNITS_SECONDS


def test_numpy_image_container_space_units(numpy_image_container):
    """ Test that the correct space units are returned from a NumpyImageContainer """
    assert numpy_image_container.space_units == UNITS_MILLIMETERS


def test_numpy_image_container_voxel_size_mm(
    numpy_image_container: NumpyImageContainer
):
    """ Test that the correct voxel size is returned from a NumpyImageContainer """

    # NIFTI header has mm in xyzt_units
    np.testing.assert_array_almost_equal(
        numpy_image_container.voxel_size_mm, np.array([2.0, 2.0, 2.2])
    )
    numpy_image_container.space_units = UNITS_METERS
    np.testing.assert_array_almost_equal(
        numpy_image_container.voxel_size_mm, np.array([2000.0, 2000.0, 2200.0])
    )
    numpy_image_container.space_units = UNITS_MICRONS
    np.testing.assert_array_almost_equal(
        numpy_image_container.voxel_size_mm, np.array([2.0e-3, 2.0e-3, 2.2e-3])
    )


def test_numpy_image_container_time_step_seconds(
    numpy_image_container: NumpyImageContainer
):
    """ Test that the correct time step is returned from a NumpyImageContainer """

    # NIFTI header has seconds in xytz_units
    assert numpy_image_container.time_step_seconds == 2000.0
    numpy_image_container.time_units = UNITS_MILLISECONDS
    assert numpy_image_container.time_step_seconds == 2e6
    numpy_image_container.time_units = UNITS_MICROSECONDS
    assert numpy_image_container.time_step_seconds == 2e9


def test_numpy_image_container_has_header(numpy_image_container: NumpyImageContainer):
    """ Test that a header is not returned from a NumpyImageContainer """

    assert numpy_image_container.header is None


def test_numpy_image_container_has_nifti(numpy_image_container: NumpyImageContainer):
    """ Test that a nifty is not returned from a NumpyImageContainer """

    assert not numpy_image_container.has_nifti


def test_image_container_unexpected_arguments():
    """ Check that passing unexpected arguments raises an error """
    with pytest.raises(TypeError):
        NumpyImageContainer(image=np.zeros((3, 3, 3)), unexpected="test")
    with pytest.raises(TypeError):
        NiftiImageContainer(
            nib.Nifti1Pair(np.zeros((3, 3, 3)), affine=np.eye(4)), unexpected="test"
        )


def test_image_container_spatial_domain_initialisation():
    """ Check that passing a string not in SPATIAL_DOMAIN or INVERSE_DOMAIN to
    data_domain raises an exception ( and vice versa )"""
    with pytest.raises(ValueError):
        NumpyImageContainer(image=np.zeros((3, 3, 3)), data_domain="foobar")

    image_container = NumpyImageContainer(
        image=np.zeros((3, 3, 3)), data_domain=SPATIAL_DOMAIN
    )  # OK
    assert image_container.data_domain == SPATIAL_DOMAIN

    image_container = NumpyImageContainer(
        image=np.zeros((3, 3, 3)), data_domain=INVERSE_DOMAIN
    )  # OK
    assert image_container.data_domain == INVERSE_DOMAIN


# Clone tests


def general_image_container_clone_tests(
    image_container: BaseImageContainer, cloned_image_container: BaseImageContainer
):
    """ Check that the image container is cloned correctly """
    assert image_container.data_domain == cloned_image_container.data_domain
    # Check images are equal, but not the same object
    numpy.testing.assert_array_equal(
        image_container.image, cloned_image_container.image
    )
    assert not id(image_container.affine) == id(cloned_image_container.affine)

    # Check affines are equal, but not the same object
    numpy.testing.assert_array_equal(
        image_container.affine, cloned_image_container.affine
    )
    assert not id(image_container.affine) == id(cloned_image_container.affine)

    assert image_container.space_units == cloned_image_container.space_units
    assert image_container.time_units == cloned_image_container.time_units

    # Check voxel sizes are equal, but not the same object
    numpy.testing.assert_array_equal(
        image_container.voxel_size_mm, cloned_image_container.voxel_size_mm
    )
    assert not id(image_container.voxel_size_mm) == id(
        cloned_image_container.voxel_size_mm
    )

    assert image_container.time_step_seconds == cloned_image_container.time_step_seconds


def test_numpy_image_container_clone():
    """ Check that the numpy image container is cloned correctly """
    # Use none of the default parameters
    image_container = NumpyImageContainer(
        image=np.ones(shape=(3, 4, 5)),
        affine=np.eye(4) * 2,
        space_units=UNITS_METERS,
        time_units=UNITS_MICROSECONDS,
        voxel_size=np.array([1, 2, 3]),
        time_step=0.5,
        data_domain=INVERSE_DOMAIN,
    )

    cloned_image_container = image_container.clone()
    general_image_container_clone_tests(image_container, cloned_image_container)


def test_nifti_image_container_clone(
    nifti_image_containers_a: List[NiftiImageContainer]
):
    """ Check that the NIFTI image container is cloned correctly """

    for image_container in nifti_image_containers_a:

        cloned_image_container = image_container.clone()
        general_image_container_clone_tests(image_container, cloned_image_container)

        # Also, check the headers are cloned correctly
        assert not id(image_container.header) == id(cloned_image_container.header)

        assert image_container.header == cloned_image_container.header
