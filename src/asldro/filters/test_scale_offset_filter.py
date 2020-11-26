""" ScaleOffsetFilter tests """
import pytest
import numpy as np
import numpy.testing as nptesting
import nibabel as nib
from asldro.containers.image import NiftiImageContainer
from asldro.filters.scale_offset_filter import ScaleOffsetFilter


@pytest.fixture(name="test_image")
def fixture_test_image():
    """Create a test image of shape 3x3x3.
    Where each element (voxel) increases in value across
    each voxel as 0,1,2,3,...,26. The last axis changes the
    fastest (C-like index order)"""
    return NiftiImageContainer(
        nifti_img=nib.Nifti1Image(
            dataobj=np.reshape(np.array(range(3 ** 3)), (3, 3, 3)), affine=np.eye(4)
        )
    )


def test_scale_offset_filter_scale_only(test_image: NiftiImageContainer):
    """Test the scale offset filter scale only"""
    scale_offset_filter = ScaleOffsetFilter()
    scale_offset_filter.add_input("image", test_image)
    scale_offset_filter.add_input("scale", 0.5)
    scale_offset_filter.run()
    nptesting.assert_array_equal(
        scale_offset_filter.outputs["image"].image,
        np.reshape(np.array(range(3 ** 3)) * 0.5, (3, 3, 3)),
    )


def test_scale_offset_filter_offset_only(test_image: NiftiImageContainer):
    """Test the scale offset filter offset only"""
    scale_offset_filter = ScaleOffsetFilter()
    scale_offset_filter.add_input("image", test_image)
    scale_offset_filter.add_input("offset", -5)
    scale_offset_filter.run()
    nptesting.assert_array_equal(
        scale_offset_filter.outputs["image"].image,
        np.reshape(np.array(range(3 ** 3)) - 5, (3, 3, 3)),
    )


def test_scale_offset_filter_scale_and_offset(test_image: NiftiImageContainer):
    """Test the scale offset filter scale and offset"""
    scale_offset_filter = ScaleOffsetFilter()
    scale_offset_filter.add_input("image", test_image)
    scale_offset_filter.add_input("scale", 2)
    scale_offset_filter.add_input("offset", -5.0)
    scale_offset_filter.run()
    nptesting.assert_array_equal(
        scale_offset_filter.outputs["image"].image,
        np.reshape(np.array(range(3 ** 3)) * 2 - 5.0, (3, 3, 3)),
    )
