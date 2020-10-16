""" InvertImageFilter tests """
import pytest

import numpy as np
from numpy.testing import assert_array_equal
import nibabel as nib

from asldro.filters.basefilter import FilterInputValidationError
from asldro.filters.invert_image_filter import InvertImageFilter
from asldro.containers.image import NiftiImageContainer, NumpyImageContainer


def test_invert_image_filter_outputs():
    """ Test the invert image filter validator throws appropriate errors"""
    invert_image_filter = InvertImageFilter()

    invert_image_filter.add_input("image", 123)  # wrong image input type
    with pytest.raises(FilterInputValidationError):
        invert_image_filter.run()


def test_invert_image_filter_with_nifti():
    """ Test the invert image filter works correctly with NiftiImageContainer"""
    invert_image_filter = InvertImageFilter()
    array = np.ones(shape=(3, 3, 3, 1), dtype=np.float32)

    img = nib.Nifti2Image(dataobj=array, affine=np.eye(4))
    nifti_image_container = NiftiImageContainer(nifti_img=img)

    invert_image_filter.add_input("image", nifti_image_container)
    invert_image_filter.run()

    assert_array_equal(invert_image_filter.outputs["image"].image, -array)


def test_invert_image_filter_with_numpy():
    """ Test the invert image filter works correctly with NumpyImageContainer"""
    invert_image_filter = InvertImageFilter()
    array = np.ones(shape=(3, 3, 3, 1), dtype=np.float32)

    nifti_image_container = NumpyImageContainer(image=array)

    invert_image_filter.add_input("image", nifti_image_container)
    invert_image_filter.run()

    assert_array_equal(invert_image_filter.outputs["image"].image, -array)
