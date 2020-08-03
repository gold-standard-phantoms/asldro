""" NiftiLoaderFilter tests """
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import nibabel as nib
import numpy as np

from asldro.filters.basefilter import FilterInputValidationError
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.containers.image import NiftiImageContainer


def test_nifti_loader_input_validation_no_input():
    """ Test all of the NiftiLoader input validation -
    No input filename (but some input so the filter will run)"""

    nifti_loader_filter = NiftiLoaderFilter()
    nifti_loader_filter.add_input("dummy", None)
    with pytest.raises(FilterInputValidationError):
        nifti_loader_filter.run()


def test_nifti_loader_input_validation_non_string_input():
    """ Test all of the NiftiLoader input validation -
    Non-string filename"""

    nifti_loader_filter = NiftiLoaderFilter()

    nifti_loader_filter.add_input("filename", 1)
    with pytest.raises(FilterInputValidationError):
        nifti_loader_filter.run()


def test_nifti_loader_input_validation_bad_nifti_filename():
    """ Test all of the NiftiLoader input validation -
    Bad NIFTI filename"""

    nifti_loader_filter = NiftiLoaderFilter()

    with TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "file.txt")
        Path(temp_file).touch()
        nifti_loader_filter.add_input("filename", temp_file)
        with pytest.raises(FilterInputValidationError):
            nifti_loader_filter.run()


def test_nifti_loader_input_validation_missing_nifti_file():
    """ Test all of the NiftiLoader input validation -
    Missing NIFTI file """

    nifti_loader_filter = NiftiLoaderFilter()

    with TemporaryDirectory() as temp_dir:
        # Missing NIFTI file
        temp_file = os.path.join(temp_dir, "file.nii")
        nifti_loader_filter.add_input("filename", temp_file)
        with pytest.raises(FilterInputValidationError):
            nifti_loader_filter.run()


def test_nifti_loader():
    """ Test the loading functionality """

    with TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, "image.nii")
        nib.save(
            img=nib.Nifti1Image(dataobj=np.zeros([3, 3, 3]), affine=np.eye(4)),
            filename=filename,
        )

        nifti_loader_filter = NiftiLoaderFilter()
        nifti_loader_filter.add_input("filename", filename)
        nifti_loader_filter.run()  # This should run OK

        assert isinstance(nifti_loader_filter.outputs["image"], NiftiImageContainer)
        assert (nifti_loader_filter.outputs["image"].image == np.zeros([3, 3, 3])).all()
