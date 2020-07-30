import pytest
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from asldro.filters.basefilter import FilterInputValidationError
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.containers.image import ImageContainer


def test_nifti_loader_input_validation_no_input():
    """ Test all of the NiftiLoader input validation - 
    No input filename (but some input so the filter will run)"""

    f = NiftiLoaderFilter()
    f.add_input("dummy", None)
    with pytest.raises(FilterInputValidationError):
        f.run()


def test_nifti_loader_input_validation_non_string_input():
    """ Test all of the NiftiLoader input validation - 
    Non-string filename"""

    f = NiftiLoaderFilter()

    f.add_input("filename", 1)
    with pytest.raises(FilterInputValidationError):
        f.run()


def test_nifti_loader_input_validation_bad_nifti_filename():
    """ Test all of the NiftiLoader input validation - 
    Bad NIFTI filename"""

    f = NiftiLoaderFilter()

    with TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "file.txt")
        Path(temp_file).touch()
        f.add_input("filename", temp_file)
        with pytest.raises(FilterInputValidationError):
            f.run()


def test_nifti_loader_input_validation_missing_nifti_file():
    """ Test all of the NiftiLoader input validation - 
    Missing NIFTI file """

    f = NiftiLoaderFilter()

    with TemporaryDirectory() as temp_dir:
        # Missing NIFTI file
        temp_file = os.path.join(temp_dir, "file.nii")
        f.add_input("filename", temp_file)
        with pytest.raises(FilterInputValidationError):
            f.run()


def test_nifti_loader():
    """ Test the loading functionality """

    with TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, "image.nii")
        nib.save(
            img=nib.Nifti1Image(dataobj=np.zeros([3, 3, 3]), affine=np.eye(4)),
            filename=filename,
        )

        f = NiftiLoaderFilter()
        f.add_input("filename", filename)
        f.run()  # This should run OK

        assert isinstance(f.outputs["image"], ImageContainer)
        assert (f.outputs["image"].image == np.zeros([3, 3, 3])).all()
