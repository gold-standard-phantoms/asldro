""" Tests for the ASL DRO Command Line Interface (CLI) """
import os
import argparse
from tempfile import TemporaryDirectory
import pytest
from asldro.cli import DirType, FileType, HrgtType
from asldro.data.filepaths import GROUND_TRUTH_DATA


def test_file_type_init_error_checking():
    """ The the FileType __init__ function error checking """
    with pytest.raises(TypeError):
        FileType(extensions=123)
    with pytest.raises(TypeError):
        FileType(extensions=["json", 123])


def test_file_type_init():
    """ The the FileType __init__ function """
    file_type = FileType()
    assert file_type.extensions == []
    assert not file_type.should_exist

    # Check preceeding dots are stripped
    file_type = FileType(extensions=["json", ".zip", "tar.gz"])
    assert file_type.extensions == ["json", "zip", "tar.gz"]


def test_file_type_call():
    """ Test the FileType __call__ function """
    with TemporaryDirectory() as temp_dir:
        # Is directory, raise error
        with pytest.raises(argparse.ArgumentTypeError):
            FileType()(path=temp_dir)
        file_path = os.path.join(temp_dir, "some.file")

        # Should run without error (is valid file path)
        assert FileType()(path=file_path) == file_path

        # We expect the file to exist - should throw error
        with pytest.raises(argparse.ArgumentTypeError):
            FileType(should_exist=True)(path=file_path)

        # Try again, but with an existing file
        with open(file_path, "w") as file_obj:
            file_obj.write("foobar")  # some text

        # Should run without error (is valid file path and exists)
        assert FileType(should_exist=True)(path=file_path) == file_path

        json_file_path = os.path.join(temp_dir, "some.json")

        # Wrong file extensions
        with pytest.raises(argparse.ArgumentTypeError):
            FileType(extensions=["zip", "tar"])(json_file_path)

        # OK file extensions
        assert (
            FileType(extensions=["zip", "tar", "json"])(json_file_path)
            == json_file_path
        )
        assert FileType(extensions=["json"])(json_file_path) == json_file_path


def test_dir_type_init():
    """The the DirType __init__ function"""
    dir_type = DirType()
    assert not dir_type.should_exist
    dir_type = DirType(should_exist=True)
    assert dir_type.should_exist


def test_dir_type_call():
    """Test the DirType __call__ function"""
    with TemporaryDirectory() as temp_dir:
        # Is directory, OK
        DirType()(path=temp_dir)
        dir_path = os.path.join(temp_dir, "some.dir")

        # Should run without error (is valid direcory path)
        assert DirType()(path=dir_path) == dir_path

        # We expect the directory to exist - should throw error
        with pytest.raises(argparse.ArgumentTypeError):
            DirType(should_exist=True)(path=dir_path)

        # Try again, but with an existing file
        os.mkdir(dir_path)

        # Should run without error (is valid file path and exists)
        assert DirType(should_exist=True)(path=dir_path) == dir_path


def test_hrgt_type_call():
    """Test the HrgtType __call__ function"""
    for test in ["foo", 1]:
        with pytest.raises(argparse.ArgumentTypeError):
            HrgtType()(test)
    for test in GROUND_TRUTH_DATA.keys():
        HrgtType()(test)  # ok
