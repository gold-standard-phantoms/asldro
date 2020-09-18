""" Tests for the ASL DRO Command Line Interface (CLI) """
import os
import argparse
from tempfile import TemporaryDirectory
import pytest
from asldro.cli import FileType


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
