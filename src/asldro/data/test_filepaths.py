""" filepaths.py tests """

import os

from asldro.data.filepaths import GROUND_TRUTH_DATA


def test_file_paths_exist():
    """Check the ground truth file paths exist and are files"""
    for value in GROUND_TRUTH_DATA.values():
        assert os.path.isfile(value["json"])
        assert os.path.isfile(value["nii"])
