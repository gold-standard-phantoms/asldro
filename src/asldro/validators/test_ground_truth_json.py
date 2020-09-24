""" Tests for the ground truth JSON validator """
from jsonschema.exceptions import ValidationError
import pytest

from asldro.validators.ground_truth_json import validate_input


def test_valid_ground_truth_json():
    """ Test valid grouth truth raises no error """
    json = {
        "quantities": [
            "perfusion_rate",
            "transit_time",
            "m0",
            "t1",
            "t2",
            "t2_star",
            "seg_label",
        ],
        "segmentation": {"grey_matter": 1, "white_matter": 2, "csf": 3, "vascular": 4},
        "parameters": {"lambda_blood_brain": 0.9, "t1_arterial_blood": 1.65},
    }
    validate_input(input_dict=json)


def test_ground_truth_typos_a():
    """ Test ground with typos in required fields raises error """
    json = {
        "quantties": [
            "perfusion_rate",
            "transit_time",
            "m0",
            "t1",
            "t2",
            "t2_star",
            "seg_label",
        ],
        "segmentation": {"grey_matter": 1, "white_matter": 2, "csf": 3, "vascular": 4},
        "parameters": {"lambda_blood_brain": 0.9, "t1_arterial_blood": 1.65},
    }
    with pytest.raises(ValidationError):
        validate_input(input_dict=json)


def test_ground_truth_typos_b():
    """ Test ground with typos in required fields raises error """
    json = {
        "quantities": [
            "perfusion_rate",
            "transit_time",
            "m0",
            "t1",
            "t2",
            "t2_star",
            "seg_label",
        ],
        "seg": {"grey_matter": 1, "white_matter": 2, "csf": 3, "vascular": 4},
        "parameters": {"lambda_blood_brain": 0.9, "t1_arterial_blood": 1.65},
    }
    with pytest.raises(ValidationError):
        validate_input(input_dict=json)


def test_ground_truth_typos_c():
    """ Test ground with typos in required fields raises error """
    json = {
        "quantities": [
            "perfusion_rate",
            "transit_time",
            "m0",
            "t1",
            "t2",
            "t2_star",
            "seg_label",
        ],
        "segmentation": {"grey_matter": 1, "white_matter": 2, "csf": 3, "vascular": 4},
        "params": {"lambda_blood_brain": 0.9, "t1_arterial_blood": 1.65},
    }
    with pytest.raises(ValidationError):
        validate_input(input_dict=json)


def test_ground_truth_json_missing_quantities():
    """ Test missing 'quantities' property raises error """
    json = {
        "segmentation": {"grey_matter": 1, "white_matter": 2, "csf": 3, "vascular": 4}
    }
    with pytest.raises(ValidationError):
        validate_input(input_dict=json)


def test_ground_truth_json_missing_segmentation():
    """ Test missing 'segmentation' property raises error """
    json = {
        "quantities": [
            "perfusion_rate",
            "transit_time",
            "m0",
            "t1",
            "t2",
            "t2_star",
            "seg_label",
        ]
    }
    with pytest.raises(ValidationError):
        validate_input(input_dict=json)


def test_ground_truth_missing_parameters():
    """ Test missing 'segmentation' property raises error """
    json = {
        "quantities": [
            "perfusion_rate",
            "transit_time",
            "m0",
            "t1",
            "t2",
            "t2_star",
            "seg_label",
        ],
        "segmentation": {"grey_matter": 1, "white_matter": 2, "csf": 3, "vascular": 4},
    }
    with pytest.raises(ValidationError):
        validate_input(input_dict=json)
