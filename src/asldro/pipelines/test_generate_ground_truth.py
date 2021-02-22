from copy import deepcopy
import os
from nibabel.nifti1 import Nifti1Image
import pytest
import json
import jsonschema
from tempfile import TemporaryDirectory
import numpy as np
from asldro.pipelines.generate_ground_truth import generate_hrgt
import numpy.testing
import nibabel as nib

from asldro.containers.image import NiftiImageContainer
from asldro.validators.schemas.index import SCHEMAS


@pytest.fixture(name="validation_data")
def input_data_fixture():
    """ Fixture with test data"""

    return {
        "seg_mask_container": NiftiImageContainer(
            nib.Nifti2Image(
                np.stack([i * np.ones((2, 2, 3), dtype=np.uint16) for i in range(4)]),
                np.eye(4),
            )
        ),
        "seg_mask_float_container": NiftiImageContainer(
            nib.Nifti2Image(
                np.stack(
                    [0.5 * i * np.ones((2, 2, 3), dtype=np.float64) for i in range(7)]
                ),
                np.eye(4),
            )
        ),
        "hrgt_params": {
            "label_values": [0, 1, 2, 3],
            "label_names": ["reg0", "reg1", "reg2", "reg3"],
            "quantities": {
                "quant1": [0.0, 1.0, 2.0, 3.0],
                "quant2": [0.0, 2.0, 4.0, 3.0],
            },
            "units": ["unit1", "unit2"],
            "parameters": {
                "t1_arterial_blood": 1.65,
                "lambda_blood_brain": 0.9,
                "magnetic_field_strength": 3.0,
            },
        },
    }


def test_hrgt_params_schema(validation_data: dict):
    """Check that the example hrgt_params passes the json schema"""
    jsonschema.validate(validation_data["hrgt_params"], SCHEMAS["generate_hrgt_params"])

    # try something that should fail - swap type for one of the arrays
    d = deepcopy(validation_data["hrgt_params"])
    d["label_names"] = d["label_values"]
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(d, SCHEMAS["generate_hrgt_params"])


def test_generate_hrgt(validation_data: dict):
    """Test generate_hrgt function"""
    with TemporaryDirectory() as temp_dir:
        json_filename = os.path.join(temp_dir, "hrgt_params.json")
        with open(json_filename, "w") as json_file:
            json.dump(validation_data["hrgt_params"], json_file, indent=4)

        nifti_filename = os.path.join(temp_dir, "seg_mask.nii.gz")
        nib.save(validation_data["seg_mask_container"].nifti_image, nifti_filename)

        results = generate_hrgt(json_filename, nifti_filename, temp_dir)

        saved_nifti: Nifti1Image = nib.load(os.path.join(temp_dir, "hrgt.nii.gz"))
        with open(os.path.join(temp_dir, "hrgt.json"), "r") as json_file:
            saved_json = json.load(json_file)

        # validate the json with the ground truth schema
        jsonschema.validate(saved_json, SCHEMAS["ground_truth"])
        # confirm it is the same data that the function returns
        assert saved_json == results["image_info"]
        numpy.testing.assert_array_equal(saved_nifti.dataobj, results["image"].image)


def test_generate_hrgt_float_seg_mask(validation_data: dict):
    """Test generate_hrgt function with float seg_mask data"""
    with TemporaryDirectory() as temp_dir:
        json_filename = os.path.join(temp_dir, "hrgt_params.json")
        with open(json_filename, "w") as json_file:
            json.dump(validation_data["hrgt_params"], json_file, indent=4)

        nifti_filename = os.path.join(temp_dir, "seg_mask.nii.gz")
        nib.save(
            validation_data["seg_mask_float_container"].nifti_image, nifti_filename
        )

        results = generate_hrgt(json_filename, nifti_filename, temp_dir)

        saved_nifti: Nifti1Image = nib.load(os.path.join(temp_dir, "hrgt.nii.gz"))
        with open(os.path.join(temp_dir, "hrgt.json"), "r") as json_file:
            saved_json = json.load(json_file)

        # validate the json with the ground truth schema
        jsonschema.validate(saved_json, SCHEMAS["ground_truth"])
        # confirm it is the same data that the function returns
        assert saved_json == results["image_info"]
        numpy.testing.assert_array_equal(saved_nifti.dataobj, results["image"].image)
