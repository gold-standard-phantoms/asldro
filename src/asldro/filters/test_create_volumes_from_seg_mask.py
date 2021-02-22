"""CreateVolumesFromSegMask tests"""

# pylint: disable=duplicate-code

from copy import deepcopy
import pytest
import numpy as np
import jsonschema

import numpy.testing
import nibabel as nib

from asldro.containers.image import BaseImageContainer, NiftiImageContainer
from asldro.filters.create_volumes_from_seg_mask import CreateVolumesFromSegMask
from asldro.utils.filter_validation import validate_filter_inputs
from asldro.filters.ground_truth_loader import GroundTruthLoaderFilter
from asldro.validators.schemas.index import SCHEMAS


@pytest.fixture(name="validation_data")
def input_validation_dict_fixture():
    """Returns an object of tuples containing test data for input validation of the 
    GroundTruthLoaderFilter"""

    seg_data_4D = np.stack([i * np.ones((2, 2, 3), dtype=np.uint16) for i in range(4)])
    seg_data_4D_5vals = np.stack(
        [i * np.ones((2, 2, 3), dtype=np.uint16) for i in range(5)]
    )
    seg_data_5D = np.stack([i * np.ones((1, 4, 4, 4)) for i in range(4)])
    seg_mask_4D = NiftiImageContainer(nib.Nifti2Image(seg_data_4D, np.eye(4)))
    seg_mask_4D_float = NiftiImageContainer(
        nib.Nifti2Image(seg_data_4D.astype(np.float32), np.eye(4))
    )
    seg_mask_5D = NiftiImageContainer(nib.Nifti2Image(seg_data_5D, np.eye(4)))
    label_values = [0, 1, 2, 3]
    label_names = ["reg0", "reg1", "reg2", "reg3"]
    quantities = {
        "quant1": [0.0, 1.0, 2.0, 3.0],
        "quant2": [0.0, 2.0, 4.0, 3.0],
    }
    quantities_fail = {
        "quant1": [0.0, 1.0, 2.0],
        "quant2": [0.0, 2.0, 4.0],
    }

    units = ["unit1", "unit2"]

    return {
        "seg_mask": seg_mask_4D,
        "label_values": label_values,
        "label_names": label_names,
        "units": units,
        "quantities": quantities,
        "input_validation_dict": {
            "seg_mask": [
                False,
                seg_mask_4D,
                seg_mask_5D,
                seg_mask_4D_float,
                seg_data_4D,
                seg_data_4D_5vals,
                1.0,
                "str",
            ],
            "label_values": [False, label_values, label_values[:3], 1.0, "str"],
            "label_names": [False, label_names, label_names[:3], 1.0, "str"],
            "quantities": [False, quantities, quantities_fail, 1.0, "str"],
            "units": [False, units, ["unit1"], 1.0],
        },
    }


def test_create_volumes_from_seg_mask_validate_inputs(validation_data: dict):
    """Check a FilterInputValidationError is raised when the inputs to the
    CreateVolumesFromSegMask filter are incorrect or missing"""

    validate_filter_inputs(
        CreateVolumesFromSegMask, validation_data["input_validation_dict"]
    )


def test_create_volumes_from_seg_mask_with_mock_data(validation_data: dict):
    """Test the CreateVolumesFromSegMask filter with some mock data"""

    input_data = deepcopy(validation_data)
    input_data.pop("input_validation_dict")
    create_volumes_filter = CreateVolumesFromSegMask()
    create_volumes_filter.add_inputs(input_data)
    create_volumes_filter.run()

    # check the outputs
    # image_info
    assert create_volumes_filter.outputs["image_info"] == {
        "quantities": ["quant1", "quant2", "seg_label"],
        "segmentation": {"reg0": 0, "reg1": 1, "reg2": 2, "reg3": 3,},
        "units": ["unit1", "unit2", ""],
    }

    seg_mask = input_data["seg_mask"]

    # check that the values in each region match what they should be
    image: NiftiImageContainer = create_volumes_filter.outputs["image"]
    for i, quantity in enumerate(input_data["quantities"].keys()):
        test_image = image.image[:, :, :, :, i]
        for j, region in enumerate(input_data["label_values"]):
            test_image = image.image[:, :, :, :, i]
            numpy.testing.assert_array_equal(
                test_image[seg_mask.image == region],
                input_data["quantities"][quantity][j],
            )

    # check that the last volume is equal to the seg_mask image
    numpy.testing.assert_array_equal(image.image[:, :, :, :, -1], seg_mask.image)
    # check image shapes
    numpy.testing.assert_array_equal(image.shape, (4, 2, 2, 3, 3))
    numpy.testing.assert_array_equal(image.header["dim"], (5, 4, 2, 2, 3, 3, 1, 1))

    # Validate the outputs using the ground truth schema and GroundTruthLoaderFilter
    # copy image info then add a parameters object
    image_info = deepcopy(create_volumes_filter.outputs["image_info"])
    image_info["parameters"] = {
        "t1_arterial_blood": 1.65,
        "lambda_blood_brain": 0.9,
        "magnetic_field_strength": 3.0,
    }
    # validate against the ground truth schema
    jsonschema.validate(image_info, SCHEMAS["ground_truth"])

    ground_truth_loader = GroundTruthLoaderFilter()
    ground_truth_loader.add_input(GroundTruthLoaderFilter.KEY_IMAGE, image)
    ground_truth_loader.add_inputs(image_info)
    # should run with no errors
    ground_truth_loader.run()

    # Try with a 3D seg_mask
    seg_mask.image = np.stack([i * np.ones((2, 3), dtype=np.uint16) for i in range(4)])
    create_volumes_filter = CreateVolumesFromSegMask()
    create_volumes_filter.add_inputs(input_data)
    create_volumes_filter.run()
    # check image shapes
    numpy.testing.assert_array_equal(
        create_volumes_filter.outputs["image"].shape, (4, 2, 3, 1, 3)
    )
    numpy.testing.assert_array_equal(
        create_volumes_filter.outputs["image"].header["dim"], (5, 4, 2, 3, 1, 3, 1, 1)
    )
