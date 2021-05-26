"""Tests for asl_quantification.py"""

import os
import pytest
import jsonschema
import json
from copy import deepcopy
import numpy as np
import numpy.testing
import nibabel as nib
from asldro.containers.image import NiftiImageContainer
from asldro.validators.schemas.index import SCHEMAS
from asldro.filters.bids_output_filter import BidsOutputFilter
from asldro.pipelines.asl_quantification import asl_quantification
from asldro.filters.asl_quantification_filter import AslQuantificationFilter
from asldro.utils.general import splitext

TEST_VOLUME_DIMS = [4, 4, 4]


@pytest.fixture(name="image_data")
def image_data_fixture():
    control_image = np.ones(TEST_VOLUME_DIMS)
    label_image = np.ones(TEST_VOLUME_DIMS) * (1 - 0.001)
    m0_image = np.ones(TEST_VOLUME_DIMS)
    image_data = np.stack([m0_image, control_image, label_image], axis=3)
    return image_data


@pytest.fixture(name="test_pcasl_data")
def pcasl_data_fixture(tmp_path, image_data):
    """
    Creates PCASL test data with non-default parameters
    Returns the outputs of the BidsOutputFilter, and also saves
    data to disk in a temporary directory that persists during the tests
    """
    image_container = NiftiImageContainer(nib.Nifti1Image(image_data, affine=np.eye(4)))
    image_container.metadata = {
        "echo_time": 0.01,
        "repetition_time": [10.0, 5.0, 5.0],
        "excitation_flip_angle": 90,
        "mr_acq_type": "3D",
        "acq_contrast": "ge",
        "series_type": "asl",
        "series_number": 1,
        "series_description": "test asl series",
        "asl_context": "m0scan control label".split(),
        "label_type": "pcasl",
        "label_duration": 1.5,
        "post_label_delay": 2.0,
        "label_efficiency": 0.95,
        "lambda_blood_brain": 0.90,
        "t1_arterial_blood": 1.65,
        "image_flavour": "PERFUSION",
        "voxel_size": [1.0, 1.0, 1.0],
        "background_suppresion": False,
        "magnetic_field_strength": 3.0,
    }
    temp_dir = tmp_path / "data"
    temp_dir.mkdir()
    bids_output_filter = BidsOutputFilter()
    bids_output_filter.add_input(BidsOutputFilter.KEY_IMAGE, image_container)
    bids_output_filter.add_input(BidsOutputFilter.KEY_OUTPUT_DIRECTORY, str(temp_dir))
    bids_output_filter.run()
    return bids_output_filter.outputs


@pytest.fixture(name="test_pcasl_data_missing_params")
def pcasl_data_missing_params_fixture(tmp_path, image_data):
    """
    Creates PCASL test data missing parameters
    returns the outputs of the BidsOutputFilter, and also saves
    data to disk in a temporary directory that persists during the tests

    """
    image_container = NiftiImageContainer(nib.Nifti1Image(image_data, affine=np.eye(4)))
    image_container.metadata = {
        "echo_time": 0.01,
        "repetition_time": [10.0, 5.0, 5.0],
        "excitation_flip_angle": 90,
        "mr_acq_type": "3D",
        "acq_contrast": "ge",
        "series_type": "asl",
        "series_number": 2,
        "series_description": "test asl series",
        "asl_context": "m0scan control label".split(),
        "label_type": "pcasl",
        "label_duration": 1.5,
        "post_label_delay": 2.0,
        "label_efficiency": 0.95,
        "lambda_blood_brain": 0.90,
        "t1_arterial_blood": 1.65,
        "image_flavour": "PERFUSION",
        "voxel_size": [1.0, 1.0, 1.0],
        "background_suppresion": False,
        "magnetic_field_strength": 3.0,
    }
    temp_dir = tmp_path / "data"
    temp_dir.mkdir()
    bids_output_filter = BidsOutputFilter()
    bids_output_filter.add_input(BidsOutputFilter.KEY_IMAGE, image_container)
    bids_output_filter.add_input(BidsOutputFilter.KEY_OUTPUT_DIRECTORY, str(temp_dir))
    bids_output_filter.run()
    new_sidecar = bids_output_filter.outputs[BidsOutputFilter.KEY_SIDECAR]
    # remove  specific keys
    for key in [
        "PostLabelingDelay",
        "LabelingEfficiency",
        "LabelingDuration",
        "BolusCutOffDelayTime",
    ]:
        new_sidecar.pop(key, None)

    # re-save the sidecar
    with open(
        bids_output_filter.outputs[BidsOutputFilter.KEY_FILENAME][1], "w"
    ) as json_file:
        json.dump(
            new_sidecar,
            json_file,
            indent=4,
        )

    return bids_output_filter.outputs


def test_quantification_parameters_schema():
    schema = SCHEMAS["asl_quantification"]

    # try a valid schema
    valid_schema = {
        "QuantificationModel": "whitepaper",
        "PostLabelingDelay": 1.8,
        "LabelingEfficiency": 0.85,
        "BloodBrainPartitionCoefficient": 0.9,
        "T1ArterialBlood": 1.65,
        "ArterialSpinLabelingType": "PCASL",
        "LabelingDuration": 1.8,
    }
    jsonschema.validate(valid_schema, schema)
    valid_schema = {
        "QuantificationModel": "whitepaper",
        "PostLabelingDelay": 1.8,
        "BolusCutOffDelayTime": 0.85,
        "BloodBrainPartitionCoefficient": 0.9,
        "T1ArterialBlood": 1.65,
        "ArterialSpinLabelingType": "PASL",
        "LabelingDuration": 1.8,
    }
    jsonschema.validate(valid_schema, schema)


def test_asl_quantification_pcasl(test_pcasl_data):
    """Tests the asl_quantification pipeline with some pcasl mock data"""
    nifi_filename = test_pcasl_data["filename"][0]
    # try with no quantification parameter, and test_asl_data has
    # non default labelling parameters
    out = asl_quantification(nifi_filename)
    # no output directory, filenames should be empty
    assert out["filenames"] == {}

    # check the output has been correctly calculated
    numpy.testing.assert_array_equal(
        out["image"].image,
        AslQuantificationFilter.asl_quant_wp_casl(
            control=np.ones(TEST_VOLUME_DIMS),
            label=(1 - 0.001) * np.ones(TEST_VOLUME_DIMS),
            m0=np.ones(TEST_VOLUME_DIMS),
            lambda_blood_brain=0.9,
            label_duration=1.5,
            post_label_delay=2.0,
            label_efficiency=0.95,
            t1_arterial_blood=1.65,
        ),
    )
    assert out["quantification_parameters"] == {
        "QuantificationModel": "whitepaper",
        "PostLabelingDelay": 2.0,
        "LabelingEfficiency": 0.95,
        "BloodBrainPartitionCoefficient": 0.9,
        "ArterialSpinLabelingType": "PCASL",
        "LabelingDuration": 1.5,
        "T1ArterialBlood": 1.65,
    }


def test_asl_quantification_pcasl_output_files(test_pcasl_data, tmp_path):
    """Tests the asl_quantification pipeline with some pcasl mock data"""
    nifi_filename = test_pcasl_data["filename"][0]
    # try with no quantification parameters, and test_asl_data has
    # non default labelling parameters
    out = asl_quantification(nifi_filename, output_dir=tmp_path)
    # no output directory, filenames should be empty
    assert out["filenames"] == {
        "nifti": os.path.join(tmp_path, "sub-001_acq-001_asl_cbf.nii.gz"),
        "json": os.path.join(tmp_path, "sub-001_acq-001_asl_cbf.json"),
    }
    loaded_nifti = nib.load(out["filenames"]["nifti"])
    # check that the nifti that is loaded is the same as what the function outputs
    numpy.testing.assert_array_equal(loaded_nifti.dataobj, out["image"].image)
    # check that the JSON that is loaded is the same as the function's output
    with open(out["filenames"]["json"]) as json_file:
        loaded_json = json.load(json_file)
    assert loaded_json == out["image"].metadata


def test_asl_quantification_pcasl(test_pcasl_data_missing_params):
    """Tests the asl_quantification pipeline with some pcasl mock data"""
    nifi_filename = test_pcasl_data_missing_params["filename"][0]
    # try with no quantification parameter, and test_asl_data has
    # non default labelling parameters
    out = asl_quantification(nifi_filename)
    # no output directory, filenames should be empty
    assert out["filenames"] == {}

    # check the output has been correctly calculated
    numpy.testing.assert_array_equal(
        out["image"].image,
        AslQuantificationFilter.asl_quant_wp_casl(
            control=np.ones(TEST_VOLUME_DIMS),
            label=(1 - 0.001) * np.ones(TEST_VOLUME_DIMS),
            m0=np.ones(TEST_VOLUME_DIMS),
            lambda_blood_brain=0.9,
            label_duration=1.8,
            post_label_delay=1.8,
            label_efficiency=0.85,
            t1_arterial_blood=1.65,
        ),
    )
    assert out["quantification_parameters"] == {
        "QuantificationModel": "whitepaper",
        "PostLabelingDelay": 1.80,
        "LabelingEfficiency": 0.85,
        "BloodBrainPartitionCoefficient": 0.9,
        "ArterialSpinLabelingType": "PCASL",
        "LabelingDuration": 1.8,
        "T1ArterialBlood": 1.65,
    }
