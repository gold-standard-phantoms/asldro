"""Load ASL BIDS Filter tests"""

import os
from shutil import copy
from tempfile import TemporaryDirectory

import pytest

import numpy as np
import numpy.testing
import nibabel as nib

from asldro.containers.image import BaseImageContainer, NiftiImageContainer
from asldro.filters.bids_output_filter import BidsOutputFilter

from asldro.utils.filter_validation import validate_filter_inputs
from asldro.filters.load_asl_bids_filter import LoadAslBidsFilter


TEST_VOLUME_DIMENSIONS = (32, 32, 32)


@pytest.fixture(name="test_data")
def test_data_fixture() -> BaseImageContainer:
    """creates a valid ASL image whichc an be saved using BidsOutputFilter
    returns a dictionary with the filenames and the data
    """
    # create test image data comprising of alternating 3D volumes with different values
    num_images = 9
    image_dims = [*TEST_VOLUME_DIMENSIONS]
    image_dims.append(num_images)
    image_data = np.zeros(image_dims)
    for i in range(num_images - 1):
        if i == 0:
            image_data[:, :, :, i] = np.ones(TEST_VOLUME_DIMENSIONS)
        elif i > 0:
            image_data[:, :, :, i] = np.ones(TEST_VOLUME_DIMENSIONS) * (i % 2 + 2)

    image = NiftiImageContainer(nib.Nifti2Image(image_data, affine=np.eye(4)))
    image.metadata = {
        "echo_time": 0.01,
        "repetition_time": [10.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        "excitation_flip_angle": 30,
        "mr_acq_type": "3D",
        "acq_contrast": "ge",
        "series_type": "asl",
        "series_number": 10,
        "series_description": "test asl series",
        "asl_context": "m0scan control label control label control label control label".split(),
        "label_type": "pcasl",
        "label_duration": 1.8,
        "post_label_delay": 1.8,
        "label_efficiency": 0.85,
        "lambda_blood_brain": 0.90,
        "t1_arterial_blood": 1.65,
        "image_flavour": "PERFUSION",
        "voxel_size": [1.0, 1.0, 1.0],
    }

    return image


def test_load_asl_bids_filter_validate_inputs(test_data):
    """Checks that a FilterInputValidationError is raised when the inputs to
    the LoadAslBidsFilter are incorrect or missing"""
    with TemporaryDirectory() as temp_dir:
        image = test_data
        bids_output_filter = BidsOutputFilter()
        bids_output_filter.add_input("image", image)
        bids_output_filter.add_input("output_directory", str(temp_dir))
        bids_output_filter.run()

        image_filename = bids_output_filter.outputs["filename"][0]
        sidecar_filename = bids_output_filter.outputs["filename"][1]
        aslcontext_filename = bids_output_filter.outputs["filename"][2]
        sidecar = bids_output_filter.outputs["sidecar"]

        image_wrong_extension = os.path.join(temp_dir, "image.img")
        copy(image_filename, image_wrong_extension)

        json_wrong_extension = os.path.join(temp_dir, "sidecar.jsn")
        copy(sidecar_filename, json_wrong_extension)

        aslcontext_wrong_extension = os.path.join(temp_dir, "aslcontext.csv")
        copy(aslcontext_filename, aslcontext_wrong_extension)

        # incorrect *_aslcontext.tsv contents 1: no volume_type
        incorrect_content_aslcontext_1 = os.path.join(
            temp_dir, "incorrect_aslcontext_1.tsv"
        )
        with open(incorrect_content_aslcontext_1, "w") as tsv_file:
            tsv_file.write("\n".join(image.metadata["asl_context"][1:]))
            tsv_file.close()

        # incorrect *_aslcontext.tsv contents 2: incorrect length
        incorrect_content_aslcontext_2 = os.path.join(
            temp_dir, "incorrect_aslcontext_1.tsv"
        )
        with open(incorrect_content_aslcontext_2, "w") as tsv_file:
            tsv_file.write("\n".join(image.metadata["asl_context"][:7]))
            tsv_file.close()

        input_validation_dictionary = {
            "image_filename": [
                False,
                image_filename,
                image_wrong_extension,
                os.path.join(temp_dir, "nonexistent.nii"),
                "image.img",
                1.0,
                image,
            ],
            "sidecar_filename": [
                False,
                sidecar_filename,
                json_wrong_extension,
                os.path.join(temp_dir, "nonexistent.json"),
                1.0,
                sidecar,
            ],
            "aslcontext_filename": [
                False,
                aslcontext_filename,
                aslcontext_wrong_extension,
                os.path.join(temp_dir, "nonexistent.jsn"),
                incorrect_content_aslcontext_1,
                incorrect_content_aslcontext_2,
                1.0,
                image.metadata["asl_context"],
            ],
        }

        validate_filter_inputs(LoadAslBidsFilter, input_validation_dictionary)


def test_load_asl_bids_filter_mock_data(test_data):
    """Tests the LoadAslBidsFilter with some mock data"""
    with TemporaryDirectory() as temp_dir:
        image = test_data
        bids_output_filter = BidsOutputFilter()
        bids_output_filter.add_input("image", image)
        bids_output_filter.add_input("output_directory", str(temp_dir))
        bids_output_filter.run()

        image_filename = bids_output_filter.outputs["filename"][0]
        sidecar_filename = bids_output_filter.outputs["filename"][1]
        aslcontext_filename = bids_output_filter.outputs["filename"][2]
        sidecar = bids_output_filter.outputs["sidecar"]

        load_bids_filter = LoadAslBidsFilter()
        load_bids_filter.add_input("image_filename", image_filename)
        load_bids_filter.add_input("sidecar_filename", sidecar_filename)
        load_bids_filter.add_input("aslcontext_filename", aslcontext_filename)

        load_bids_filter.run()

        # check outputs
        assert sidecar == load_bids_filter.outputs["sidecar"]
        # source image data should be equivalent to `image`
        numpy.testing.assert_array_equal(
            image.image, load_bids_filter.outputs["source"].image
        )
        numpy.testing.assert_array_equal(
            image.image[:, :, :, 0], load_bids_filter.outputs["m0"].image
        )
        assert load_bids_filter.outputs["m0"].metadata["asl_context"] == ["m0scan"]
        assert load_bids_filter.outputs["m0"].metadata["RepetitionTimePreparation"] == [
            10.0
        ]
        numpy.testing.assert_array_equal(
            image.image[:, :, :, 1::2], load_bids_filter.outputs["control"].image
        )
        assert load_bids_filter.outputs["control"].metadata["asl_context"] == [
            "control",
            "control",
            "control",
            "control",
        ]
        assert load_bids_filter.outputs["control"].metadata[
            "RepetitionTimePreparation"
        ] == [
            5.0,
            5.0,
            5.0,
            5.0,
        ]
        numpy.testing.assert_array_equal(
            image.image[:, :, :, 2::2], load_bids_filter.outputs["label"].image
        )
        assert load_bids_filter.outputs["label"].metadata["asl_context"] == [
            "label",
            "label",
            "label",
            "label",
        ]
        assert load_bids_filter.outputs["label"].metadata[
            "RepetitionTimePreparation"
        ] == [
            5.0,
            5.0,
            5.0,
            5.0,
        ]
