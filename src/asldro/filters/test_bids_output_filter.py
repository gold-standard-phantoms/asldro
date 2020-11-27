""" Tests for BidsOutputFilter """

from copy import deepcopy
from unittest.mock import Mock, patch
import os
import datetime
import json
from tempfile import TemporaryDirectory
import pytest

import numpy as np
import numpy.testing
import nibabel as nib

from asldro.filters.bids_output_filter import BidsOutputFilter
from asldro.filters.basefilter import FilterInputValidationError
from asldro.containers.image import NiftiImageContainer

from asldro import __version__ as asldro_version

TEST_VOLUME_DIMENSIONS = (32, 32, 32)
TEST_NIFTI_ONES = nib.Nifti2Image(
    np.ones(TEST_VOLUME_DIMENSIONS),
    affine=np.array(((1, 0, 0, -16), (0, 1, 0, -16), (0, 0, 1, -16), (0, 0, 0, 1),)),
)
TEST_NIFTI_CON_ONES = NiftiImageContainer(
    nifti_img=TEST_NIFTI_ONES,
    metadata={"series_number": 1, "series_type": "structural", "modality": "T1w",},
)

INPUT_VALIDATION_DICT = {
    "image": [False, TEST_NIFTI_CON_ONES, TEST_NIFTI_ONES, 1, "str"],
    "output_directory": [False, "tempdir_placeholder", 1, "str"],
    "filename_prefix": [True, "prefix", 1, TEST_NIFTI_CON_ONES],
}

METDATA_VALIDATION_DICT_STRUCT = {
    "series_number": [False, 1, "001", 1.0],
    "series_type": [False, "structural", "struct", 1],
    "modality": [False, "T1w", "t1w", 1],
}

METDATA_VALIDATION_DICT_ASL = {
    "series_number": [False, 1, "001", 1.0],
    "series_type": [False, "asl", "struct", 1],
    "asl_context": [False, ["m0scan", "control", "label"], 1, "str"],
    "label_duration": [False, 1.8, 1, "str"],
    "label_type": [False, "pcasl", 1, TEST_NIFTI_ONES],
    "image_flavour": [False, "PERFUSION", 1, TEST_NIFTI_ONES],
    "post_label_delay": [False, 1.8, 1, (3.2, 4.5, 6.7)],
}

METDATA_VALIDATION_DICT_GROUND_TRUTH = {
    "series_number": [False, 1, "001", 1.0],
    "series_type": [False, "ground_truth", "struct", 1],
    "quantity": [False, "perfusion_rate", 1, 1.0],
    "units": [False, "ml/100g/min", 1, 1.0],
}


def test_bids_output_filter_validate_inputs():
    """ Check a FilterInputValidationError is raised when the inputs to the
    BidsOutputFilter are incorrect or missing """
    with TemporaryDirectory() as temp_dir:
        test_filter = BidsOutputFilter()
        test_data = deepcopy(INPUT_VALIDATION_DICT)
        test_data["output_directory"][1] = temp_dir

        # check with inputs that should pass
        for data_key in test_data:
            test_filter.add_input(data_key, test_data[data_key][1])

        test_filter.run()

    for inputs_key in INPUT_VALIDATION_DICT:
        with TemporaryDirectory() as temp_dir:
            test_data = deepcopy(INPUT_VALIDATION_DICT)
            test_data["output_directory"][1] = temp_dir
            test_filter = BidsOutputFilter()
            is_optional: bool = test_data[inputs_key][0]

            # remove key
            test_data.pop(inputs_key)
            for data_key in test_data:
                test_filter.add_input(data_key, test_data[data_key][1])

            # optional inputs should run without issue
            if is_optional:
                test_filter.run()
            else:
                with pytest.raises(FilterInputValidationError):
                    test_filter.run()

            # Try data that should fail
            for test_value in INPUT_VALIDATION_DICT[inputs_key][2:]:
                test_filter = BidsOutputFilter()
                for data_key in test_data:
                    test_filter.add_input(data_key, test_data[data_key][1])
                test_filter.add_input(inputs_key, test_value)

                with pytest.raises(FilterInputValidationError):
                    test_filter.run()


@pytest.mark.parametrize(
    "validation_metadata",
    [
        METDATA_VALIDATION_DICT_STRUCT,
        METDATA_VALIDATION_DICT_ASL,
        METDATA_VALIDATION_DICT_GROUND_TRUTH,
    ],
)
def test_bids_output_filter_validate_metadata(validation_metadata: dict):
    """ Check a FilterInputValidationError is raised when the inputs 'image' metadata
    are incorrect or missing """

    with TemporaryDirectory() as temp_dir:
        test_filter = BidsOutputFilter()
        test_data = deepcopy(validation_metadata)
        passing_inputs = {}

        # create inputs dictionary of passing data
        for data_key in INPUT_VALIDATION_DICT:
            passing_inputs[data_key] = deepcopy(INPUT_VALIDATION_DICT[data_key][1])
        passing_inputs["output_directory"] = temp_dir
        # add passing metdata to the input image
        for metadata_key in test_data:
            passing_inputs["image"].metadata[metadata_key] = test_data[metadata_key][1]

        test_filter.add_inputs(passing_inputs)

        # should pass
        test_filter.run()

    for metadata_key in validation_metadata:
        with TemporaryDirectory() as temp_dir:
            passing_inputs["output_directory"] = temp_dir
            passing_inputs["image"].metadata = {}

            test_filter = BidsOutputFilter()
            is_optional: bool = test_data[metadata_key][0]

            # remove key
            test_data.pop(metadata_key)
            # add remaining
            for data_key in test_data:
                passing_inputs["image"].metadata[data_key] = test_data[data_key][1]

            test_filter.add_inputs(passing_inputs)
            # optional inputs should run without issue
            if is_optional:
                test_filter.run()
            else:
                with pytest.raises(FilterInputValidationError):
                    test_filter.run()

            # Try data that should fail
            for test_value in validation_metadata[metadata_key][2:]:
                test_filter = BidsOutputFilter()
                passing_inputs["image"].metadata[metadata_key] = test_value
                test_filter.add_inputs(passing_inputs)

                with pytest.raises(FilterInputValidationError):
                    test_filter.run()


@pytest.fixture(name="structural_input")
def structural_input_fixture() -> (NiftiImageContainer, dict):
    image = deepcopy(TEST_NIFTI_CON_ONES)
    image.metadata = {
        "echo_time": 0.01,
        "repetition_time": 0.3,
        "excitation_flip_angle": 30,
        "mr_acq_type": "3D",
        "acq_contrast": "ge",
        "series_type": "structural",
        "series_number": 1,
        "series_description": "test structural series",
        "modality": "T1w",
        "voxel_size": [1.0, 1.0, 1.0],
        "magnetic_field_strength": 3,
    }

    d = {
        "EchoTime": 0.01,
        "RepetitionTime": 0.3,
        "FlipAngle": 30,
        "MrAcquisitionType": "3D",
        "ScanningSequence": "GR",
        "SeriesNumber": 1,
        "SeriesDescription": "test structural series",
        "DROSoftware": "ASLDRO",
        "DROSoftwareVersion": asldro_version,
        "DROSoftwareUrl": [
            "code: https://github.com/gold-standard-phantoms/asldro",
            "pypi: https://pypi.org/project/asldro/",
            "docs: https://asldro.readthedocs.io/",
        ],
        "AcquisitionVoxelSize": [1.0, 1.0, 1.0],
        "ComplexImageComponent": "MAGNITUDE",
        "ImageType": ["ORIGINAL", "PRIMARY", "T1W", "NONE",],
        "MagneticFieldStrength": 3,
    }
    return (image, d)


def test_bids_output_filter_mock_data_structural(structural_input):
    """ Tests the BidsOutputFilter with some mock data """
    with TemporaryDirectory() as temp_dir:
        image = structural_input[0]
        d = structural_input[1]
        bids_output_filter = BidsOutputFilter()
        bids_output_filter.add_input("image", image)
        bids_output_filter.add_input("output_directory", temp_dir)
        bids_output_filter.add_input("filename_prefix", "prefix")
        bids_output_filter.run()

        # remove AcquisitionDateTime entry as this can't be compared here
        acq_date_time = bids_output_filter.outputs["sidecar"].pop("AcquisitionDateTime")
        assert bids_output_filter.outputs["sidecar"] == d
        bids_output_filter.outputs["sidecar"]["AcquisitionDateTime"] = acq_date_time

        # Check filenames
        assert bids_output_filter.outputs["filename"][0] == os.path.join(
            temp_dir, "anat", "prefix_001_T1w.nii.gz"
        )
        assert bids_output_filter.outputs["filename"][1] == os.path.join(
            temp_dir, "anat", "prefix_001_T1w.json"
        )

        # load in the files and check against what they should be
        loaded_nifti = nib.load(bids_output_filter.outputs["filename"][0])
        numpy.testing.assert_array_equal(
            loaded_nifti.dataobj, TEST_NIFTI_CON_ONES.image
        )
        assert loaded_nifti.header == TEST_NIFTI_CON_ONES.header

        with open(bids_output_filter.outputs["filename"][1], "r") as json_file:
            loaded_json = json.load(json_file)
        assert loaded_json == bids_output_filter.outputs["sidecar"]


@pytest.fixture(name="asl_input")
def asl_input_fixture() -> (NiftiImageContainer, dict):
    """creates test data for testing BIDS output of an ASL image"""
    image = deepcopy(TEST_NIFTI_CON_ONES)
    image.metadata = {
        "echo_time": 0.01,
        "repetition_time": 0.3,
        "excitation_flip_angle": 30,
        "mr_acq_type": "3D",
        "acq_contrast": "ge",
        "series_type": "asl",
        "series_number": 10,
        "series_description": "test asl series",
        "asl_context": "m0scan m0scan control label control label control label".split(),
        "label_type": "pcasl",
        "label_duration": 1.8,
        "post_label_delay": 1.8,
        "label_efficiency": 0.85,
        "lambda_blood_brain": 0.90,
        "t1_arterial_blood": 1.65,
        "image_flavour": "PERFUSION",
        "voxel_size": [1.0, 1.0, 1.0],
    }

    d = {
        "EchoTime": 0.01,
        "RepetitionTime": 0.3,
        "FlipAngle": 30,
        "MrAcquisitionType": "3D",
        "ScanningSequence": "GR",
        "SeriesNumber": 10,
        "SeriesDescription": "test asl series",
        "DROSoftware": "ASLDRO",
        "DROSoftwareVersion": asldro_version,
        "DROSoftwareUrl": [
            "code: https://github.com/gold-standard-phantoms/asldro",
            "pypi: https://pypi.org/project/asldro/",
            "docs: https://asldro.readthedocs.io/",
        ],
        "LabelingType": "PCASL",
        "LabelingDuration": 1.8,
        "PostLabelingDelay": 1.8,
        "LabelingEfficiency": 0.85,
        "AcquisitionVoxelSize": [1.0, 1.0, 1.0],
        "M0": True,
        "ComplexImageComponent": "MAGNITUDE",
        "ImageType": ["ORIGINAL", "PRIMARY", "PERFUSION", "NONE",],
    }
    return (image, d)


def test_bids_output_filter_mock_data_asl(asl_input):
    """ Tests the BidsOutputFilter with some mock data """
    with TemporaryDirectory() as temp_dir:
        image = asl_input[0]
        d = asl_input[1]

        bids_output_filter = BidsOutputFilter()
        bids_output_filter.add_input("image", image)
        bids_output_filter.add_input("output_directory", temp_dir)
        bids_output_filter.add_input("filename_prefix", "prefix")
        bids_output_filter.run()

        # remove AcquisitionDateTime entry as this can't be compared here
        acq_date_time = bids_output_filter.outputs["sidecar"].pop("AcquisitionDateTime")
        assert bids_output_filter.outputs["sidecar"] == d
        bids_output_filter.outputs["sidecar"]["AcquisitionDateTime"] = acq_date_time

        # Check filenames
        assert bids_output_filter.outputs["filename"][0] == os.path.join(
            temp_dir, "asl", "prefix_010_asl.nii.gz"
        )
        assert bids_output_filter.outputs["filename"][1] == os.path.join(
            temp_dir, "asl", "prefix_010_asl.json"
        )

        assert bids_output_filter.outputs["filename"][2] == os.path.join(
            temp_dir, "asl", "prefix_010_aslcontext.tsv"
        )

        # load in the files and check against what they should be
        loaded_nifti = nib.load(bids_output_filter.outputs["filename"][0])
        numpy.testing.assert_array_equal(
            loaded_nifti.dataobj, TEST_NIFTI_CON_ONES.image
        )
        assert loaded_nifti.header == TEST_NIFTI_CON_ONES.header

        with open(bids_output_filter.outputs["filename"][1], "r") as json_file:
            loaded_json = json.load(json_file)
        assert loaded_json == bids_output_filter.outputs["sidecar"]

        with open(bids_output_filter.outputs["filename"][2], "r") as tsv_file:
            loaded_tsv = tsv_file.readlines()
            tsv_file.close()
        assert loaded_tsv == [
            "volume_type\n",
            "m0scan\n",
            "m0scan\n",
            "control\n",
            "label\n",
            "control\n",
            "label\n",
            "control\n",
            "label",
        ]


def test_bids_output_filter_m0_float(asl_input):
    """Tests the BidsOutputFilter with mock ASL data where m0 is a float"""
    with TemporaryDirectory() as temp_dir:
        image = asl_input[0]
        d = asl_input[1]
        # first the ASL image, just control and label, with a float M0
        image.metadata[
            "asl_context"
        ] = "control label control label control label".split()
        image.metadata["m0"] = 100.0

        bids_output_filter = BidsOutputFilter()
        bids_output_filter.add_input("image", image)
        bids_output_filter.add_input("output_directory", temp_dir)
        bids_output_filter.add_input("filename_prefix", "prefix")
        bids_output_filter.run()

        d["M0"] = 100.0
        # remove AcquisitionDateTime entry as this can't be compared here
        acq_date_time = bids_output_filter.outputs["sidecar"].pop("AcquisitionDateTime")
        assert bids_output_filter.outputs["sidecar"] == d


def test_bids_output_filter_m0scan(structural_input):
    """Tests the BidsOutputFilter with mock ASL data where there is a separate m0 scan"""
    with TemporaryDirectory() as temp_dir:
        image = structural_input[0]
        d = structural_input[1]
        image.metadata["asl_context"] = "m0scan"
        image.metadata["series_type"] = "asl"

        bids_output_filter = BidsOutputFilter()
        bids_output_filter.add_input("image", image)
        bids_output_filter.add_input("output_directory", temp_dir)
        bids_output_filter.add_input("filename_prefix", "prefix")
        bids_output_filter.run()

        d["ImageType"] = ["ORIGINAL", "PRIMARY", "PROTON_DENSITY", "NONE"]
        # remove AcquisitionDateTime entry as this can't be compared here
        acq_date_time = bids_output_filter.outputs["sidecar"].pop("AcquisitionDateTime")
        assert bids_output_filter.outputs["sidecar"] == d
        bids_output_filter.outputs["sidecar"]["AcquisitionDateTime"] = acq_date_time

        # Check filenames
        assert bids_output_filter.outputs["filename"][0] == os.path.join(
            temp_dir, "asl", "prefix_001_m0scan.nii.gz"
        )
        assert bids_output_filter.outputs["filename"][1] == os.path.join(
            temp_dir, "asl", "prefix_001_m0scan.json"
        )

        # load in the files and check against what they should be
        loaded_nifti = nib.load(bids_output_filter.outputs["filename"][0])
        numpy.testing.assert_array_equal(
            loaded_nifti.dataobj, TEST_NIFTI_CON_ONES.image
        )
        assert loaded_nifti.header == TEST_NIFTI_CON_ONES.header

        with open(bids_output_filter.outputs["filename"][1], "r") as json_file:
            loaded_json = json.load(json_file)
        assert loaded_json == bids_output_filter.outputs["sidecar"]


def test_bids_output_filter_mock_data_ground_truth():
    """ Tests the BidsOutputFilter with some mock data """
    with TemporaryDirectory() as temp_dir:
        image = deepcopy(TEST_NIFTI_CON_ONES)
        image.metadata = {
            "series_type": "ground_truth",
            "series_number": 110,
            "series_description": "test ground truth series",
            "quantity": "t1",
            "units": "s",
            "voxel_size": [1.0, 1.0, 1.0],
        }
        bids_output_filter = BidsOutputFilter()
        bids_output_filter.add_input("image", image)
        bids_output_filter.add_input("output_directory", temp_dir)
        bids_output_filter.add_input("filename_prefix", "prefix")
        bids_output_filter.run()

        d = {
            "SeriesNumber": 110,
            "SeriesDescription": "test ground truth series",
            "DROSoftware": "ASLDRO",
            "DROSoftwareVersion": asldro_version,
            "DROSoftwareUrl": [
                "code: https://github.com/gold-standard-phantoms/asldro",
                "pypi: https://pypi.org/project/asldro/",
                "docs: https://asldro.readthedocs.io/",
            ],
            "AcquisitionVoxelSize": [1.0, 1.0, 1.0],
            "Units": "s",
            "Quantity": "t1",
            "ComplexImageComponent": "MAGNITUDE",
            "ImageType": ["ORIGINAL", "PRIMARY", "T1", "NONE",],
        }

        # remove AcquisitionDateTime entry as this can't be compared here
        acq_date_time = bids_output_filter.outputs["sidecar"].pop("AcquisitionDateTime")
        assert bids_output_filter.outputs["sidecar"] == d
        bids_output_filter.outputs["sidecar"]["AcquisitionDateTime"] = acq_date_time

        # Check filenames
        assert bids_output_filter.outputs["filename"][0] == os.path.join(
            temp_dir, "ground_truth", "prefix_110_ground_truth_t1.nii.gz"
        )
        assert bids_output_filter.outputs["filename"][1] == os.path.join(
            temp_dir, "ground_truth", "prefix_110_ground_truth_t1.json"
        )

        # load in the files and check against what they should be
        loaded_nifti = nib.load(bids_output_filter.outputs["filename"][0])
        numpy.testing.assert_array_equal(
            loaded_nifti.dataobj, TEST_NIFTI_CON_ONES.image
        )
        assert loaded_nifti.header == TEST_NIFTI_CON_ONES.header

        with open(bids_output_filter.outputs["filename"][1], "r") as json_file:
            loaded_json = json.load(json_file)
        assert loaded_json == bids_output_filter.outputs["sidecar"]


def test_bids_output_filter_mock_data_ground_truth_seg_label():
    """ Tests the BidsOutputFilter with some mock data """
    with TemporaryDirectory() as temp_dir:
        image = deepcopy(TEST_NIFTI_CON_ONES)
        image.metadata = {
            "series_type": "ground_truth",
            "series_number": 110,
            "series_description": "test ground truth series",
            "quantity": "seg_label",
            "segmentation": {
                "background": 0,
                "grey_matter": 1,
                "white_matter": 2,
                "csf": 3,
                "vascular": 4,
                "lesion": 5,
            },
            "units": "",
            "voxel_size": [1.0, 1.0, 1.0],
        }
        bids_output_filter = BidsOutputFilter()
        bids_output_filter.add_input("image", image)
        bids_output_filter.add_input("output_directory", temp_dir)
        bids_output_filter.add_input("filename_prefix", "prefix")
        bids_output_filter.run()

        d = {
            "SeriesNumber": 110,
            "SeriesDescription": "test ground truth series",
            "DROSoftware": "ASLDRO",
            "DROSoftwareVersion": asldro_version,
            "DROSoftwareUrl": [
                "code: https://github.com/gold-standard-phantoms/asldro",
                "pypi: https://pypi.org/project/asldro/",
                "docs: https://asldro.readthedocs.io/",
            ],
            "AcquisitionVoxelSize": [1.0, 1.0, 1.0],
            "Units": "",
            "Quantity": "seg_label",
            "LabelMap": {"BG": 0, "GM": 1, "WM": 2, "CSF": 3, "VS": 4, "L": 5,},
            "ComplexImageComponent": "MAGNITUDE",
            "ImageType": ["ORIGINAL", "PRIMARY", "SEG_LABEL", "NONE",],
        }

        # remove AcquisitionDateTime entry as this can't be compared here
        acq_date_time = bids_output_filter.outputs["sidecar"].pop("AcquisitionDateTime")
        assert bids_output_filter.outputs["sidecar"] == d
        bids_output_filter.outputs["sidecar"]["AcquisitionDateTime"] = acq_date_time

        # Check filenames
        assert bids_output_filter.outputs["filename"][0] == os.path.join(
            temp_dir, "ground_truth", "prefix_110_ground_truth_seg_label.nii.gz"
        )
        assert bids_output_filter.outputs["filename"][1] == os.path.join(
            temp_dir, "ground_truth", "prefix_110_ground_truth_seg_label.json"
        )

        # load in the files and check against what they should be
        loaded_nifti = nib.load(bids_output_filter.outputs["filename"][0])
        numpy.testing.assert_array_equal(
            loaded_nifti.dataobj, TEST_NIFTI_CON_ONES.image
        )
        assert loaded_nifti.header == TEST_NIFTI_CON_ONES.header

        with open(bids_output_filter.outputs["filename"][1], "r") as json_file:
            loaded_json = json.load(json_file)
        assert loaded_json == bids_output_filter.outputs["sidecar"]


def test_bids_output_filter_acquisition_date_time():
    """Mocks a call to datetime.datetime.now to test that the AcquisitionDateTime field of the
    output sidecar from BidsOutputFilter"""
    datetime_mock = Mock(wraps=datetime.datetime)
    test_datetime = datetime.datetime(2020, 10, 11, 17, 1, 32)
    with patch("asldro.filters.bids_output_filter.datetime", new=datetime_mock):
        datetime_mock.now.return_value = test_datetime
        with TemporaryDirectory() as temp_dir:

            bids_output_filter = BidsOutputFilter()
            bids_output_filter.add_input("image", TEST_NIFTI_CON_ONES)
            bids_output_filter.add_input("output_directory", temp_dir)
            bids_output_filter.add_input("filename_prefix", "prefix")
            bids_output_filter.run()

        assert (
            bids_output_filter.outputs["sidecar"]["AcquisitionDateTime"]
            == "2020-10-11T17:01:32.000000"
        )


def test_bids_output_filter_determine_asl_modality_label():
    """tests the static method determine_asl_modality_label()"""
    assert BidsOutputFilter.determine_asl_modality_label("m0scan") == "m0scan"
    assert BidsOutputFilter.determine_asl_modality_label(["m0scan"]) == "m0scan"
    assert (
        BidsOutputFilter.determine_asl_modality_label(["m0scan", "m0scan"]) == "m0scan"
    )
    assert BidsOutputFilter.determine_asl_modality_label(["m0scan", "control"]) == "asl"
    assert BidsOutputFilter.determine_asl_modality_label("control") == "asl"
    assert BidsOutputFilter.determine_asl_modality_label(["control"]) == "asl"
    assert BidsOutputFilter.determine_asl_modality_label(["m0scan", "control"]) == "asl"
    assert BidsOutputFilter.determine_asl_modality_label("str") == "asl"


def test_bids_output_filter_complex_image_component():
    """tests that the field ComplexImageComponent is correctly set"""
    with TemporaryDirectory() as temp_dir:
        image = deepcopy(TEST_NIFTI_CON_ONES)
        image.image_type = "REAL_IMAGE_TYPE"
        bids_output_filter = BidsOutputFilter()
        bids_output_filter.add_input("image", image)
        bids_output_filter.add_input("output_directory", temp_dir)
        bids_output_filter.run()
        assert bids_output_filter.outputs["sidecar"]["ComplexImageComponent"] == "REAL"

    with TemporaryDirectory() as temp_dir:
        image = deepcopy(TEST_NIFTI_CON_ONES)
        image.image_type = "IMAGINARY_IMAGE_TYPE"
        bids_output_filter = BidsOutputFilter()
        bids_output_filter.add_input("image", image)
        bids_output_filter.add_input("output_directory", temp_dir)
        bids_output_filter.run()
        assert (
            bids_output_filter.outputs["sidecar"]["ComplexImageComponent"]
            == "IMAGINARY"
        )

    with TemporaryDirectory() as temp_dir:
        image = deepcopy(TEST_NIFTI_CON_ONES)
        image.image_type = "PHASE_IMAGE_TYPE"
        bids_output_filter = BidsOutputFilter()
        bids_output_filter.add_input("image", image)
        bids_output_filter.add_input("output_directory", temp_dir)
        bids_output_filter.run()
        assert bids_output_filter.outputs["sidecar"]["ComplexImageComponent"] == "PHASE"

    with TemporaryDirectory() as temp_dir:
        image = NiftiImageContainer(
            nifti_img=nib.Nifti2Image(
                np.random.normal(100, 10, TEST_VOLUME_DIMENSIONS),
                affine=np.array(
                    ((1, 0, 0, -16), (0, 1, 0, -16), (0, 0, 1, -16), (0, 0, 0, 1),)
                ),
            ),
            metadata={
                "series_number": 1,
                "series_type": "structural",
                "modality": "T1w",
            },
        )
        image.image_type = "COMPLEX_IMAGE_TYPE"
        bids_output_filter = BidsOutputFilter()
        bids_output_filter.add_input("image", image)
        bids_output_filter.add_input("output_directory", temp_dir)
        bids_output_filter.run()
        assert (
            bids_output_filter.outputs["sidecar"]["ComplexImageComponent"] == "COMPLEX"
        )


def test_bids_output_filter_directory_tests():
    """Checks that the BidsOutputFilter can correctly handle situations where
    the output directories either exist or don't exist"""

    # sub-directory already exists
    with TemporaryDirectory() as temp_dir:
        bids_output_filter = BidsOutputFilter()
        bids_output_filter.add_input("image", TEST_NIFTI_CON_ONES)
        bids_output_filter.add_input("output_directory", temp_dir)

        os.makedirs(os.path.join(temp_dir, "anat"))

        bids_output_filter.run()
