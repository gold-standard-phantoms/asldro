"""Tests for asl_quantification_filter.py"""

from copy import deepcopy
import pytest

import numpy as np
from numpy.random import default_rng
import numpy.testing
import nibabel as nib

from asldro.containers.image import NiftiImageContainer
from asldro.utils.filter_validation import validate_filter_inputs

from asldro.filters.asl_quantification_filter import AslQuantificationFilter
from asldro.filters.gkm_filter import GkmFilter

TEST_DIM_3D = (2, 2, 2)
TEST_DIM_4D = (2, 2, 2, 10)
TEST_NIFTI_ONES = nib.Nifti2Image(
    np.ones(TEST_DIM_3D),
    affine=np.array(((1, 0, 0, -16), (0, 1, 0, -16), (0, 0, 1, -16), (0, 0, 0, 1),)),
)
TEST_NIFTI_CON_ONES = NiftiImageContainer(nifti_img=TEST_NIFTI_ONES,)

TEST_METADATA = {
    "RepetitionTime": 5.0,
    "RepetitionTimePreparation": 5.0,
    "EchoTime": 0.01,
    "FlipAngle": 90,
    "M0Type": "Included",
    "ComplexImageComponent": "REAL",
    "ImageType": ["ORIGINAL", "PRIMARY", "PERFUSION", "NONE",],
    "BackgroundSuppression": False,
    "VascularCrushing": False,
    "LabelingDuration": 1.8,
    "PostLabelingDelay": 1.8,
    "LabelingEfficiency": 0.85,
}


@pytest.fixture(name="test_data_wp")
def test_data_wp_fixture() -> dict:
    label_3d = NiftiImageContainer(
        nib.Nifti2Image(0.99 * np.ones(TEST_DIM_3D), affine=np.eye(4)),
        metadata=TEST_METADATA,
    )
    control_3d = NiftiImageContainer(
        nib.Nifti2Image(np.ones(TEST_DIM_3D), affine=np.eye(4)), metadata=TEST_METADATA,
    )
    label_4d = NiftiImageContainer(
        nib.Nifti2Image(0.99 * np.ones(TEST_DIM_4D), affine=np.eye(4)),
        metadata=TEST_METADATA,
    )
    control_4d = NiftiImageContainer(
        nib.Nifti2Image(np.ones(TEST_DIM_4D), affine=np.eye(4)), metadata=TEST_METADATA,
    )
    m0 = NiftiImageContainer(nib.Nifti2Image(np.ones(TEST_DIM_3D), affine=np.eye(4)))
    return {
        "control_3d": control_3d,
        "label_3d": label_3d,
        "control_4d": control_4d,
        "label_4d": label_4d,
        "m0": m0,
        "lambda_blood_brain": 0.9,
        "label_duration": 1.8,
        "post_label_delay": 1.8,
        "label_efficiency": 0.85,
        "model": "whitepaper",
        "t1_arterial_blood": 1.65,
    }


@pytest.fixture(name="multiphase_data")
def test_fixture_multiphase() -> dict:
    """Generates test data using the general kinetic model with ground truths
    for perfusion rate and transit time"""
    n_vols = 10
    rng = default_rng(1234)
    label_duration = 1.8
    signal_times = rng.uniform(label_duration, label_duration + 3, n_vols)
    post_label_delay = signal_times - label_duration
    perfusion_rate = np.linspace(10, 80, num=8).reshape(TEST_DIM_3D)
    transit_time = np.linspace(0.5, 1.5, num=8).reshape(TEST_DIM_3D)
    label_efficiency = 0.85
    t1_arterial_blood = 1.65
    t1_tissue = 1.3 * np.ones(TEST_DIM_3D)
    m0_tissue = 73.7 * np.ones(TEST_DIM_3D)
    lambda_blood_brain = 0.9 * np.ones(TEST_DIM_3D)

    delta_m = {
        label_type: np.stack(
            [
                GkmFilter.calculate_delta_m_gkm(
                    perfusion_rate,
                    transit_time,
                    m0_tissue,
                    label_duration,
                    signal_time,
                    label_efficiency,
                    lambda_blood_brain,
                    t1_arterial_blood,
                    t1_tissue,
                    label_type,
                )
                for signal_time in signal_times
            ],
            axis=3,
        )
        for label_type in ["pasl", "casl"]
    }
    return {
        "lambda_blood_brain": lambda_blood_brain,
        "label_duration": label_duration,
        "post_label_delay": post_label_delay.tolist(),
        "label_efficiency": label_efficiency,
        "multiphase_index": list(range(0, signal_times.size)),
        "t1_arterial_blood": t1_arterial_blood,
        "perfusion_rate": perfusion_rate,
        "transit_time": transit_time,
        "t1_tissue": t1_tissue,
        "m0_tissue": m0_tissue,
        "control": {
            label_type: np.stack([m0_tissue] * n_vols, 3)
            for label_type in ["pasl", "casl"]
        },
        "label": {
            label_type: np.stack([m0_tissue] * n_vols, 3) - delta_m[label_type]
            for label_type in ["pasl", "casl"]
        },
    }


@pytest.fixture(name="test_data_full")
def test_data_full_fixture(multiphase_data) -> dict:
    dim_4d_2 = (2, 2, 2, 9)
    metadata = deepcopy(TEST_METADATA)
    metadata["PostLabelingDelay"] = multiphase_data["post_label_delay"]
    metadata["MultiphaseIndex"] = multiphase_data["multiphase_index"]

    label = NiftiImageContainer(
        nib.Nifti2Image(multiphase_data["label"]["casl"], affine=np.eye(4)),
        metadata=metadata,
    )
    control = NiftiImageContainer(
        nib.Nifti2Image(multiphase_data["control"]["casl"], affine=np.eye(4)),
        metadata=metadata,
    )
    label_2 = NiftiImageContainer(
        nib.Nifti2Image(0.99 * np.ones(dim_4d_2), affine=np.eye(4)), metadata=metadata,
    )
    control_2 = NiftiImageContainer(
        nib.Nifti2Image(np.ones(dim_4d_2), affine=np.eye(4)), metadata=metadata
    )
    m0 = NiftiImageContainer(
        nib.Nifti2Image(multiphase_data["m0_tissue"], affine=np.eye(4))
    )
    return {
        **multiphase_data,
        **{
            "control": control,
            "label": label,
            "control_2": control_2,
            "label_2": label_2,
            "m0": m0,
            "t1_tissue": multiphase_data["t1_tissue"][0, 0, 0],
            "lambda_blood_brain": multiphase_data["lambda_blood_brain"][
                0, 0, 0
            ],  # only float required
            "label_type": "casl",
        },
    }


@pytest.fixture(name="validation_data")
def input_validation_dict_fixture(test_data_wp, test_data_full):
    """Returns a dictionary for validating the filter inputs"""

    # input validation dictionary, for each key the list provides:
    # [0] bool for Optional, [1] passes, [2:] fail
    return {
        "whitepaper": {
            "control": [False, test_data_wp["control_3d"], TEST_NIFTI_ONES, 1.0, "str"],
            "label": [False, test_data_wp["label_3d"], TEST_NIFTI_ONES, 1.0, "str"],
            "m0": [False, test_data_wp["m0"], TEST_NIFTI_ONES, 1.0, "str"],
            "label_type": [False, "casl", "CSL", 1.0, TEST_NIFTI_CON_ONES],
            "model": [False, "whitepaper", "whitpaper", 1.0, TEST_NIFTI_CON_ONES],
            "label_duration": [False, 1.8, -1.8, "str"],
            "post_label_delay": [False, 1.8, -1.8, "str"],
            "label_efficiency": [False, 0.85, 1.85, "str"],
            "lambda_blood_brain": [False, 0.9, 1.9, "str"],
            "t1_arterial_blood": [False, 1.65, -1.65, "str"],
        },
        "full": {
            "control": [
                False,
                test_data_full["control"],
                test_data_full["control_2"],
                TEST_NIFTI_ONES,
                1.0,
                "str",
            ],
            "label": [
                False,
                test_data_full["label"],
                test_data_full["label_2"],
                TEST_NIFTI_ONES,
                1.0,
                "str",
            ],
            "m0": [False, test_data_full["m0"], TEST_NIFTI_ONES, 1.0, "str"],
            "t1_tissue": [False, test_data_full["t1_tissue"], TEST_NIFTI_ONES, "str"],
            "label_type": [False, "casl", "CSL", 1.0, TEST_NIFTI_CON_ONES],
            "model": [False, "full", "fll", 1.0, TEST_NIFTI_CON_ONES],
            "label_duration": [False, 1.8, -1.8, "str"],
            "post_label_delay": [
                False,
                test_data_full["post_label_delay"],
                test_data_full["post_label_delay"][:-1],
                1.8,
                "str",
            ],
            "label_efficiency": [False, 0.85, 1.85, "str"],
            "lambda_blood_brain": [
                False,
                test_data_full["lambda_blood_brain"],
                1.9,
                "str",
            ],
            "t1_arterial_blood": [False, 1.65, -1.65, "str"],
            "multiphase_index": [
                False,
                test_data_full["multiphase_index"],
                test_data_full["multiphase_index"][:-1],
                1,
                "str",
            ],
        },
    }


CASL_VALIDATION_DATA = (
    (0.0, 1.0, 0.9, 1.8, 1.8, 0.85, 1.65, 0.000000000000),
    (0.0005, 1.0, 0.9, 1.8, 1.8, 0.85, 1.65, 4.314996006478),
    (0.001, 1.0, 0.9, 1.8, 1.8, 0.85, 1.65, 8.629992012956),
    (0.0015, 1.0, 0.9, 1.8, 1.8, 0.85, 1.65, 12.944988019434),
    (0.002, 1.0, 0.9, 1.8, 1.8, 0.85, 1.65, 17.259984025912),
    (0.0025, 1.0, 0.9, 1.8, 1.8, 0.85, 1.65, 21.574980032390),
    (0.003, 1.0, 0.9, 1.8, 1.8, 0.85, 1.65, 25.889976038868),
    (0.0035, 1.0, 0.9, 1.8, 1.8, 0.85, 1.65, 30.204972045346),
    (0.004, 1.0, 0.9, 1.8, 1.8, 0.85, 1.65, 34.519968051824),
    (0.0045, 1.0, 0.9, 1.8, 1.8, 0.85, 1.65, 38.834964058302),
    (0.005, 1.0, 0.9, 1.8, 1.8, 0.85, 1.65, 43.149960064780),
    (0.0055, 1.0, 0.9, 1.8, 1.8, 0.85, 1.65, 47.464956071258),
    (0.006, 1.0, 0.9, 1.8, 1.8, 0.85, 1.65, 51.779952077736),
    (0.0065, 1.0, 0.9, 1.8, 1.8, 0.85, 1.65, 56.094948084214),
    (0.007, 1.0, 0.9, 1.8, 1.8, 0.85, 1.65, 60.409944090692),
    (0.0075, 1.0, 0.9, 1.8, 1.8, 0.85, 1.65, 64.724940097170),
)

PASL_VALIDATION_DATA = (
    (0.0000, 1.00, 0.90, 0.80, 1.80, 0.98, 1.65, 0.000000000000),
    (0.0005, 1.00, 0.90, 0.80, 1.80, 0.98, 1.65, 5.126175896834),
    (0.0010, 1.00, 0.90, 0.80, 1.80, 0.98, 1.65, 10.252351793669),
    (0.0015, 1.00, 0.90, 0.80, 1.80, 0.98, 1.65, 15.378527690503),
    (0.0020, 1.00, 0.90, 0.80, 1.80, 0.98, 1.65, 20.504703587338),
    (0.0025, 1.00, 0.90, 0.80, 1.80, 0.98, 1.65, 25.630879484172),
    (0.0030, 1.00, 0.90, 0.80, 1.80, 0.98, 1.65, 30.757055381007),
    (0.0035, 1.00, 0.90, 0.80, 1.80, 0.98, 1.65, 35.883231277841),
    (0.0040, 1.00, 0.90, 0.80, 1.80, 0.98, 1.65, 41.009407174676),
    (0.0045, 1.00, 0.90, 0.80, 1.80, 0.98, 1.65, 46.135583071510),
    (0.0050, 1.00, 0.90, 0.80, 1.80, 0.98, 1.65, 51.261758968345),
    (0.0055, 1.00, 0.90, 0.80, 1.80, 0.98, 1.65, 56.387934865179),
    (0.0060, 1.00, 0.90, 0.80, 1.80, 0.98, 1.65, 61.514110762013),
    (0.0065, 1.00, 0.90, 0.80, 1.80, 0.98, 1.65, 66.640286658848),
    (0.0070, 1.00, 0.90, 0.80, 1.80, 0.98, 1.65, 71.766462555682),
    (0.0075, 1.00, 0.90, 0.80, 1.80, 0.98, 1.65, 76.892638452517),
)


def test_asl_quantification_filter_validate_inputs(validation_data):
    """Check a FilterInputValidationError is raised when the inputs to the
    AslQuantificationFilter are incorrect or missing"""
    validate_filter_inputs(AslQuantificationFilter, validation_data["whitepaper"])
    validate_filter_inputs(AslQuantificationFilter, validation_data["full"])


@pytest.mark.parametrize(
    "delta_m, m0, lambda_blood_brain, label_dur, pld, lab_eff, t1, expected",
    CASL_VALIDATION_DATA,
)
def test_asl_quantification_verify_casl_numeric(
    delta_m, m0, lambda_blood_brain, label_dur, pld, lab_eff, t1, expected
):
    """Verifies the numerical output of the asl_quant_wp_casl static method"""
    actual = (
        AslQuantificationFilter.asl_quant_wp_casl(
            m0, m0 - delta_m, m0, lambda_blood_brain, label_dur, pld, lab_eff, t1,
        ),
    )
    np.testing.assert_array_almost_equal(
        actual, expected,
    )


@pytest.mark.parametrize(
    "delta_m, m0, lambda_blood_brain, bol_dur, inv_time, lab_eff, t1, expected",
    PASL_VALIDATION_DATA,
)
def test_asl_quantification_filter_verify_pasl_numeric(
    delta_m, m0, lambda_blood_brain, bol_dur, inv_time, lab_eff, t1, expected
):
    """Verifies the numerical output of the asl_quant_wp_pasl static method"""
    actual = (
        AslQuantificationFilter.asl_quant_wp_pasl(
            m0, m0 - delta_m, m0, lambda_blood_brain, bol_dur, inv_time, lab_eff, t1,
        ),
    )
    np.testing.assert_array_almost_equal(
        actual, expected,
    )


def test_asl_quantification_filter_asl_quant_wp_casl():
    """Test that the static function asl_quant_wp_casl produces correct results"""

    control = np.ones(TEST_DIM_3D)
    label = (1 - 0.001) * np.ones(TEST_DIM_3D)
    m0 = np.ones(TEST_DIM_3D)
    lambda_blood_brain = 0.9
    label_duration = 1.8
    post_label_delay = 1.8
    label_efficiency = 0.85
    t1_arterial_blood = 1.65
    calc_cbf = AslQuantificationFilter.asl_quant_wp_casl(
        control,
        label,
        m0,
        lambda_blood_brain,
        label_duration,
        post_label_delay,
        label_efficiency,
        t1_arterial_blood,
    )
    numpy.testing.assert_array_equal(
        calc_cbf,
        np.divide(
            6000
            * lambda_blood_brain
            * (control - label)
            * np.exp(post_label_delay / t1_arterial_blood),
            2
            * label_efficiency
            * t1_arterial_blood
            * m0
            * (1 - np.exp(-label_duration / t1_arterial_blood)),
            out=np.zeros_like(m0),
            where=m0 != 0,
        ),
    )


def test_asl_quantification_filter_with_mock_data_casl(test_data_wp):
    """Tests the AslQuantificationFilter with some mock data for CASL"""

    input_params = {
        "control": test_data_wp["control_3d"],
        "label": test_data_wp["label_3d"],
        "m0": test_data_wp["m0"],
        "label_type": "casl",
        "model": "whitepaper",
        "lambda_blood_brain": 0.9,
        "label_duration": 1.8,
        "post_label_delay": 1.8,
        "label_efficiency": 0.85,
        "t1_arterial_blood": 1.65,
    }

    asl_quantification_filter = AslQuantificationFilter()
    asl_quantification_filter.add_inputs(input_params)
    asl_quantification_filter.run()

    numpy.testing.assert_array_equal(
        asl_quantification_filter.outputs["perfusion_rate"].image,
        AslQuantificationFilter.asl_quant_wp_casl(
            test_data_wp["control_3d"].image,
            test_data_wp["label_3d"].image,
            test_data_wp["m0"].image,
            input_params["lambda_blood_brain"],
            input_params["label_duration"],
            input_params["post_label_delay"],
            input_params["label_efficiency"],
            input_params["t1_arterial_blood"],
        ),
    )

    # check the image metadata
    assert asl_quantification_filter.outputs["perfusion_rate"].metadata == {
        "ComplexImageComponent": "REAL",
        "ImageType": ["DERIVED", "PRIMARY", "PERFUSION", "RCBF",],
        "BackgroundSuppression": False,
        "VascularCrushing": False,
        "LabelingDuration": 1.8,
        "PostLabelingDelay": 1.8,
        "LabelingEfficiency": 0.85,
        "asl_context": "cbf",
        "Units": "ml/100g/min",
        "EstimationAlgorithm": """Calculated using the single subtraction simplified model for
CBF quantification from the ASL White Paper:

Alsop et. al., Recommended implementation of arterial
spin-labeled perfusion MRI for clinical applications:
a consensus of the ISMRM perfusion study group and the
european consortium for ASL in dementia. Magnetic Resonance
in Medicine, 73(1):102–116, apr 2014. doi:10.1002/mrm.25197
""",
    }


def test_asl_quantification_filter_with_mock_timeseries(test_data_wp):
    """Tests the AslQuantificationFilter with some mock timeseries data"""

    input_params = {
        "control": test_data_wp["control_4d"],
        "label": test_data_wp["label_4d"],
        "m0": test_data_wp["m0"],
        "label_type": "casl",
        "model": "whitepaper",
        "lambda_blood_brain": 0.9,
        "label_duration": 1.8,
        "post_label_delay": 1.8,
        "label_efficiency": 0.85,
        "t1_arterial_blood": 1.65,
    }

    asl_quantification_filter = AslQuantificationFilter()
    asl_quantification_filter.add_inputs(input_params)
    asl_quantification_filter.run()

    numpy.testing.assert_array_equal(
        asl_quantification_filter.outputs["perfusion_rate"].image,
        AslQuantificationFilter.asl_quant_wp_casl(
            test_data_wp["control_4d"].image[:, :, :, 0],
            test_data_wp["label_4d"].image[:, :, :, 0],
            test_data_wp["m0"].image,
            input_params["lambda_blood_brain"],
            input_params["label_duration"],
            input_params["post_label_delay"],
            input_params["label_efficiency"],
            input_params["t1_arterial_blood"],
        ),
    )


def test_asl_quantification_filter_asl_quant_wp_pasl():
    """Test that the static function asl_quant_wp_pasl produces correct results"""

    control = np.ones(TEST_DIM_3D)
    label = (1 - 0.001) * np.ones(TEST_DIM_3D)
    m0 = np.ones(TEST_DIM_3D)
    lambda_blood_brain = 0.9
    bolus_duration = 0.8
    inversion_time = 1.8
    label_efficiency = 0.85
    t1_arterial_blood = 1.65
    calc_cbf = AslQuantificationFilter.asl_quant_wp_pasl(
        control,
        label,
        m0,
        lambda_blood_brain,
        bolus_duration,
        inversion_time,
        label_efficiency,
        t1_arterial_blood,
    )
    numpy.testing.assert_array_equal(
        calc_cbf,
        np.divide(
            6000
            * lambda_blood_brain
            * (control - label)
            * np.exp(inversion_time / t1_arterial_blood),
            2 * label_efficiency * bolus_duration * m0,
            out=np.zeros_like(m0),
            where=m0 != 0,
        ),
    )


def test_asl_quantification_filter_with_mock_data_pasl(test_data_wp):
    """Tests the AslQuantificationFilter with some mock data for PASL"""

    input_params = {
        "control": test_data_wp["control_3d"],
        "label": test_data_wp["label_3d"],
        "m0": test_data_wp["m0"],
        "label_type": "pasl",
        "model": "whitepaper",
        "lambda_blood_brain": 0.9,
        "label_duration": 0.8,
        "post_label_delay": 1.8,
        "label_efficiency": 0.85,
        "t1_arterial_blood": 1.65,
    }

    asl_quantification_filter = AslQuantificationFilter()
    asl_quantification_filter.add_inputs(input_params)
    asl_quantification_filter.run()

    numpy.testing.assert_array_equal(
        asl_quantification_filter.outputs["perfusion_rate"].image,
        AslQuantificationFilter.asl_quant_wp_pasl(
            test_data_wp["control_3d"].image,
            test_data_wp["label_3d"].image,
            test_data_wp["m0"].image,
            input_params["lambda_blood_brain"],
            input_params["label_duration"],
            input_params["post_label_delay"],
            input_params["label_efficiency"],
            input_params["t1_arterial_blood"],
        ),
    )

    # check the image metadata
    assert asl_quantification_filter.outputs["perfusion_rate"].metadata == {
        "ComplexImageComponent": "REAL",
        "ImageType": ["DERIVED", "PRIMARY", "PERFUSION", "RCBF",],
        "BackgroundSuppression": False,
        "VascularCrushing": False,
        "LabelingDuration": 1.8,
        "PostLabelingDelay": 1.8,
        "LabelingEfficiency": 0.85,
        "asl_context": "cbf",
        "Units": "ml/100g/min",
        "EstimationAlgorithm": """Calculated using the single subtraction simplified model for
CBF quantification from the ASL White Paper:

Alsop et. al., Recommended implementation of arterial
spin-labeled perfusion MRI for clinical applications:
a consensus of the ISMRM perfusion study group and the
european consortium for ASL in dementia. Magnetic Resonance
in Medicine, 73(1):102–116, apr 2014. doi:10.1002/mrm.25197
""",
    }


## Tests for when the AslQuantificationFilter uses the 'full' model.


def test_asl_quantification_filter_asl_quant_lsq_gkm(multiphase_data):
    """Numerically tests the least squares fitting to the general kinetic model
    for both the casl and pasl signal models."""
    for label_type in ["pasl", "casl"]:

        results = AslQuantificationFilter.asl_quant_lsq_gkm(
            multiphase_data["control"][label_type],
            multiphase_data["label"][label_type],
            multiphase_data["m0_tissue"],
            multiphase_data["lambda_blood_brain"],
            multiphase_data["label_duration"],
            multiphase_data["post_label_delay"],
            multiphase_data["label_efficiency"],
            multiphase_data["t1_arterial_blood"],
            multiphase_data["t1_tissue"],
            label_type,
        )
        numpy.testing.assert_array_almost_equal(
            results["perfusion_rate"], multiphase_data["perfusion_rate"]
        )
        numpy.testing.assert_array_almost_equal(
            results["transit_time"], multiphase_data["transit_time"]
        )


def test_asl_quantification_filter_full_mock_data(test_data_full, multiphase_data):
    """Tests the AslQuantificationFilter when using the full GKM for
    quantification with some mock data"""
    for label_type in ["casl"]:
        asl_quantification_filter = AslQuantificationFilter()
        asl_quantification_filter.add_inputs(test_data_full)
        asl_quantification_filter.add_input("model", "full")
        asl_quantification_filter.run()

        # compare the image with the ground truth
        numpy.testing.assert_array_almost_equal(
            asl_quantification_filter.outputs["perfusion_rate"].image,
            multiphase_data["perfusion_rate"],
        )
        numpy.testing.assert_array_almost_equal(
            asl_quantification_filter.outputs["transit_time"].image,
            multiphase_data["transit_time"],
        )

        # check the image metadata
        assert asl_quantification_filter.outputs["perfusion_rate"].metadata == {
            "ComplexImageComponent": "REAL",
            "ImageType": ["DERIVED", "PRIMARY", "PERFUSION", "RCBF",],
            "BackgroundSuppression": False,
            "VascularCrushing": False,
            "LabelingDuration": test_data_full["label_duration"],
            "PostLabelingDelay": test_data_full["post_label_delay"],
            "LabelingEfficiency": test_data_full["label_efficiency"],
            "asl_context": "cbf",
            "Units": "ml/100g/min",
            "EstimationAlgorithm": """Least Squares fit to the General Kinetic Model for
Arterial Spin Labelling:

Buxton et. al., A general
kinetic model for quantitative perfusion imaging with arterial
spin labeling. Magnetic Resonance in Medicine, 40(3):383–396,
sep 1998. doi:10.1002/mrm.1910400308.""",
        }
