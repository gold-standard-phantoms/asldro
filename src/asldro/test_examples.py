""" Example tests - only run when --runslow is passed to pytest """

import pytest
import numpy as np
import numpy.testing
from asldro.examples import run_full_pipeline
from asldro.validators.user_parameter_input import get_example_input_params
from asldro.containers.image import NiftiImageContainer


@pytest.mark.slow
def test_run_default_pipeline():
    """ Runs the full ASL DRO pipeline """
    droout = run_full_pipeline()

    # check the segmentation_mask resampled ground truth
    seg_label_index = [
        idx
        for idx, im in enumerate(droout["asldro_output"])
        if im.metadata.get("quantity") == "seg_label"
    ]
    gt_seg_label: NiftiImageContainer = droout["asldro_output"][seg_label_index[0]]

    # interpolation is nearest for the default so no new values should be created, check the
    # unique values against the original ground truth
    numpy.testing.assert_array_equal(
        np.unique(gt_seg_label.image), np.unique(droout["hrgt"]["seg_label"].image)
    )


@pytest.mark.slow
def test_run_extended_pipeline():
    """Runs the full ASL DRO pipeline with modified input parameters"""
    input_params = get_example_input_params()

    asl_params = input_params["image_series"][0]["series_parameters"]

    asl_params["asl_context"] = "m0scan control label control label control label"
    timeseries_length = len(asl_params["asl_context"].split())
    asl_params["echo_time"] = [0.01] * timeseries_length
    asl_params["repetition_time"] = [10.0] + [5.0] * (timeseries_length - 1)
    asl_params["rot_x"] = [0.0] * timeseries_length
    asl_params["rot_y"] = [0.0] * timeseries_length
    asl_params["rot_z"] = [0.0] * timeseries_length
    asl_params["transl_x"] = [0.0] * timeseries_length
    asl_params["transl_y"] = [0.0] * timeseries_length
    asl_params["transl_z"] = [0.0] * timeseries_length

    # remove the ground truth and structural image series
    input_params["image_series"] = [
        x for x in input_params["image_series"] if x["series_type"] in ["asl"]
    ]
    run_full_pipeline(input_params=input_params)


@pytest.mark.slow
def test_run_asl_pipeline_multiphase():
    """Runs the full ASL DRO pipeline for muliphase ASL 
    with the SNR=zero to check this works"""
    input_params = get_example_input_params()
    # remove the ground truth and structural image series
    input_params["image_series"] = [
        x for x in input_params["image_series"] if x["series_type"] in ["asl"]
    ]
    input_params["image_series"][0]["series_parameters"]["desired_snr"] = 0.0
    input_params["image_series"][0]["series_parameters"]["signal_time"] = [
        1.0,
        1.25,
        1.5,
    ]

    run_full_pipeline(input_params=input_params)


@pytest.mark.slow
def test_run_asl_pipeline_qasper_hrgt():
    """Runs the ASL pipeline using the qasper ground truth"""
    input_params = get_example_input_params()

    input_params["global_configuration"]["ground_truth"] = "qasper_3t"

    input_params["image_series"][0]["series_parameters"]["desired_snr"] = 0.0
    # remove the ground truth and structural image series
    input_params["image_series"] = [
        x for x in input_params["image_series"] if x["series_type"] in ["asl"]
    ]

    run_full_pipeline(input_params=input_params)

