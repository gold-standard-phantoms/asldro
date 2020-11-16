""" Example tests - only run when --runslow is passed to pytest """

import pytest
from asldro.examples import run_full_pipeline
from asldro.validators.user_parameter_input import get_example_input_params


@pytest.mark.slow
def test_run_full_pipeline():
    """ Runs the full ASL DRO pipeline """
    run_full_pipeline()


@pytest.mark.slow
def test_run_full_pipeline_extended_params():
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

    run_full_pipeline(input_params=input_params)

