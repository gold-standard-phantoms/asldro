""" Example tests - only run when --runslow is passed to pytest """

import pytest
import numpy as np
import numpy.testing
from asldro.examples import run_full_pipeline
from asldro.filters.load_asl_bids_filter import LoadAslBidsFilter
from asldro.validators.user_parameter_input import (
    get_example_input_params,
    ARRAY_PARAMS,
)
from asldro.containers.image import NiftiImageContainer
from asldro.filters.asl_quantification_filter import AslQuantificationFilter


@pytest.mark.slow
def test_run_default_pipeline():
    """Runs the full ASL DRO pipeline"""
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
    # remove the array parameters so they are autogenerates
    [asl_params.pop(param) for param in ARRAY_PARAMS]

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


@pytest.mark.slow
def test_run_asl_pipeline_whitepaper():
    """Runs the ASL pipeline with gkm_model set to "whitepaper"
    Checks the results using the AslQuantificationFilter"""

    input_params = get_example_input_params()

    input_params["image_series"][0]["series_parameters"]["gkm_model"] = "whitepaper"
    input_params["image_series"][0]["series_parameters"]["desired_snr"] = 0.0
    # set the TR of the M0 to very long so there's no real effect (important for
    # comparing values)
    input_params["image_series"][0]["series_parameters"]["repetition_time"] = [
        1.0e6,
        5.0,
        5.0,
    ]
    input_params["image_series"][0]["series_parameters"]["acq_matrix"] = input_params[
        "image_series"
    ][2]["series_parameters"]["acq_matrix"]
    # remove the ground truth and structural image series

    input_params["image_series"] = [
        x for x in input_params["image_series"] if x["series_type"] in ["asl"]
    ]

    out = run_full_pipeline(input_params=input_params)

    asl_source: NiftiImageContainer = out["asldro_output"][0]
    assert asl_source.metadata["gkm_model"] == "whitepaper"
    asl_data = {}  # empty dictionary for asl data
    asl_context = asl_source.metadata["asl_context"]
    for key in ["control", "label", "m0scan"]:
        volume_indices = [i for (i, val) in enumerate(asl_context) if val == key]
        if volume_indices is not None:
            asl_data[key] = asl_source.clone()
            asl_data[key].image = np.squeeze(asl_source.image[:, :, :, volume_indices])
            asl_data[key].metadata["asl_context"] = key * len(volume_indices)

        # adjust metadata lists to correspond to one value per volume
        new_metadata = {
            metadata_key: [metadata_val[i] for i in volume_indices]
            for (metadata_key, metadata_val) in asl_data[key].metadata.items()
            if (
                (metadata_key not in ["voxel_size"])
                & (
                    (len(metadata_val) == len(asl_context))
                    if isinstance(metadata_val, list)
                    else False
                )
            )
        }

        asl_data[key].metadata = {**asl_data[key].metadata, **new_metadata}
    asl_quantification_filter = AslQuantificationFilter()
    asl_quantification_filter.add_inputs(
        {
            key: value
            for (key, value) in asl_data["control"].metadata.items()
            if key
            in [
                AslQuantificationFilter.KEY_LABEL_TYPE,
                AslQuantificationFilter.KEY_LABEL_DURATION,
                AslQuantificationFilter.KEY_POST_LABEL_DELAY,
                AslQuantificationFilter.KEY_LABEL_EFFICIENCY,
                AslQuantificationFilter.KEY_T1_ARTERIAL_BLOOD,
                AslQuantificationFilter.KEY_LAMBDA_BLOOD_BRAIN,
            ]
        }
    )
    asl_quantification_filter.add_inputs(
        asl_data, io_map={"m0scan": "m0", "control": "control", "label": "label"}
    )
    asl_quantification_filter.add_input("model", "whitepaper")

    asl_quantification_filter.run()

    # compare the quantified perfusion_rate with the ground truth to 12 d.p.
    np.testing.assert_array_almost_equal(
        asl_quantification_filter.outputs["perfusion_rate"].image,
        out["hrgt"]["perfusion_rate"].image,
        12,
    )
