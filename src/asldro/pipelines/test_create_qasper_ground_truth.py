"""Test for create_qasper_ground_truth.py"""

import os
from tempfile import TemporaryDirectory

import pytest
import numpy as np
import numpy.testing
from asldro.examples import run_full_pipeline

from asldro.pipelines.create_qasper_ground_truth import generate_qasper
from asldro.validators.user_parameter_input import (
    get_example_input_params,
    ACQ_MATRIX,
    DESIRED_SNR,
    REPETITION_TIME,
    LABEL_DURATION,
    SIGNAL_TIME,
    OUTPUT_IMAGE_TYPE,
)
from asldro.containers.image import NiftiImageContainer


@pytest.mark.slow
def test_create_qasper():
    """Tests creating a QASPER ground truth, then runs it through
    the ASLDRO ASL pipeline, comparing the results against the defined
    bulk flow rate"""

    with TemporaryDirectory() as temp_dir:
        generate_qasper(temp_dir)
        qasper_flow_rate = 350  # bulk flow rate in ml/min
        # now use this qasper hrgt in the ASLDRO pipeline
        input_params = get_example_input_params()

        input_params["global_configuration"]["ground_truth"] = {
            "nii": os.path.join(temp_dir, "qasper_hrgt.nii.gz"),
            "json": os.path.join(temp_dir, "qasper_hrgt.json"),
        }
        input_params["global_configuration"]["ground_truth_modulate"] = {
            "perfusion_rate": {"scale": qasper_flow_rate},
        }
        # set the SNR to 0 -> no noise added
        input_params["image_series"][0]["series_parameters"][DESIRED_SNR] = 0.0
        input_params["image_series"][0]["series_parameters"][ACQ_MATRIX] = [
            260,
            260,
            130,
        ]
        # set TR to 1,000,000 s for the m0scan, this makes any TR effects negligible.
        input_params["image_series"][0]["series_parameters"][REPETITION_TIME] = [
            1.0e6,
            5.0,
            5.0,
        ]

        input_params["image_series"][0]["series_parameters"][LABEL_DURATION] = 0.5
        input_params["image_series"][0]["series_parameters"][SIGNAL_TIME] = 3.6
        input_params["image_series"][0]["series_parameters"][
            OUTPUT_IMAGE_TYPE
        ] = "complex"

        # remove the ground truth and structural image series
        input_params["image_series"] = [
            x for x in input_params["image_series"] if x["series_type"] in ["asl"]
        ]

        # run the ASLDRO pipeline
        output = run_full_pipeline(input_params=input_params)

    asl_source_container: NiftiImageContainer = output["asldro_output"][0]
    hrgt = output["hrgt"]

    m0_data = np.squeeze(asl_source_container.image[:, :, :, 0])
    control_data = np.squeeze(asl_source_container.image[:, :, :, 1])
    label_data = np.squeeze(asl_source_container.image[:, :, :, 2])
    diff_data = control_data - label_data

    # use the long format of the GKM for a single timepoint, using the input ground
    # truth to provide the actual values.
    t1: np.ndarray = hrgt["t1"].image
    f: np.ndarray = hrgt["perfusion_rate"].image
    bbpc: np.ndarray = hrgt["lambda_blood_brain"].image
    tt: np.ndarray = hrgt["transit_time"].image
    lab_dur = asl_source_container.metadata["label_duration"]
    pld = asl_source_container.metadata["post_label_delay"]
    lab_eff = asl_source_container.metadata["label_efficiency"]
    t1b = hrgt["t1_arterial_blood"]

    cbf = calc_cbf_gkm_casl(
        diff_data, m0_data, f, tt, t1, t1b, bbpc, lab_dur, pld, lab_eff
    )

    porous_mask = (
        hrgt["seg_label"].image == hrgt["seg_label"].metadata["segmentation"]["porous"]
    )
    inlet_mask = (
        hrgt["seg_label"].image == hrgt["seg_label"].metadata["segmentation"]["inlet"]
    )
    outlet_mask = (
        hrgt["seg_label"].image == hrgt["seg_label"].metadata["segmentation"]["outlet"]
    )

    tt_threshold_mask = tt < pld

    porous_ring_mask = porous_mask & tt_threshold_mask
    # Calculate the global flow rate within the porous material in the voxels where
    # the bolus has been fully delivered
    global_calc_cbf = np.mean(cbf[porous_ring_mask])
    voxel_volume_ml = np.prod(hrgt["t1"].voxel_size_mm) / 1000.0
    porous_mass = np.sum(porous_mask) * voxel_volume_ml * np.mean(bbpc[porous_mask])
    # calculate the global flow rate in ml/min
    calc_global_flow = global_calc_cbf * porous_mass / 100.0

    # check that the calculated global flow rate in the porous material matches the value
    # originally set
    numpy.testing.assert_array_almost_equal(calc_global_flow, qasper_flow_rate)


def calc_cbf_gkm_casl(
    delta_m,
    m0,
    f,
    transit_time,
    t1_tissue,
    t1_blood,
    blood_brain_pc,
    lab_dur,
    pld,
    lab_eff,
) -> np.ndarray:
    r"""Calculates CBF using the full general kinetic model for a single PLD.

    :param delta_m: Difference in magnetisation between control and label
    :type delta_m: np.ndarray
    :param m0: Equilibrium magnetisation
    :type m0: np.ndarray
    :param f: perfusion rate in ml/100g/min - yes the quantity required is needed to calculate it!
      This is only used to calculate :math:`\frac{f}{\lambda}, so if omitted then this is
      equivalent to approximating that :math:`T_1' == T_{1,\text{tissue}}`
    :type f: np.ndarray
    :param transit_time: The transit time in seconds.
    :type transit_time: np.ndarray
    :param t1_tissue: The tissue T1 in seconds.
    :type t1_tissue: np.ndarray
    :param t1_blood: The blood T1 in seconds.
    :type t1_blood: float
    :param blood_brain_pc: The blood brain partition coefficient, :math:`\lambda` in g/ml.
    :type blood_brain_pc: np.ndarray
    :param lab_dur: label duration in seconds.
    :type lab_dur: float
    :param pld: The post labelling delay in seconds.
    :type pld: float
    :param lab_eff: The labelling efficiency
    :type lab_eff: float
    :return: The calculated CBF
    :rtype: np.ndarray
    """
    # pre-calculate anything where an array is involved in a division using np.divide
    one_over_t1 = np.divide(
        1, t1_tissue, out=np.zeros_like(t1_tissue), where=t1_tissue != 0
    )
    flow_over_lambda = np.divide(
        f / 6000,
        blood_brain_pc,
        out=np.zeros_like(blood_brain_pc),
        where=blood_brain_pc != 0,
    )
    denominator = one_over_t1 + flow_over_lambda

    t1_prime = np.divide(
        1, denominator, out=np.zeros_like(denominator), where=denominator != 0
    )

    q_ss = 1 - np.exp(
        -np.divide(lab_dur, t1_prime, out=np.zeros_like(t1_prime), where=t1_prime != 0)
    )

    exp_pld_tt_t1p = np.exp(
        np.divide(
            (pld - transit_time),
            t1_prime,
            out=np.zeros_like(t1_prime),
            where=t1_prime != 0,
        )
    )
    exp_tt_t1b = np.exp(transit_time / t1_blood)
    norm_delta_m = np.divide(delta_m, m0, out=np.zeros_like(m0), where=m0 != 0)
    denominator = 2 * lab_eff * q_ss * t1_prime
    cbf = 6000 * (
        np.divide(
            blood_brain_pc * norm_delta_m,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator != 0,
        )
        * exp_tt_t1b
        * exp_pld_tt_t1p
    )

    return cbf
