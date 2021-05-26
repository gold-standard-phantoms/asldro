"""Pipeline to perform ASL quantification and save CBF map"""
import logging
import os
import json
import nibabel as nib
from asldro.filters.json_loader import JsonLoaderFilter
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.filters.asl_quantification_filter import AslQuantificationFilter
from asldro.filters.load_asl_bids_filter import LoadAslBidsFilter
from asldro.validators.schemas.index import SCHEMAS
from asldro.utils.general import splitext, map_dict

PASL_DEFAULT_PARAMS = {
    "QuantificationModel": "whitepaper",
    "PostLabelingDelay": 1.8,
    "BolusCutOffDelayTime": 0.98,
    "BloodBrainPartitionCoefficient": 0.9,
    "ArterialSpinLabelingType": "PCASL",
    "LabelingDuration": 1.8,
}
CASL_DEFAULT_PARAMS = {
    "QuantificationModel": "whitepaper",
    "PostLabelingDelay": 1.8,
    "LabelingEfficiency": 0.85,
    "BloodBrainPartitionCoefficient": 0.9,
    "ArterialSpinLabelingType": "PCASL",
    "LabelingDuration": 1.8,
}

DEFAULT_QUANT_PARAMS = {
    "PASL": PASL_DEFAULT_PARAMS,
    "PCASL": CASL_DEFAULT_PARAMS,
    "CASL": CASL_DEFAULT_PARAMS,
}
DEFAULT_T1_ARTERIAL_BLOOD = {3: 1.65, 1.5: 1.35}

BIDS_TO_ASLDRO_MAPPING = {
    "QuantificationModel": "model",
    "PostLabelingDelay": "post_label_delay",
    "LabelingDuration": "label_duration",
    "LabelingEfficiency": "label_efficiency",
    "BloodBrainPartitionCoefficient": "lambda_blood_brain",
    "T1ArterialBlood": "t1_arterial_blood",
    "ArterialSpinLabelingType": "label_type",
}


def asl_quantification(
    asl_nifti_filename: str, quant_params_filename: str = None, output_dir: str = None
) -> dict:
    """[summary]

    :param asl_image_filename: [description]
    :type asl_image_filename: str
    :param quant_params_filename: [description]
    :type quant_params_filename: str
    :param output_dir: [description], defaults to None
    :type output_dir: str, optional
    :return: [description]
    :rtype: dict
    """
    # load in the asl images and quantification parameters, validate against
    # the schema
    input_quant_params = {}
    if quant_params_filename is not None:
        json_filter = JsonLoaderFilter()
        json_filter.add_input("filename", quant_params_filename)
        json_filter.add_input("schema", SCHEMAS["asl_quantification"])
        json_filter.run()
        input_quant_params = json_filter.outputs

    # construct the *_asl.json and *_aslcontext.tsv from the nifti filename
    base_filename = splitext(asl_nifti_filename)
    asl_sidecar_filename = base_filename[0] + ".json"
    aslcontext_filename = base_filename[0] + "context.tsv"

    asl_bids_loader = LoadAslBidsFilter()
    asl_bids_loader.add_input(LoadAslBidsFilter.KEY_IMAGE_FILENAME, asl_nifti_filename)
    asl_bids_loader.add_input(
        LoadAslBidsFilter.KEY_SIDECAR_FILENAME, asl_sidecar_filename
    )
    asl_bids_loader.add_input(
        LoadAslBidsFilter.KEY_ASLCONTEXT_FILENAME, aslcontext_filename
    )
    asl_bids_loader.run()
    label_image = asl_bids_loader.outputs[LoadAslBidsFilter.KEY_LABEL]
    # pull out the required fields from the image's metadata (BIDS sidecar)
    params_from_image = {
        key: label_image.metadata.get(key)
        for key in [
            "PostLabelingDelay",
            "LabelingEfficiency",
            "LabelingDuration",
            "ArterialSpinLabelingType",
            "BolusCutOffDelayTime",
        ]
        if key in label_image.metadata
    }

    # the image must have the BIDS field "ArterialSpinLabelingType" otherwise
    # it is not possible to process
    if not params_from_image.get("ArterialSpinLabelingType") in [
        "PCASL",
        "CASL",
        "PASL",
    ]:
        raise ValueError(
            "Input ASL image must have BIDS field 'ArterialSpinLabelingType "
            "set to either 'PASL', 'CASL', or 'PCASL'"
            f"\nvalue is {params_from_image.get('ArterialSpinLabelingType')}"
        )

    # merge image derived parameters with the input quantification parameters (if
    # supplied), priority to input parameters.
    quant_params = {
        **params_from_image,
        **input_quant_params,
    }
    # merge with defaults, overriding any that are missing:
    quant_params = {
        **DEFAULT_QUANT_PARAMS[quant_params["ArterialSpinLabelingType"]],
        **quant_params,
    }
    # t1 arterial blood is field strength dependent, determine based on this
    if input_quant_params.get("T1ArterialBlood", None) is None:
        quant_params["T1ArterialBlood"] = DEFAULT_T1_ARTERIAL_BLOOD[
            label_image.metadata["MagneticFieldStrength"]
        ]

    asl_quantification_filter = AslQuantificationFilter()
    asl_quantification_filter.add_inputs(map_dict(quant_params, BIDS_TO_ASLDRO_MAPPING))
    asl_quantification_filter.add_inputs(asl_bids_loader.outputs)
    asl_quantification_filter.run()

    # save to pseudo-BIDS format (i.e. NIFTI + JSON sidecar but not with the
    # correct folder structure or filename conventions etc)
    # construct the filename
    output_filenames = {}
    if output_dir is not None:
        output_base_filename = os.path.split(base_filename[0])[1] + "_cbf"
        output_nifti_filename = os.path.join(
            output_dir, output_base_filename + ".nii.gz"
        )
        output_json_filename = os.path.join(output_dir, output_base_filename + ".json")

        nib.save(
            asl_quantification_filter.outputs["perfusion_rate"].nifti_image,
            output_nifti_filename,
        )
        with open(output_json_filename, "w") as json_file:
            json.dump(
                asl_quantification_filter.outputs["perfusion_rate"].metadata,
                json_file,
                indent=4,
            )
        output_filenames = {
            "nifti": output_nifti_filename,
            "json": output_json_filename,
        }

    return {
        "image": asl_quantification_filter.outputs["perfusion_rate"],
        "filenames": output_filenames,
        "quantification_parameters": quant_params,
    }
