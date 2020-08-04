""" Examples of filter chains """
import pprint

from asldro.filters.ground_truth_loader import GroundTruthLoaderFilter
from asldro.filters.json_loader import JsonLoaderFilter
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.data.filepaths import (
    HRGT_ICBM_2009A_NLS_V3_JSON,
    HRGT_ICBM_2009A_NLS_V3_NIFTI,
)


def run_full_pipeline():
    """ A function that runs the entire DRO pipeline. This
    can be extended as more functionality is included.
    This function is deliberately verbose to explain the
    operation, inputs and outputs of individual filters. """

    json_filter = JsonLoaderFilter()
    json_filter.add_input("filename", HRGT_ICBM_2009A_NLS_V3_JSON)

    nifti_filter = NiftiLoaderFilter()
    nifti_filter.add_input("filename", HRGT_ICBM_2009A_NLS_V3_NIFTI)

    ground_truth_filter = GroundTruthLoaderFilter()
    ground_truth_filter.add_parent_filter(nifti_filter)
    ground_truth_filter.add_parent_filter(json_filter)

    ground_truth_filter.run()

    print(f"JsonLoaderFilter outputs:\n{pprint.pformat(json_filter.outputs)}")
    print(f"NiftiLoaderFilter outputs:\n{pprint.pformat(nifti_filter.outputs)}")
    print(
        f"GroundTruthLoaderFilter outputs:\n{pprint.pformat(ground_truth_filter.outputs)}"
    )


if __name__ == "__main__":
    run_full_pipeline()
