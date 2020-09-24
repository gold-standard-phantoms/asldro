""" GroundTruthLoaderFilter tests """
import os
from tempfile import TemporaryDirectory

import numpy as np
import numpy.testing
import nibabel as nib

from asldro.filters.ground_truth_loader import GroundTruthLoaderFilter
from asldro.filters.json_loader import JsonLoaderFilter
from asldro.filters.nifti_loader import NiftiLoaderFilter
from asldro.containers.image import NiftiImageContainer
from asldro.data.filepaths import (
    HRGT_ICBM_2009A_NLS_V3_JSON,
    HRGT_ICBM_2009A_NLS_V3_NIFTI,
)


def test_ground_truth_loader_filter_with_mock_data():
    """ Test the ground truth loader filter with some mock data """
    with TemporaryDirectory() as temp_dir:
        images = [np.ones(shape=(3, 3, 3, 1), dtype=np.float32) * i for i in range(7)]

        stacked_image = np.stack(arrays=images, axis=4)
        # Create a 5D numpy image (float32) where the value of voxel corresponds
        # with the distance across 5th dimension (from 0 to 6 inclusive)
        img = nib.Nifti2Image(dataobj=stacked_image, affine=np.eye(4))

        temp_file = os.path.join(temp_dir, "file.nii")
        nib.save(img, filename=temp_file)
        img = nib.load(filename=temp_file)

        nifti_image_container = NiftiImageContainer(nifti_img=img)

        ground_truth_filter = GroundTruthLoaderFilter()
        ground_truth_filter.add_input("image", nifti_image_container)
        ground_truth_filter.add_input(
            "quantities",
            [
                "perfusion_rate",
                "transit_time",
                "t1",
                "t2",
                "t2_star",
                "m0",
                "seg_label",
            ],
        )
        ground_truth_filter.add_input(
            "segmentation", {"csf": 3, "grey_matter": 1, "white_matter": 2}
        )
        ground_truth_filter.add_input(
            "parameters", {"lambda_blood_brain": 0.9, "t1_arterial_blood": 1.65}
        )

        # Should run without error
        ground_truth_filter.run()

        # Should be piped through
        assert ground_truth_filter.outputs["segmentation"] == {
            "csf": 3,
            "grey_matter": 1,
            "white_matter": 2,
        }

        # Parameters should be piped through individually
        assert ground_truth_filter.outputs["lambda_blood_brain"] == 0.9
        assert ground_truth_filter.outputs["t1_arterial_blood"] == 1.65

        assert ground_truth_filter.outputs["perfusion_rate"].image.dtype == np.float32
        numpy.testing.assert_array_equal(
            ground_truth_filter.outputs["perfusion_rate"].image,
            np.zeros((3, 3, 3), dtype=np.float32),
        )

        assert ground_truth_filter.outputs["transit_time"].image.dtype == np.float32
        numpy.testing.assert_array_equal(
            ground_truth_filter.outputs["transit_time"].image,
            np.ones((3, 3, 3), dtype=np.float32),
        )

        assert ground_truth_filter.outputs["t1"].image.dtype == np.float32
        numpy.testing.assert_array_equal(
            ground_truth_filter.outputs["t1"].image,
            np.ones((3, 3, 3), dtype=np.float32) * 2,
        )

        assert ground_truth_filter.outputs["t2"].image.dtype == np.float32
        numpy.testing.assert_array_equal(
            ground_truth_filter.outputs["t2"].image,
            np.ones((3, 3, 3), dtype=np.float32) * 3,
        )

        assert ground_truth_filter.outputs["t2_star"].image.dtype == np.float32
        numpy.testing.assert_array_equal(
            ground_truth_filter.outputs["t2_star"].image,
            np.ones((3, 3, 3), dtype=np.float32) * 4,
        )

        assert ground_truth_filter.outputs["m0"].image.dtype == np.float32
        numpy.testing.assert_array_equal(
            ground_truth_filter.outputs["m0"].image,
            np.ones((3, 3, 3), dtype=np.float32) * 5,
        )

        # Check the seg_label type has changed to a uint16
        assert ground_truth_filter.outputs["seg_label"].image.dtype == np.uint16
        numpy.testing.assert_array_equal(
            ground_truth_filter.outputs["seg_label"].image,
            np.ones((3, 3, 3), dtype=np.uint16) * 6,
        )
        del img
        del nifti_image_container
        del ground_truth_filter


def test_ground_truth_loader_filter_with_test_data():
    """ Test the ground truth loader filter with the included
    test data """

    json_filter = JsonLoaderFilter()
    json_filter.add_input("filename", HRGT_ICBM_2009A_NLS_V3_JSON)

    nifti_filter = NiftiLoaderFilter()
    nifti_filter.add_input("filename", HRGT_ICBM_2009A_NLS_V3_NIFTI)

    ground_truth_filter = GroundTruthLoaderFilter()
    ground_truth_filter.add_parent_filter(nifti_filter)
    ground_truth_filter.add_parent_filter(json_filter)

    # Should run without error
    ground_truth_filter.run()
