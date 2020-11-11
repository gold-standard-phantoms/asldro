""" Visual checking for resampling.py
Can be run from the command line: python example_resampling.py
For visualising the results of resampling on a test image. Requires
matplotlib which is not a dependency of asldro
 """

import numpy as np
import matplotlib.pyplot as plt
import nilearn as nil
import asldro.utils.resampling as rs
from asldro.utils.test_resampling import create_test_image
from asldro.containers.image import NiftiImageContainer


def check_resampling():
    """Function to visualise the resampling of a test image using pylot
    requires matplotlib.pyplot"""

    rotation = (0.0, 0.0, -45.0)
    translation = (10.0, -10.0, 0.0)
    target_shape = (128, 128, 1)
    (nifti_image, rotation_origin) = create_test_image()

    # use transform_resample_affine to obtain the affine
    target_affine_1, _ = rs.transform_resample_affine(
        nifti_image, translation, rotation, rotation_origin, target_shape
    )

    # resample using nilearn function
    resampled_nifti = nil.image.resample_img(
        nifti_image, target_affine=target_affine_1, target_shape=target_shape
    )
    nifti_image_container = NiftiImageContainer(nifti_image)
    resampled_nifti_image_container = NiftiImageContainer(resampled_nifti)

    # visually check
    plt.figure()
    plt.imshow(np.fliplr(np.rot90(nifti_image_container.image, axes=(1, 0))))
    plt.title("original image")
    plt.axis("image")

    plt.figure()
    plt.imshow(np.fliplr(np.rot90(resampled_nifti_image_container.image, axes=(1, 0))))
    plt.title("transformed and resampled with filter")
    plt.axis("image")
    plt.text(
        0,
        resampled_nifti_image_container.shape[1],
        f"rotation={rotation}"
        "\n"
        f"rotation origin-{rotation_origin}"
        "\n"
        f"translation={translation}"
        "\n"
        f"shape = {resampled_nifti_image_container.shape}"
        "\n",
        {"color": "white"},
    )
    plt.show()


if __name__ == "__main__":
    check_resampling()
