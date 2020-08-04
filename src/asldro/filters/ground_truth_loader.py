""" Ground truth loader filter """
import copy
import numpy as np

from asldro.containers.image import NiftiImageContainer
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError


class GroundTruthLoaderFilter(BaseFilter):
    """ A filter for loading ground truth NIFTI/JSON
    file pairs.

    Must have two inputs named 'image' and 'quantities'.
    These correspond with a NiftiImageContainer an array of strings.

    Will create multiple ImageContainer outputs each corresponding
    with a 3D/4D subset of the nifti input (split along the 5th
    dimension). These will be output using keys from the JSON file
    (the quantities section). The data types of images will be the
    same as those input EXCEPT for a quantity labelled "seg_label"
    which will be converted to a uint16 data type.

    The "segmentation" object from the
    JSON file will also be piped through to an output.
    """

    def __init__(self):
        super().__init__("GroundTruthLoader")

    def _run(self):
        """ Load the inputs using a NiftiLoaderFilter and JsonLoaderFilter.
        Create the image outputs and the segmentation key outputs """
        image_container: NiftiImageContainer = self.inputs["image"]
        for i, quantity in enumerate(self.inputs["quantities"]):
            # Create a new NiftiContainer - easier as we can just augment
            # the header to remove the 5th dimension

            header = copy.deepcopy(image_container.header)
            header["dim"][0] = 4  #  Remove the 5th dimension
            if header["dim"][4] == 1:
                # If we only have 1 time-step, reduce to 3D
                header["dim"][0] = 3
            header["dim"][5] = 1  # tidy the 5th dimensions size

            # Grab the relevant image data
            image_data: np.ndarray = image_container.image[:, :, :, :, i]
            if header["dim"][0] == 3:
                # squeeze the 4th dimension if there is only one time-step
                image_data = np.squeeze(image_data, axis=3)

            # If we have a segmentation label, round and squash the
            # data to uint16 and update the NIFTI header
            if quantity == "seg_label":
                header["datatype"] = 512
                image_data = np.around(image_data).astype(dtype=np.uint16)

            nifti_image_type = image_container.nifti_type

            new_image_container = NiftiImageContainer(
                nifti_img=nifti_image_type(
                    dataobj=image_data, affine=image_container.affine, header=header
                )
            )
            self.outputs[quantity] = new_image_container

    def _validate_inputs(self):
        """ There must be a input called 'image' with a ImageContainer.
        There must also be a 'quantities' array.
        The size of the 5th dimension of the image must equal the number of 'quantities'
        """

        if self.inputs.get("image", None) is None or not isinstance(
            self.inputs["image"], NiftiImageContainer
        ):
            raise FilterInputValidationError(
                "GroundTruthLoader filter requires a `image` input of ImageContainer type"
            )

        image_container: NiftiImageContainer = self.inputs["image"]
        if len(image_container.shape) != 5:
            raise FilterInputValidationError(
                f"{self} filter requires an input nifti which is 5D"
            )

        if image_container.shape[4] != len(self.inputs["quantities"]):
            raise FilterInputValidationError(
                f"{self} filter requires an input nifti which has the "
                "same number of images (across the 5th dimension) as the JSON filter "
                "supplies in 'quantities'"
            )
