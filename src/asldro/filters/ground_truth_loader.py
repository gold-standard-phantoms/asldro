""" Ground truth loader filter """
import copy
import jsonschema
import numpy as np

from asldro.containers.image import NiftiImageContainer
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.validators.parameters import (
    Parameter,
    ParameterValidator,
    isinstance_validator,
    for_each_validator,
)
from asldro.validators.schemas.index import SCHEMAS


class GroundTruthLoaderFilter(BaseFilter):
    """A filter for loading ground truth NIFTI/JSON
    file pairs.

    **Inputs**

    Input Parameters are all keyword arguments for the :class:`GroundTruthLoaderFilter.add_inputs()`
    member function. They are also accessible via class constants,
    for example :class:`GroundTruthLoaderFilter.KEY_IMAGE`

    :param 'image': ground truth image, must be 5D and the 5th dimension have the same length as
        the number of quantities.
    :type 'image': NiftiImageContainer
    :param 'quantities': list of quantity names
    :type 'quantities': list[str]
    :param 'units': list of units corresponding to the quantities, must be the same length as
        quantities
    :type 'units': list[str]
    :param 'parameters': dictionary containing keys
        "t1_arterial_blood", "lambda_blood_brain"
        and "magnetic_field_strength".
    :type 'parameters': dict
    :param 'segmentation': dictionary containing key-value pairs corresponding
        to tissue type and label value in the "seg_label" volume.
    :param 'image_override': (optional) dictionary containing single-value override values
    for any of the 'image' that are loaded. The keys must match the quantity name
    defined in 'quantities'.
    :type 'image_override': dict
    :param 'parameter_override': (optional) dictionary containing single-value override values
    for any of the 'parameters' that are loaded. The keys must match the key defined in
    'parameters'.
    :type 'parameter_override': dict
    :param 'ground_truth_modulate': dictionary with keys corresponding with quantity names.
    The possible dictionary values (both optional) are:
    {
        "scale": N,
        "offset": M
    }
    Any corresponding images will have the corresponding scale and offset applied before being
    output. See :class:`ScaleOffsetFilter` for more details.
    :type 'ground_truth_modulate': dict


    **Outputs**

    Once run, the filter will populate the dictionary
    :class:`GroundTruthLoaderFilter.outputs`
    with output fields based on the input 'quantities'.
    Each key in 'quantities' will result in a NiftiImageContainer
    corresponding to a 3D/4D subset of the nifti input (split along the 5th
    dimension). The data types of images will be the
    same as those input EXCEPT for a quantity labelled "seg_label"
    which will be converted to a uint16 data type.
    If 'override_image' is defined, the corresponding 'image' will be set to the overriding
    value before being output.
    If 'override_parameters' is defined, the corresponding parameter will be set to the
    overriding value before being output.
    If 'ground_truth_modulate' is defined, the corresponding 'image'(s) will be scaled and/or
    offset by the corresponding values.
    The keys-value pairs in the input 'parameters' will also
    be destructured and piped through to the output, for example:
    :param 't1': volume of T1 relaxation times
    :type 't1': NiftiImageContainer
    :param 'seg_label': segmentation label mask corresponding to different tissue types.
    :type 'seg_label': NiftiImageContainer (uint16 data type)
    :param 'magnetic_field_strength': the magnetic field strenght in Tesla.
    :type 'magnetic_field_strength': float
    :param 't1_arterial_blood': the T1 relaxation time of arterial blood
    :type 't1_arterial_blood': float
    :param 'lambda_blood_brain': the blood-brain-partition-coefficient
    :type 'lambda_blood_brain': float


    A field metadata will be created in each image container, with the
    following fields:

    * ``magnetic_field_strength``: corresponds to the value in the
      "parameters" object.
    * ``quantity``: corresponds to the entry in the "quantities" array.
    * ``units``: corresponds with the entry in the "units" array.


    The "segmentation" object from the JSON file will also be
    piped through to the metadata entry of the "seg_label" image container.
    """

    KEY_IMAGE = "image"
    KEY_UNITS = "units"
    KEY_SEGMENTATION = "segmentation"
    KEY_PARAMETERS = "parameters"
    KEY_QUANTITIES = "quantities"
    KEY_QUANTITY = "quantity"
    KEY_MAG_STRENGTH = "magnetic_field_strength"
    KEY_IMAGE_OVERRIDE = "image_override"
    KEY_PARAMETER_OVERRIDE = "parameter_override"
    KEY_GROUND_TRUTH_MODULATE = "ground_truth_modulate"

    def __init__(self):
        super().__init__("GroundTruthLoader")

    def _run(self):
        """Load the inputs using a NiftiLoaderFilter and JsonLoaderFilter.
        Create the image outputs and the segmentation key outputs"""
        image_container: NiftiImageContainer = self.inputs[self.KEY_IMAGE]
        for i, quantity in enumerate(self.inputs[self.KEY_QUANTITIES]):
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

            # If we have a corresponding 'image_override' value, update the
            # 'image_data' with that value
            if (
                self.KEY_IMAGE_OVERRIDE in self.inputs
                and quantity in self.inputs[self.KEY_IMAGE_OVERRIDE]
            ):
                image_data.fill(self.inputs[self.KEY_IMAGE_OVERRIDE][quantity])

            # If we have a segmentation label, round and squash the
            # data to uint16 and update the NIFTI header
            metadata = {}
            if quantity == "seg_label":
                header["datatype"] = 512
                image_data = np.around(image_data).astype(dtype=np.uint16)
                metadata[self.KEY_SEGMENTATION] = self.inputs[self.KEY_SEGMENTATION]

            nifti_image_type = image_container.nifti_type
            metadata[self.KEY_MAG_STRENGTH] = self.inputs[self.KEY_PARAMETERS][
                self.KEY_MAG_STRENGTH
            ]
            metadata[self.KEY_QUANTITY] = quantity
            metadata[self.KEY_UNITS] = self.inputs[self.KEY_UNITS][i]

            # If we have a ground_truth_modulate input, and this quantity is to be modulated
            if (
                "ground_truth_modulate" in self.inputs
                and quantity in self.inputs["ground_truth_modulate"]
            ):
                scale_offset = self.inputs["ground_truth_modulate"][quantity]
                if "scale" in scale_offset:
                    # Allow unsafe casting (allow data-type conversion)
                    image_data = np.multiply(
                        image_data, scale_offset["scale"], casting="unsafe"
                    )
                if "offset" in scale_offset:
                    # Allow unsafe casting (allow data-type conversion)
                    image_data = np.add(
                        image_data, scale_offset["offset"], casting="unsafe"
                    )

            new_image_container = NiftiImageContainer(
                nifti_img=nifti_image_type(
                    dataobj=image_data, affine=image_container.affine, header=header
                ),
                metadata=metadata,
            )

            self.outputs[quantity] = new_image_container

        # Get the parameter_override dictionary (empty dict if it doesn't exist)
        overrides = (
            self.inputs[self.KEY_PARAMETER_OVERRIDE]
            if self.KEY_PARAMETER_OVERRIDE in self.inputs
            else {}
        )
        # Pipe through all parameters
        self.outputs = {**self.outputs, **self.inputs[self.KEY_PARAMETERS], **overrides}

    def _validate_inputs(self):
        """Checks that the inputs meet their validation criteria.
        'image': NiftiImageContainer, must be 5D, 5th dimension same length as 'quantities'
        'quantities': list[str]
        'units': list[str]
        'segmentation': dict
        'parameters': dict
        'image_override': dict (optional)
        'parameter_override': dict (optional)
        The number of 'units' and 'quantities' should be equal.
        The size of the 5th dimension of the image must equal the number of 'quantities'.
        If 'image_override' is present, must be a dict, each of the keys must be a string
        and the values a float or int type. The key must match an entry in 'quantities'.
        If 'parameter_override' is present, must be a dict, each of the keys must be a
        string and the values a float or int type.
        """
        input_validator = ParameterValidator(
            parameters={
                self.KEY_IMAGE: Parameter(
                    validators=isinstance_validator(NiftiImageContainer)
                ),
                self.KEY_QUANTITIES: Parameter(
                    validators=for_each_validator(isinstance_validator(str))
                ),
                self.KEY_UNITS: Parameter(
                    validators=for_each_validator(isinstance_validator(str))
                ),
                self.KEY_SEGMENTATION: Parameter(validators=isinstance_validator(dict)),
                self.KEY_PARAMETERS: Parameter(validators=isinstance_validator(dict)),
                self.KEY_GROUND_TRUTH_MODULATE: Parameter(
                    validators=isinstance_validator(dict), optional=True
                ),
            }
        )
        input_validator.validate(self.inputs, error_type=FilterInputValidationError)

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

        if len(self.inputs["units"]) != len(self.inputs["quantities"]):
            raise FilterInputValidationError(
                f"{self} filter requires an input 'units' which is the same length as the input "
                "'quantities'"
            )

        for input_key in [self.KEY_IMAGE_OVERRIDE, self.KEY_PARAMETER_OVERRIDE]:
            if input_key in self.inputs:
                # Check the value is a dictionary
                if not isinstance(self.inputs[input_key], dict):
                    raise FilterInputValidationError(
                        f"{input_key} must be a dictionary, is {self.inputs[input_key]}"
                    )
                # Check all of the keys are strings and the values are int or float
                for key, value in self.inputs[input_key].items():
                    if not isinstance(key, str):
                        raise FilterInputValidationError(
                            f"All keys in the {input_key} dictionary must be strings. "
                            f"{key} is not."
                        )
                    if not isinstance(value, (int, float)):
                        raise FilterInputValidationError(
                            f"Values in the {input_key} dictionary must be int or float. "
                            f"The value for {key} ({value}) is not."
                        )

        if self.KEY_IMAGE_OVERRIDE in self.inputs:
            # Check all of the keys are present in self.inputs['quantities']
            for key, value in self.inputs[self.KEY_IMAGE_OVERRIDE].items():
                if key not in self.inputs[self.KEY_QUANTITIES]:
                    raise FilterInputValidationError(
                        f"{key} is not in the input 'quantities' list"
                    )
        if self.KEY_GROUND_TRUTH_MODULATE in self.inputs:
            try:
                jsonschema.validate(
                    self.inputs[self.KEY_GROUND_TRUTH_MODULATE],
                    SCHEMAS["input_params"]["properties"]["global_configuration"][
                        "properties"
                    ]["ground_truth_modulate"],
                )
            except jsonschema.ValidationError as exception:
                raise FilterInputValidationError from exception
