""" Fourier Transform filter """
import numpy as np
#from asldro.containers.image import NumpyImageContainer, NiftiImageContainer
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError


class FftFilter(BaseFilter):
    """ A filter for performing a n-dimensional fast fourier transform of input.
    Input is a numpy array named 'image'.
    Output is a complex numpy array of the discrete fourier transform named 'kdata'"""

    def __init__(self):
        super().__init__(name="fft")

    def _run(self):
        """ performs a n-dimensional fast fourier transform on the input
        and creates an 'output' with the result
        """
        self.outputs["kdata"] = np.fft.fftn(self.inputs["image"])

    def _validate_inputs(self):
        """" Input must be a numpy array"""
        input_value = self.inputs["image"]
        if not isinstance(input_value, np.ndarray):
            raise FilterInputValidationError(
                f"Input image is not a ndarray (is {type(input_value)})"
                )

class IfftFilter(BaseFilter):
    """ A filter for performing a n-dimensional inverse fast fourier transform of input.
    Input is a numpy array named 'kdata'.
    Output is a complex numpy array of the inverse discrete fourier transform named 'image' """

    def __init__(self):
        super().__init__(name="fft")

    def _run(self):
        """ performs a n-dimensional inverse fast fourier transform on the input
        and creates an 'output' with the result
        """
        self.outputs["image"] = np.fft.ifftn(self.inputs["kdata"])

    def _validate_inputs(self):
        """" Input must be a numpy array"""
        input_value = self.inputs["kdata"]
        if not isinstance(input_value, np.ndarray):
            raise FilterInputValidationError(
                f"Input kdata is not a ndarray (is {type(input_value)})"
                )
