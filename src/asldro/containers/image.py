""" Classes for image encapsulation
Used to create a standard interface for ND images which can
be instantiated with either NIFTI files or using numpy arrays """

from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Union, Tuple, Type


import numpy as np
import nibabel as nib


UNITS_METERS = "meter"
UNITS_MILLIMETERS = "mm"
UNITS_MICRONS = "micron"
UNITS_SECONDS = "sec"
UNITS_MILLISECONDS = "msec"
UNITS_MICROSECONDS = "usec"

SPATIAL_DOMAIN = "SPATIAL_DOMAIN"
INVERSE_DOMAIN = "INVERSE_DOMAIN"

REAL_IMAGE_TYPE = "REAL_IMAGE_TYPE"
IMAGINARY_IMAGE_TYPE = "IMAGINARY_IMAGE_TYPE"
MAGNITUDE_IMAGE_TYPE = "MAGNITUDE_IMAGE_TYPE"
PHASE_IMAGE_TYPE = "PHASE_IMAGE_TYPE"
COMPLEX_IMAGE_TYPE = "COMPLEX_IMAGE_TYPE"


class BaseImageContainer(ABC):
    """
    An abstract (interface) for an ND image. Defines
    a set of accessors
    :param data_domain: defines the data domain as SPATIAL_DOMAIN or
    INVERSE_DOMAIN (default is SPATIAL_DOMAIN)
    :param image_type: the image type. Must be one of:
    - REAL_IMAGE_TYPE
    - IMAGINARY_IMAGE_TYPE
    - MAGNITUDE_IMAGE_TYPE
    - PHASE_IMAGE_TYPE
    - COMPLEX_IMAGE_TYPE
    if this is not specified, it will be set to MAGNITUDE_IMAGE_TYPE for scalar
    image dtypes and COMPLEX_IMAGE_TYPE for complex dtypes
    """

    def __init__(self, data_domain: str = SPATIAL_DOMAIN, image_type=None, **kwargs):
        if data_domain not in [SPATIAL_DOMAIN, INVERSE_DOMAIN]:
            raise ValueError(
                "data_domain is not of of SPATIAL_DOMAIN or INVERSE_DOMAIN"
            )
        self.data_domain = data_domain

        # Set the default image type based on the image dtype
        if image_type is None:
            if self.image.dtype in [np.complex64, np.complex128]:
                image_type = COMPLEX_IMAGE_TYPE
            else:
                image_type = MAGNITUDE_IMAGE_TYPE

        # Check the image_type is in the valid options
        if image_type not in [
            IMAGINARY_IMAGE_TYPE,
            REAL_IMAGE_TYPE,
            PHASE_IMAGE_TYPE,
            COMPLEX_IMAGE_TYPE,
            MAGNITUDE_IMAGE_TYPE,
        ]:
            raise ValueError(f"{self} has bad value for image_type ({image_type})")

        # Check the image type matches the image dtype
        # complex dtypes must match COMPLEX_IMAGE_TYPE
        # non-complex dtype must not match COMPLEX_IMAGE_TYPE
        if (
            image_type is COMPLEX_IMAGE_TYPE
            and self.image.dtype not in [np.complex64, np.complex128]
            or image_type is not COMPLEX_IMAGE_TYPE
            and self.image.dtype in [np.complex64, np.complex128]
        ):
            raise ValueError(
                f"{self} created with image type of {image_type} "
                f"and image dtype is {self.image.dtype}"
            )

        self.image_type = image_type

        # Check we aren't passed unexpected parameters
        if len(kwargs) != 0:
            raise TypeError(
                f"BaseImageContainer received unexpected arguments {kwargs}"
            )

    def clone(self) -> "BaseImageContainer":
        """ Makes a deep copy of all member variables in a new ImageContainer """
        return deepcopy(self)

    @property
    @abstractmethod
    def has_nifti(self):
        """ Returns True if the image has an associated nifti container """

    @property
    @abstractmethod
    def image(self):
        """ Return the image data as a numpy array """

    @image.setter
    @abstractmethod
    def image(self, new_image: np.ndarray):
        """ Sets the image data """

    @property
    @abstractmethod
    def affine(self) -> np.ndarray:
        """ Return a 4x4 numpy array with the image affine transformation """

    @property
    @abstractmethod
    def space_units(self):
        """
        Uses the NIFTI header xyzt_units to extract the space units.
        Returns one of:
        'meter'
        'mm'
        'micron'
        """

    @property
    @abstractmethod
    def time_units(self):
        """
        Uses the NIFTI header xyzt_units to extract the time units.
        Returns one of:
        'sec'
        'msec'
        'usec'
        """

    @time_units.setter
    @abstractmethod
    def time_units(self, units: str):
        pass

    @space_units.setter
    @abstractmethod
    def space_units(self, units: str):
        pass

    @property
    @abstractmethod
    def voxel_size_mm(self) -> np.ndarray:
        """ Returns the voxel size in mm """

    @property
    @abstractmethod
    def time_step_seconds(self) -> float:
        """ Return the time step in seconds """

    @property
    @abstractmethod
    def shape(self) -> Tuple[int]:
        """ Returns the shape of the image [x, y, z, t, etc] """

    @staticmethod
    def _validate_time_units(units: str):
        """ Checks whether the given time units is a valid string """
        if units not in [UNITS_MICROSECONDS, UNITS_MILLISECONDS, UNITS_SECONDS]:
            raise ValueError(f'Unit "{units}" not in time units')

    @staticmethod
    def _validate_space_units(units: str):
        """ Checks whether the given space units is a valid string """
        if units not in [UNITS_MICRONS, UNITS_MILLIMETERS, UNITS_METERS]:
            raise ValueError(f'Unit "{units}" not in space units')

    @staticmethod
    def _time_units_to_seconds(units: str) -> float:
        """ Returns the time units in seconds.
        Raises a ValueError if the string is an invalid time unit """
        if units == UNITS_MILLISECONDS:
            return 1e3
        if units == UNITS_MICROSECONDS:
            return 1e6
        if units == UNITS_SECONDS:
            return 1.0
        raise ValueError(f'Unit "{units}" not in time units')

    @staticmethod
    def _space_units_to_mm(units: str) -> float:
        """ Returns the space units in mm.
        Raises a ValueError if the string is an invalid space unit """
        if units == UNITS_METERS:
            return 1e3
        if units == UNITS_MICRONS:
            return 1e-3
        if units == UNITS_MILLIMETERS:
            return 1.0
        raise ValueError(f'Unit "{units}" not in space units')


class NumpyImageContainer(BaseImageContainer):
    """ A container for an ND image. Must be initialised with
    a a numpy array and some associated metadata. """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        image: np.ndarray,
        affine: np.ndarray = np.eye(4),
        space_units: str = UNITS_MILLIMETERS,
        time_units: str = UNITS_SECONDS,
        voxel_size=np.array([1.0, 1.0, 1.0]),
        time_step=1.0,
        **kwargs,
    ):
        """ Creates an image container from a numpy array. May
        provide one or more additional arguments.
        :param image: an ND numpy array containing voxel data
        :param affine: a 4x4 affine matrix (numpy array).
        :param space_units: the units across the space dimensions
        :param time_units: the units across the time dimension
        :param voxel_size: a tuple containing the voxel size (in units of space_units)
        :param time_step: the time step (in units of time_units)
        :param **kwargs: any additional arguments accepted by BaseImageContainer
        """
        self._image: np.ndarray = image
        self._affine: np.ndarray = affine
        self._space_units: str = space_units
        self._time_units: str = time_units
        self._voxel_size: np.array = np.array(voxel_size)
        self._time_step: float = time_step
        super().__init__(**kwargs)  # Call super last as we check member variables

    @property
    def has_nifti(self):
        """ Returns True if the image has an associated nifti container """
        return False

    @property
    def image(self):
        """ Return the image data as a numpy array """
        return self._image

    @image.setter
    def image(self, new_image: np.ndarray):
        """ Sets the image data - does not copy it! """
        self._image = new_image

    @property
    def header(self) -> Union[nib.Nifti1Header, nib.Nifti2Header]:
        """ Returns the NIFTI header if initialised from a NIFTI file,
        othewise returns None """
        return None

    @property
    def affine(self):
        """ Return a 4x4 numpy array with the image affine transformation """
        return self._affine

    @property
    def space_units(self):
        """
        Uses the NIFTI header xyzt_units to extract the space units.
        Returns one of:
        'meter'
        'mm'
        'micron'
        """
        return self._space_units

    @property
    def time_units(self):
        """
        Uses the NIFTI header xyzt_units to extract the time units.
        Returns one of:
        'sec'
        'msec'
        'usec'
        """
        return self._time_units

    @time_units.setter
    def time_units(self, units: str):
        self._validate_time_units(units)
        self._time_units = units

    @space_units.setter
    def space_units(self, units: str):
        self._validate_space_units(units)
        self._space_units = units

    @property
    def voxel_size_mm(self):
        """ Returns the voxel size in mm """
        return self._voxel_size * self._space_units_to_mm(self.space_units)

    @property
    def time_step_seconds(self):
        """ Return the time step in seconds """
        return self._time_step * self._time_units_to_seconds(self.time_units)

    @property
    def shape(self):
        """ Returns the shape of the image [x, y, z, t, etc] """
        return self._image.shape


class NiftiImageContainer(BaseImageContainer):
    """ A container for an ND image. Must be initialised with
    a nibabel Nifti1Image or Nifti2Image """

    def __init__(
        self, nifti_img: Union[nib.Nifti1Image, nib.Nifti2Image] = None, **kwargs
    ):
        """
        :param nifti_img: A nibabel Nifti1Image or Nifti2Image
        :param **kwargs: any additional arguments accepted by BaseImageContainer
        """
        self._nifti_image: Union[nib.Nifti1Image, nib.Nifti2Image] = nifti_img
        super().__init__(**kwargs)  # Call super last as we check member variables

    @property
    def nifti_type(self) -> Type[Union[nib.Nifti1Image, nib.Nifti2Image]]:
        """ Return the type of NIFTI data contained here (nib.Nifti1Image or nib.Nifti2Image) """
        return type(self._nifti_image)

    @property
    def has_nifti(self):
        """ Returns True if the image has an associated nifti container """
        return True

    @property
    def image(self):
        """ Return the image data as a numpy array.
        Returns data in the type it is created (i.e. won't convert to float64 as
        .get_fdata() will)"""
        return np.asanyarray(self._nifti_image.dataobj)
        # return self._nifti_image.get_fdata()

    @image.setter
    def image(self, new_image: np.ndarray):
        """ Sets the image data """
        self._nifti_image = self.nifti_type(
            dataobj=new_image, affine=self.affine, header=self.header
        )
        # Make sure the header matches the new image data
        self._nifti_image.set_data_dtype(new_image.dtype)
        self._nifti_image.update_header()

    @property
    def header(self) -> Union[nib.Nifti1Header, nib.Nifti2Header]:
        """ Returns the NIFTI header if initialised from a NIFTI file,
        othewise returns None """
        return self._nifti_image.header

    @property
    def affine(self):
        """ Return a 4x4 numpy array with the image affine transformation """
        return self._nifti_image.affine

    @property
    def space_units(self):
        """
        Uses the NIFTI header xyzt_units to extract the space units.
        Returns one of:
        'meter'
        'mm'
        'micron'
        """
        return self.header.get_xyzt_units()[0]

    @property
    def time_units(self):
        """
        Uses the NIFTI header xyzt_units to extract the time units.
        Returns one of:
        'sec'
        'msec'
        'usec'
        """
        return self.header.get_xyzt_units()[1]

    @time_units.setter
    def time_units(self, units: str):
        self._validate_time_units(units)
        self.header.set_xyzt_units(t=units)

    @space_units.setter
    def space_units(self, units: str):
        self._validate_space_units(units)
        self.header.set_xyzt_units(xyz=units)

    @property
    def voxel_size_mm(self):
        """ Returns the voxel size in mm """
        return self.header["pixdim"][1:4] * self._space_units_to_mm(self.space_units)

    @property
    def time_step_seconds(self):
        """ Return the time step in seconds """
        return self.header["pixdim"][4] * self._time_units_to_seconds(self.time_units)

    @property
    def shape(self):
        """ Returns the shape of the image [x, y, z, t, etc] """
        return self._nifti_image.shape
