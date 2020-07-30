from typing import Union

import numpy as np
import nibabel as nib


class ImageContainer:
    """ A container for an ND image. Many be initialised with
    a nibabel Nifti1Image or Nifti2Image, or from a numpy array
    and some associated metadata. """

    UNITS_METERS = "meters"
    UNITS_MILLIMETERS = "mm"
    UNITS_MICRONS = "micron"
    UNITS_SECONDS = "sec"
    UNITS_MILLISECONDS = "msec"
    UNITS_MICROSECONDS = "usec"

    def __init__(self, nifti_img: Union[nib.Nifti1Image, nib.Nifti2Image] = None):
        self._nifti_image: Union[nib.Nifti1Image, nib.Nifti2Image] = nifti_img
        self._image: np.array = None
        self._affine = None
        self._pixdim = None

    @property
    def has_nifti(self):
        """ Returns True if the image has an associated nifti container """
        return self._nifti_image is not None

    @property
    def image(self):
        """ Return the image data as a numpy array """
        if self.has_nifti:
            return self._nifti_image.get_fdata()
        return self._image

    @property
    def header(self) -> Union[nib.Nifti1Header, nib.Nifti2Header]:
        """ Returns the NIFTI header if initialised from a NIFTI file,
        othewise returns None """
        if self.has_nifti:
            return self._nifti_image.header
        return None

    @property
    def affine(self):
        """ Return a 4x4 numpy array with the image affine transformation """
        if self.has_nifti:
            return self._nifti_image.affine
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
        if self.header is not None:
            return self.header.get_xyzt_units()[0]
        return None

    @property
    def time_units(self):
        """
        Uses the NIFTI header xyzt_units to extract the time units.
        Returns one of:
        'sec'
        'msec'
        'usec'
        """
        if self.header is not None:
            return self.header.get_xyzt_units()[1]
        return None

    @property
    def pixdim(self):
        """ Return a list of length 4 [x,y,z,t]. Each entry corresponds
        with the size of each dimension (voxel width or timestep) """
        if self.has_nifti:
            return self._nifti_image.header["pixdim"][1:5]
        return self._pixdim

    @property
    def shape(self):
        """ Returns the shape of the image [x, y, z, (t)] """
        if self.has_nifti:
            return self._nifti_image.shape
        if self._image is not None:
            return self._image.shape
        return None
