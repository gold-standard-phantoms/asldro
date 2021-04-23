import nibabel as nib
from nibabel.nifti1 import Nifti1Image

from asldro.data.filepaths import GROUND_TRUTH_DATA


def check_nifti_units():
    for key in GROUND_TRUTH_DATA.keys():
        img: Nifti1Image = nib.load(GROUND_TRUTH_DATA[key]["nii"])
        print(img.header.get_xyzt_units())
        if img.header.get_xyzt_units() is None:
            img.header.set_xyzt_units(xyz="mm", t="sec")
            nib.save(img, GROUND_TRUTH_DATA[key])


if __name__ == "__main__":
    check_nifti_units()
