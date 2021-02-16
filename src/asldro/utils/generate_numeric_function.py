"""Functions to generate arrays of mathematical function values"""

from typing import Union
import numpy as np


def generate_gaussian_function(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    loc: Union[tuple, list, np.ndarray] = np.zeros((3,)),
    fwhm: Union[tuple, list, np.ndarray] = np.ones((3,)),
    theta: float = 0.0,
) -> np.ndarray:
    r"""
    Generates a 3-dimensional gaussian function.

    :param xx: Array of x-coordinates, generate by meshgrid. Can be sparse.
    :type xx: np.ndarray
    :param yy: Array of y-coordinates, generate by meshgrid. Can be sparse.
    :type yy: np.ndarray
    :param zz: Array of z-coordinates, generate by meshgrid. Can be sparse.
    :type zz: np.ndarray
    :param loc: origin of the gaussian, :math:`[x_0, y_0, z_0]`, defaults to np.zeros((3,))
    :type loc: Union[tuple, list, np.ndarray], optional
    :param fwhm: full-width-half-maximum of the gaussian, :math:`[\text{fwhm}_x,
     \text{fwhm}_y, \text{fwhm}_z]`, defaults to np.ones((3,))
    :type fwhm: Union[tuple, list, np.ndarray], optional
    :param theta: polar rotation of the gaussian (about an axis parallel to z at the gaussian
      origin), degrees, defaults to 0.0
    :type theta: float, optional

    :raises ValueError: If xx, yy, and zz do not all have shapes that permit broadcasting
    :raises ValueError: If loc is not 1D and of length 3
    :raises ValueError: If fwhm is not 1D and of length 3
    :raises ValueError: If any entry in fwhm is < 0.0
    :return: An array the same size and shape as xx, yy, and zz, with values given by the
      3D gaussian function.
    :rtype: np.ndarray

    The gaussian is generated according to:

    .. math::
        &f(x,y,z) = e^{-(a(x-x_0)^2+ 2b(x-x_0)(y-y_0)+c(y-y_0)^2+d(z-z_0)^2)}\\
        \text{where,}\\
        &a=\frac{\cos^2\theta}{2\sigma^2_x}+\frac{\sin^2\theta}{2\sigma^2_y}\\
        &b=-\frac{\sin 2\theta}{4\sigma^2_x}+\frac{\sin 2\theta}{4\sigma^2_y}\\
        &c=\frac{\sin^2\theta}{2\sigma^2_x}+\frac{\cos^2\theta}{2\sigma^2_y}\\
        &d=\frac{1}{2\sigma^2_z}\\
        &\sigma_{a}=\frac{\text{fwhm}_a}{2\sqrt{2\ln 2}}

    """
    # argument validation
    # both regular and sparse grids allowed - check that xx, yy and zz can be
    # broadcasted
    np.broadcast(xx, yy, zz)

    loc = np.asarray(loc)
    if not (loc.size == 3 and loc.ndim == 1):
        raise ValueError("loc must be 1D and be of length 3")

    fwhm = np.asarray(fwhm)
    if not (fwhm.size == 3 and fwhm.ndim == 1):
        raise ValueError("fwhm must be 1D and be of length 3")

    if (fwhm < 0.0).any():
        raise ValueError("all entries of fwhm must be 0.0 or positive")

    if not isinstance(theta, (float, int)):
        raise TypeError("theta must be a float or int")

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # calculate the width parameter, sigma

    theta_rad = np.radians(theta)
    a = np.cos(theta_rad) ** 2 / (2 * sigma[0] ** 2) + np.sin(theta_rad) ** 2 / (
        2 * sigma[1] ** 2
    )
    b = -np.sin(2 * theta_rad) / (4 * sigma[0] ** 2) + np.sin(2 * theta_rad) / (
        4 * sigma[1] ** 2
    )
    c = np.sin(theta_rad) ** 2 / (2 * sigma[0] ** 2) + np.cos(theta_rad ** 2) / (
        2 * sigma[1] ** 2
    )
    d = 1 / (2 * sigma[2] ** 2)

    return np.exp(
        -(
            a * np.square(xx - loc[0])
            + 2 * b * (xx - loc[0]) * (yy - loc[1])
            + c * np.square(yy - loc[0])
            + d * np.square(zz - loc[2])
        )
    )
