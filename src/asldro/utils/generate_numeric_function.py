"""Functions to generate arrays of mathematical function values"""

from typing import Union
from copy import deepcopy
import pdb
import numpy as np

from asldro.utils.resampling import rot_z_mat, translate_mat

# A function for testing generate array functions
def generate_point_function(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    loc: Union[tuple, list, np.ndarray] = np.zeros((3,)),
) -> np.ndarray:
    """
    Generates an array where the element closest to the location defined
    by ``loc`` is 1.0. If ``loc`` is out of bounds then no point is created.
    Can be used with :func:`generate_circular_function_array`.

    
    :param xx: Array of x-coordinates, generate by meshgrid. Can be sparse.
    :type xx: np.ndarray
    :param yy: Array of y-coordinates, generate by meshgrid. Can be sparse.
    :type yy: np.ndarray
    :param zz: Array of z-coordinates, generate by meshgrid. Can be sparse.
    :type zz: np.ndarray
    :param loc: location of the point, :math:`[x_0, y_0, z_0]`, defaults to np.zeros((3,))
    :type loc: Union[tuple, list, np.ndarray], optional

    
    :return: An array with shape the result of broadcasting ``xx``, ``yy``, and ``zz``,
      where the element closes to the location defined by ``loc`` is 1.0, other elements are 0.0.
      If ``loc`` is out of bounds all elements are 0.0.
    :rtype: np.ndarray
    """
    out = np.zeros(np.broadcast(xx, yy, zz).shape)
    # check the values in loc are in the ranges supplied by xx, yy and zz, otherwise
    # return an array of zeros
    if (
        np.amin(xx) <= loc[0] <= np.amax(xx)
        and np.amin(yy) <= loc[1] <= np.amax(yy)
        and np.amin(zz) <= loc[2] <= np.amax(zz)
    ):
        idx = (np.abs(xx - loc[0])).argmin()
        idy = (np.abs(yy - loc[1])).argmin()
        idz = (np.abs(zz - loc[2])).argmin()

        out[idx, idy, idz] = 1.0
    return out


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
    Can be used with :func:`generate_circular_function_array`.

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

    :return: An array with shape the result of broadcasting ``xx``, ``yy``, and ``zz``, 
      values given by the 3D gaussian function.
    :rtype: np.ndarray

    **Equation**

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
    c = np.sin(theta_rad) ** 2 / (2 * sigma[0] ** 2) + np.cos(theta_rad) ** 2 / (
        2 * sigma[1] ** 2
    )
    d = 1 / (2 * sigma[2] ** 2)
    return np.exp(
        -(
            a * np.square(xx - loc[0])
            + 2 * b * (xx - loc[0]) * (yy - loc[1])
            + c * np.square(yy - loc[1])
            + d * np.square(zz - loc[2])
        )
    )


def generate_circular_function_array(
    func,
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    array_size: int,
    array_angular_increment: Union[float, int],
    func_params: dict,
    array_origin: Union[np.ndarray, list, tuple] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Produces a superposition of the supplied function in a circular array. The circular
    array's axis is about the z axis.

    :param func: The function to use to generate the circular array. This should accept
      arrays of x, y and z values of the entire domain to generate the function (i.e. 3d),
      as generated by np.meshgrid. It must also accept the argument ''loc'', which is the
      origin of the function, and ``theta``, which is the angle in degrees to rotate the
      function (mathematically) about an axis parallel to z that runs through ``loc``.
      An example function is  :func:`generate_point_function`.
    :type func: function
    :param xx: Array of x-coordinates, generate by meshgrid. Can be sparse.
    :type xx: np.ndarray
    :param yy: Array of y-coordinates, generate by meshgrid. Can be sparse.
    :type yy: np.ndarray
    :param zz: Array of z-coordinates, generate by meshgrid. Can be sparse.
    :type zz: np.ndarray
    :param array_size: The number of instances of ``func`` in the array.
    :type array_size: int
    :param array_angular_increment: The amount, in degrees, to increment the function each
      step.
    :type array_angular_increment: Union[float, int]
    :param func_params: A dictionary of function parameters. Must have entries:
    
      * 'loc': np.ndarray, list or tuple length 3, :math:`[x_0, y_0, z_0]` values.
    :type func_params: dict
    :param array_origin: The origin of the circular array, :math:`[x_{a,0}, y_{a,0}, z_{a,0}]`,
      defaults to (0.0, 0.0, 0.0)
    :type array_origin: Union[np.ndarray, list, tuple], optional

    :raises KeyError: If argument `func_params` does not have an entry 'loc'.
    :raises ValueError: If value of `func_params['loc']` is not a np.ndarray, list or tuple.
    :raises ValueError: If value of `func_params['loc']` is not 1D and of lenght 3

    :return: An array, comprising the function output arrayed at each position, normalised so 
      that its maximum value is 1.0
    :rtype: np.ndarray
    """
    if func_params.get("loc") is None:
        raise KeyError("dictionary `func_params` must have an entry 'loc'")
    elif isinstance(func_params["loc"], (np.ndarray, tuple, list)):

        if not (
            np.asarray(func_params["loc"]).size == 3
            and np.asarray(func_params["loc"]).ndim == 1
        ):
            raise ValueError(
                "value of `func_params['loc']` must be 1D and be of length 3"
            )
    else:
        raise ValueError(
            "value of `func_params['loc']` must be a np.ndarray, list or tuple"
        )

    array_origin: np.ndarray = np.asarray(array_origin)
    func_params = deepcopy(func_params)
    # create the function origin in homogeneous coords
    func_origin_homogeneous: np.ndarray = np.append(np.asarray(func_params["loc"]), 1)
    # create the output array, fill with zeroes
    out = np.zeros(np.broadcast(xx, yy, zz).shape)

    for i in range(array_size):
        func_params["loc"] = func_origin_homogeneous[:3]
        # add the function to out
        out += func(xx=xx, yy=yy, zz=zz, **func_params)
        # rotate the coordinates about the array origin
        func_origin_homogeneous = (
            translate_mat(-array_origin)
            @ rot_z_mat(array_angular_increment)
            @ translate_mat(array_origin)
            @ func_origin_homogeneous
        )

        # increment theta if present
        if func_params.get("theta") is not None:
            func_params["theta"] += array_angular_increment

    # normalise out by its maximum
    out /= np.amax(out)

    return out

