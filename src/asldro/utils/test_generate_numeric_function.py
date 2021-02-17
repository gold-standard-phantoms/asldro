"""Tests for generate_numeric_function.py"""


from typing import Union
import pytest

import numpy as np
import numpy.testing
from numpy.random import default_rng

from asldro.utils.generate_numeric_function import (
    generate_circular_function_array,
    generate_gaussian_function,
    generate_point_function,
)
from asldro.utils.resampling import rot_z_mat


def test_generate_point_function():
    x = np.arange(-10.0, 10.0, 0.1)
    y = np.arange(-10.0, 10.0, 0.1)
    z = np.arange(-10.0, 10.0, 0.1)
    xx, yy, zz = np.meshgrid(x, y, z, sparse=True)

    rng = default_rng()

    for i in range(10):
        loc = rng.uniform(-10, 10, (3,))
        out = generate_point_function(xx, yy, zz, loc)
        idx = (np.abs(xx - loc[0])).argmin()
        idy = (np.abs(yy - loc[1])).argmin()
        idz = (np.abs(zz - loc[2])).argmin()
        numpy.testing.assert_array_equal(out[idx, idy, idz], 1.0)
        # mask the element at idx, idy, idz
        out = np.ma.array(out, mask=False)
        out.mask[idx, idy, idz] = True
        assert np.all(out == 0)  # all remaining elements should be zero

    # try value out of range
    loc[0] = -11.0
    out = generate_point_function(xx, yy, zz, loc)
    assert np.all(out == 0)


def test_generate_gaussian_function():
    """Tests for generate_gaussian_function"""

    # input validation
    # xx, yy and zz have shapes that cannot be broadcast
    with pytest.raises(ValueError):
        generate_gaussian_function(
            xx=np.ones((3, 4, 5)), yy=np.ones((7, 5, 7)), zz=np.ones((4, 4, 4))
        )
    x = np.arange(-10.0, 10.0, 0.1)
    y = np.arange(-10.0, 10.0, 0.1)
    z = np.arange(-10.0, 10.0, 0.1)

    # check defaults with sparse grids
    xx, yy, zz = np.meshgrid(x, y, z, sparse=True)
    g_sparse = generate_gaussian_function(xx, yy, zz)

    # check defaults with normal grids
    xx, yy, zz = np.meshgrid(x, y, z, sparse=False)
    g_normal = generate_gaussian_function(xx, yy, zz)

    numpy.testing.assert_array_equal(g_sparse, g_normal)

    g_normal += generate_gaussian_function(
        xx, yy, zz, fwhm=[1.0, 5.0, 1.0], loc=[0.0, 5.0, 0.0], theta=45
    )

    # Check with loc as a tuple
    generate_gaussian_function(xx, yy, zz, loc=(0.1, 0.2, 0.3))

    # loc has length 4
    with pytest.raises(ValueError):
        generate_gaussian_function(xx, yy, zz, loc=(0.0, 0.0, 0.0, 0.0))

    # loc is 2D
    with pytest.raises(ValueError):
        generate_gaussian_function(xx, yy, zz, loc=np.zeros((3, 1)))

    # Check with fwhm as a tuple
    generate_gaussian_function(xx, yy, zz, fwhm=(1.0, 1.0, 1.0))

    # fwhm has negative values
    with pytest.raises(ValueError):
        generate_gaussian_function(xx, yy, zz, fwhm=(1.0, -1.0, 1.0))

    # fwhm has length 4
    with pytest.raises(ValueError):
        generate_gaussian_function(xx, yy, zz, fwhm=(1.0, 1.0, 1.0, 1.0))

    # fwhm is 2D
    with pytest.raises(ValueError):
        generate_gaussian_function(xx, yy, zz, fwhm=np.zeros((3, 2)))

    # theta is a str
    with pytest.raises(TypeError):
        generate_gaussian_function(xx, yy, zz, theta="90°")

    # Test a 1D example
    y = 0.0
    z = 0.0
    xx, yy, zz = np.meshgrid(x, y, z, sparse=True)
    g_1d = generate_gaussian_function(xx, yy, zz)

    numpy.testing.assert_array_almost_equal(
        g_1d, np.exp(-(xx ** 2) / (2 * (1 / (2 * np.sqrt(2 * np.log(2)))) ** 2))
    )


def test_generate_circular_function_array_with_gaussian():
    x = np.arange(-10.0, 10.0, 0.1)
    y = np.arange(-10.0, 10.0, 0.1)
    z = np.arange(-10.0, 10.0, 0.1)
    xx, yy, zz = np.meshgrid(x, y, z, sparse=True)
    gaussian_params = {"loc": [0.0, 5.0, 0.0], "theta": 0.0, "fwhm": [0.5, 0.5, 0.5]}
    out = generate_circular_function_array(
        func=generate_gaussian_function,
        xx=xx,
        yy=yy,
        zz=zz,
        array_origin=[0.0, 0.0, 0.0],
        array_size=4,
        array_angular_increment=45,
        func_params=gaussian_params,
    )
    # assert that the maximum value in the array is 1.0 - it has been normalised correctly.
    assert np.amax(out) == 1.0


def test_generate_circular_function_array_with_point():
    x = np.arange(-10.0, 10.0, 0.1)
    y = np.arange(-10.0, 10.0, 0.1)
    z = np.arange(-10.0, 10.0, 0.1)
    xx, yy, zz = np.meshgrid(x, y, z, sparse=True)

    point_params = {
        "loc": [5.0, 0.0, 0.0],
    }

    out = generate_circular_function_array(
        func=generate_point_function,
        xx=xx,
        yy=yy,
        zz=zz,
        array_origin=[0.0, 0.0, 0.0],
        array_size=8,
        array_angular_increment=45,
        func_params=point_params,
    )
    # calculate this example manually here
    test_array = np.zeros(np.broadcast(xx, yy, zz).shape)
    loc = np.array([5.0, 0.0, 0.0, 1.0])

    for i in range(8):
        test_array += generate_point_function(xx, yy, zz, loc[:3])
        loc = rot_z_mat(45) @ loc

    numpy.testing.assert_array_equal(out, test_array)
