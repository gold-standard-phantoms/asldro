"""Tests for generate_numeric_function.py"""

import numpy as np
import numpy.testing
import pytest
from asldro.utils.generate_numeric_function import generate_gaussian_function


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
