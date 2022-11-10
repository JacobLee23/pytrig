"""
Unit tests for :py:mod:`pytrig.trig`.
"""

import decimal

import pytest

from pytrig import trig
from pytrig.trig import PI
from pytrig.trig import INF, NINF, NAN
from pytrig.trig import PRECISION


decimal.getcontext().prec = PRECISION


D = decimal.Decimal


@pytest.mark.parametrize(
    "x, value", [
        (-2 * PI, 0),
        (-11 * PI / 6, 1 / D(2)), (-7 * PI / 4, D(2).sqrt() / 2), (-5 * PI / 3, D(3).sqrt() / 2),
        (-3 * PI / 2, 1),
        (-4 * PI / 3, D(3).sqrt() / 2), (-5 * PI / 4, D(2).sqrt() / 2), (-7 * PI / 6, 1 / D(2)),
        (-PI, 0),
        (-5 * PI / 6, -1 / D(2)), (-3 * PI / 4, -D(2).sqrt() / 2), (-2 * PI / 3, -D(3).sqrt() / 2),
        (-PI / 2, -1),
        (-PI / 3, -D(3).sqrt() / 2), (-PI / 4, -D(2).sqrt() / 2), (-PI / 6, -1 / D(2)),
        (D(0), 0),
        (PI / 6, 1 / D(2)), (PI / 4, D(2).sqrt() / 2), (PI / 3, D(3).sqrt() / 2),
        (PI / 2, 1),
        (2 * PI / 3, D(3).sqrt() / 2), (3 * PI / 4, D(2).sqrt() / 2), (5 * PI / 6, 1 / D(2)),
        (PI, 0),
        (7 * PI / 6, -1 / D(2)), (5 * PI / 4, -D(2).sqrt() / 2), (4 * PI / 3, -D(3).sqrt() / 2),
        (3 * PI / 2, -1),
        (5 * PI / 3, -D(3).sqrt() / 2), (7 * PI / 4, -D(2).sqrt() / 2), (11 * PI / 6, -1 / D(2)),
        (2 * PI, 0)
    ]
)
def test_sine(x: D, value: D):
    """

    :param x:
    :param value:
    :return:
    """
    res = trig.sine(x, PRECISION)
    assert abs(res - value) <= D(10) ** -(PRECISION - 1), (x, (res, value))


@pytest.mark.parametrize(
    "x, value", [
        (-2 * PI, 1),
        (-11 * PI / 6, D(3).sqrt() / 2), (-7 * PI / 4, D(2).sqrt() / 2), (-5 * PI / 3, 1 / D(2)),
        (-3 * PI / 2, D(0)),
        (-4 * PI / 3, -1 / D(2)), (-5 * PI / 4, -D(2).sqrt() / 2), (-7 * PI / 6, -D(3).sqrt() / 2),
        (-PI, -1),
        (-5 * PI / 6, -D(3).sqrt() / 2), (-3 * PI / 4, -D(2).sqrt() / 2), (-2 * PI / 3, -1 / D(2)),
        (-PI / 2, 0),
        (-PI / 3, 1 / D(2)), (-PI / 4, D(2).sqrt() / 2), (-PI / 6, D(3).sqrt() / D(2)),
        (D(0), 1),
        (PI / 6, D(3).sqrt() / 2), (PI / 4, D(2).sqrt() / 2), (PI / 3, 1 / D(2)),
        (PI / 2, 0),
        (2 * PI / 3, -1 / D(2)), (3 * PI / 4, -D(2).sqrt() / 2), (5 * PI / 6, -D(3).sqrt() / 2),
        (PI, -1),
        (7 * PI / 6, -D(3).sqrt() / 2), (5 * PI / 4, -D(2).sqrt() / 2), (4 * PI / 3, -1 / D(2)),
        (3 * PI / 2, 0),
        (5 * PI / 3, 1 / D(2)), (7 * PI / 4, D(2).sqrt() / 2), (11 * PI / 6, D(3).sqrt() / D(2)),
        (2 * PI, 1)
    ]
)
def test_cosine(x: D, value: D):
    """

    :param x:
    :param value:
    :return:
    """
    res = trig.cosine(x, PRECISION)
    assert abs(res - value) <= D(10) ** -(PRECISION - 1), (x, (res, value))


@pytest.mark.parametrize(
    "x, value", [
        (-2 * PI, 0),
        (-3 * PI / 2, NAN),
        (-PI, 0),
        (-PI / 2, NAN),
        (0, 0),
        (PI / 2, NAN),
        (PI, 0),
        (3 * PI / 2, NAN),
        (2 * PI, 0),
    ]
)
def test_tangent(x: D, value: D):
    """

    :param x:
    :param value:
    :return:
    """
    res = trig.tangent(x, PRECISION)
    if value is NAN:
        assert res is value
    else:
        assert abs(res - value) <= D(10) ** -(PRECISION - 1), (x, (res, value))


@pytest.mark.parametrize(
    "x, value", [
        (-2 * PI, 1),
        (-3 * PI / 2, NAN),
        (-PI, -1),
        (-PI / 2, NAN),
        (0, 1),
        (PI / 2, NAN),
        (PI, -1),
        (3 * PI / 2, NAN),
        (2 * PI, 1)
    ]
)
def test_secant(x: D, value: D):
    """

    :param x:
    :param value:
    :return:
    """
    res = trig.secant(x, PRECISION)
    if value is NAN:
        assert res is value
    else:
        assert abs(res - value) <= D(100) ** -(PRECISION - 1), (x, (res, value))


@pytest.mark.parametrize(
    "x, value", [
        (-2 * PI, NAN),
        (-3 * PI / 2, 1),
        (-PI, NAN),
        (-PI / 2, -1),
        (0, NAN),
        (PI / 2, 1),
        (PI, NAN),
        (3 * PI / 2, -1),
        (2 * PI, NAN)
    ]
)
def test_cosecant(x: D, value: D):
    """

    :param x:
    :param value:
    :return:
    """
    res = trig.cosecant(x, PRECISION)
    if value is NAN:
        assert res is value
    else:
        assert abs(res - value) <= D(100) ** -(PRECISION - 1), (x, (res, value))


@pytest.mark.parametrize(
    "x, value", [
        (-2 * PI, NAN),
        (-3 * PI / 2, 0),
        (-PI, NAN),
        (-PI / 2, 0),
        (0, NAN),
        (PI / 2, 0),
        (PI, NAN),
        (3 * PI / 2, 0),
        (2 * PI, NAN),
    ]
)
def test_cotangent(x: D, value: D):
    """

    :param x:
    :param value:
    :return:
    """
    res = trig.cotangent(x, PRECISION)
    if value is NAN:
        assert res is value
    else:
        assert abs(res - value) <= D(10) ** -(PRECISION - 1), (x, (res, value))
