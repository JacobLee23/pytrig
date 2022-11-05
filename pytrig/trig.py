"""

"""

import decimal
import typing

from . import pi
from ._precision import PRECISION
from .maclaurin_series import sine as _sine
from .maclaurin_series import cosine as _cosine
from .maclaurin_series import arcsine as _arcsine
from .maclaurin_series import arctangent as _arctangent
from .maclaurin_series import hyperbolic_sine as _hyperbolic_sine
from .maclaurin_series import hyperbolic_cosine as _hyperbolic_cosine
from .maclaurin_series import hyperbolic_arcsine as _hyperbolic_arcsine
from .maclaurin_series import hyperbolic_arctangent as _hyperbolic_arctangent


D = decimal.Decimal

PI = pi.chudnovsky_algorithm()
INF = D("Infinity")
NINF = D("-Infinity")


def _precision(func: typing.Callable[[D], D]) -> typing.Callable:
    """

    :param func:
    :return:
    """
    def wrapper(x: D, precision: int = PRECISION) -> D:
        """

        :param x:
        :param precision:
        :return:
        """
        with decimal.localcontext() as ctx:
            ctx.prec = precision + 3

            res = func(x)

        with decimal.localcontext() as ctx:
            ctx.prec = precision + 1

            return +res

    return wrapper


# ------------------------------------ Trigonometric Functions ------------------------------------


@_precision
def sine(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    # x = k(π) - π/2 => sin(x) = -1
    if ((x + PI / 2) / PI) % 1 == 0:
        return -D(1)
    # x = k(π) - π/3 => sin(x) = -√(3)/2
    elif ((x + PI / 3) / PI) % 1 == 0:
        return -D(3).sqrt() / 2
    # x = k(π) - π/4 => sin(x) = -√(2)/2
    elif ((x + PI / 4) / PI) % 1 == 0:
        return -D(2).sqrt() / 2
    # x = k(π) - π/6 => sin(x) = -1/2
    elif ((x + PI / 6) / PI) % 1 == 0:
        return -1 / D(2)
    # x = k(π) => sin(x) = 0
    elif (x / PI) % 1 == 0:
        return D(0)
    # x = k(π) + π/6 => sin(x) = 1/2
    elif ((x - PI / 6) / PI) % 1 == 0:
        return 1 / D(2)
    # x = k(π) + π/4 => sin(x) = √(2)/2
    elif ((x - PI / 4) / PI) % 1 == 0:
        return D(2).sqrt() / 2
    # x = k(π) + π/3 => sin(x) = √(3)/2
    elif ((x - PI / 3) / PI) % 1 == 0:
        return D(3).sqrt() / 2
    # x = k(π) + π/2 => sin(x) = 1
    elif ((x - PI / 2) / PI) % 1 == 0:
        return D(1)
    else:
        return sum(_sine(x))


@_precision
def cosine(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    return sum(_cosine(x))


@_precision
def tangent(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    try:
        return sine(x) / cosine(x)
    except ZeroDivisionError:
        return INF if sine(x) > 0 else NINF


@_precision
def secant(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    try:
        return 1 / cosine(x)
    except ZeroDivisionError:
        return INF if sine(x) > 0 else NINF


@_precision
def cosecant(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    try:
        return 1 / sine(x)
    except ZeroDivisionError:
        return INF if cosine(x) > 0 else NINF


@_precision
def cotangent(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    try:
        return cosine(x) / sine(x)
    except ZeroDivisionError:
        return INF if cosine(x) > 0 else NINF


# Shorthands for trigonometric functions
sin, cos, tan = sine, cosine, tangent
sec, csc, cot = secant, cosecant, cotangent


# -------------------------------- Inverse Trigonometric Functions --------------------------------


@_precision
def arcsine(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    if not abs(x) <= 1:
        raise ValueError(
            "Value of argument 'x' is outside the domain of arcsin(x)"
        )

    if x == -1:
        return -PI / 2
    elif x == 1:
        return PI / 2
    else:
        return sum(_arcsine(x))


@_precision
def arccosine(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    if not abs(x) <= 1:
        raise ValueError(
            "Value of argument 'x' is outside the domain of arccos(x)"
        )

    if x == -1:
        return PI
    elif x == 1:
        return D(0)
    else:
        return PI / 2 - arcsine(x)


@_precision
def arctangent(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    if x is NINF:
        return -PI / 2
    elif x is INF:
        return PI / 2
    elif -1 < x < 1:
        return sum(_arctangent(x))
    else:
        return arcsine(x / (D(1) + x ** 2).sqrt())


@_precision
def arcsecant(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    if not abs(x) >= 1:
        raise ValueError(
            "Value of argument 'x' is outside the domain of arcsec(x)"
        )

    if x is NINF:
        return -PI / 2
    elif x is INF:
        return PI / 2
    else:
        return arccosine(1 / x)


@_precision
def arccosecant(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    if not abs(x) >= 1:
        raise ValueError(
            "Value of argument 'x' is outside the domain of arccsc(x)"
        )

    if x is NINF:
        return PI
    elif x is INF:
        return D(0)
    else:
        return arcsine(1 / x)


@_precision
def arccotangent(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    if x is NINF:
        return PI
    elif x is INF:
        return D(0)
    else:
        return arctangent(1 / x)


# Shorthands for inverse trigonometric functions
arcsin, arccos, arctan = arcsine, arccosine, arctangent
arcsec, arccsc, arccot = arcsecant, arccosecant, arccotangent


# -------------------------------------- Hyperbolic Functions -------------------------------------

@_precision
def hyperbolic_sine(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    return sum(_hyperbolic_sine(x))


@_precision
def hyperbolic_cosine(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    return sum(_hyperbolic_cosine(x))


@_precision
def hyperbolic_tangent(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    return hyperbolic_sine(x) / hyperbolic_cosine(x)


@_precision
def hyperbolic_secant(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    return 1 / hyperbolic_cosine(x)


@_precision
def hyperbolic_cosecant(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    try:
        return 1 / hyperbolic_sine(x)
    except ZeroDivisionError:
        return INF if hyperbolic_sine(x) > 0 else NINF


@_precision
def hyperbolic_cotangent(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    try:
        return hyperbolic_cosine(x) / hyperbolic_sine(x)
    except ZeroDivisionError:
        return INF if hyperbolic_sine(x) > 0 else NINF


sinh, cosh, tanh = hyperbolic_sine, hyperbolic_cosine, hyperbolic_tangent
sech, csch, coth = hyperbolic_secant, hyperbolic_cosecant, hyperbolic_cotangent


# ---------------------------------- Inverse Hyperbolic Functions ----------------------------------

@_precision
def hyperbolic_arcsine(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    return sum(_hyperbolic_arcsine(x))


@_precision
def hyperbolic_arccosine(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    if not abs(x) > 1:
        raise ValueError(
            "Value of argument 'x' is outside the domain of arcosh(x)"
        )
    return hyperbolic_arcsine(x ** 2 - 1)


@_precision
def hyperbolic_arctangent(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    if not abs(x) < 1:
        raise ValueError(
            "Value of argument 'x' is outside the domain of artanh(x)"
        )
    return sum(_hyperbolic_arctangent(x))


@_precision
def hyperbolic_arcsecant(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    if not 0 < x <= 1:
        raise ValueError(
            "Value of argument 'x' is outside the domain of arsech(x)"
        )
    return hyperbolic_cosine(1 / x)


@_precision
def hyperbolic_arccosecant(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    if not abs(x) > 1:
        raise ValueError(
            "Value of argument 'x' is outside the domain of arcsch(x)"
        )
    return hyperbolic_arcsine(1 / x)


@_precision
def hyperbolic_arccotangent(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    if not abs(x) > 1:
        raise ValueError(
            "Value of argument 'x' is outside the domain of artanh(x)"
        )
    return hyperbolic_tangent(1 / x)


arsinh, arcosh, artanh = hyperbolic_arcsine, hyperbolic_arccosine, hyperbolic_arctangent
arsech, arcsch, arcoth = hyperbolic_arcsecant, hyperbolic_arccosecant, hyperbolic_arccotangent
