"""

"""

from decimal import Decimal

from .constants import INF, NINF
from .maclaurin_series import sine as _sine
from .maclaurin_series import cosine as _cosine
from .maclaurin_series import arcsine as _arcsine
from .maclaurin_series import arctangent as _arctangent

# ------------------------------------ Trigonometric Functions -------------------------------------


def sine(x: Decimal) -> Decimal:
    r"""

    :param x:
    :return:
    """
    return sum(_sine(x))


def cosine(x: Decimal) -> Decimal:
    r"""

    :param x:
    :return:
    """
    return sum(_cosine(x))


def tangent(x: Decimal) -> Decimal:
    r"""

    :param x:
    :return:
    """
    try:
        return sine(x) / cosine(x)
    except ZeroDivisionError:
        return INF if sine(x) > 0 else NINF


def secant(x: Decimal) -> Decimal:
    r"""

    :param x:
    :return:
    """
    try:
        return 1 / cosine(x)
    except ZeroDivisionError:
        return INF if sine(x) > 0 else NINF


def cosecant(x: Decimal) -> Decimal:
    r"""

    :param x:
    :return:
    """
    try:
        return 1 / sine(x)
    except ZeroDivisionError:
        return INF if cosine(x) > 0 else NINF


def cotangent(x: Decimal) -> Decimal:
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


# -------------------------------- Inverse Trigonometric Functions ---------------------------------


def arcsine(x: Decimal) -> Decimal:
    r"""

    :param x:
    :return:
    """
    if not abs(x) <= 1:
        raise ValueError(
            "Value of argument 'x' is outside the domain of arcsin(x)"
        )

    if x == -1:
        return # -PI / 2
    elif x == 1:
        return # PI / 2
    else:
        return sum(_arcsine(x))


def arccosine(x: Decimal) -> Decimal:
    r"""

    :param x:
    :return:
    """
    if not abs(x) <= 1:
        raise ValueError(
            "Value of argument 'x' is outside the domain of arccos(x)"
        )

    if x == -1:
        return # PI
    elif x == 1:
        return Decimal(0)
    else:
        return # PI / 2 - arcsine(x)


def arctangent(x: Decimal) -> Decimal:
    r"""

    :param x:
    :return:
    """
    if x is NINF:
        return # -PI / 2
    elif x is INF:
        return # PI / 2
    elif -1 < x < 1:
        return sum(_arctangent(x))
    else:
        return arcsine(x / (Decimal(1) + x ** 2).sqrt())


def arcsecant(x: Decimal) -> Decimal:
    r"""

    :param x:
    :return:
    """
    if not abs(x) >= 1:
        raise ValueError(
            "Value of argument 'x' is outside the domain of arcsec(x)"
        )

    if x is NINF:
        return # -PI / 2
    elif x is INF:
        return # PI / 2
    else:
        return arccosine(1 / x)


def arccosecant(x: Decimal) -> Decimal:
    r"""

    :param x:
    :return:
    """
    if not abs(x) >= 1:
        raise ValueError(
            "Value of argument 'x' is outside the domain of arccsc(x)"
        )

    if x is NINF:
        return # PI
    elif x is INF:
        return Decimal(0)
    else:
        return arcsine(1 / x)


def arccotangent(x: Decimal) -> Decimal:
    r"""

    :param x:
    :return:
    """
    if x is NINF:
        return # PI
    elif x is INF:
        return Decimal(0)
    else:
        return arctangent(1 / x)


# Shorthands for inverse trigonometric functions
arcsin, arccos, arctan = arcsine, arccosine, arctangent
arcsec, arccsc, arccot = arcsecant, arccosecant, arccotangent
