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
NAN = D("NaN")


def _precision(func: typing.Callable[[D, int], D]) -> typing.Callable:
    """

    :param func:
    :return:
    """
    def wrapper(x: D, prec: int = None) -> D:
        """

        :param x:
        :param prec:
        :return:
        """
        precision = (PRECISION if prec is None else prec)

        with decimal.localcontext() as ctx:
            ctx.prec = precision + 4

            res = func(x, ctx.prec)
            if res is NAN:
                return res
            return D(res).quantize(D(10) ** -precision).normalize()

    return wrapper


# ------------------------------------ Trigonometric Functions ------------------------------------


class UnitCircle:
    r"""
    `Unit Circle`_

    .. py:attribute:: angles

        A list of the 12 special unit circle angles.

        .. note::

            Angles are measured in radians.

    .. py:attribute:: _checks

        A list of functions that determines whether a given angle is a multiple of one of the 12
        special unit circle angles.

        .. note::

            Angles are measured in radians.

        Let:

        .. math::

            t \in \{
                0, \frac{\pi}{6}, \frac{\pi}{4}, \frac{\pi}{3},
                \frac{\pi}{2}, \frac{2\pi}{3}, \frac{3\pi}{4}, \frac{5\pi}{6},
                \pi, \frac{7\pi}{6}, \frac{5\pi}{4}, \frac{4\pi}{3},
                \frac{3\pi}{2}, \frac{5\pi}{3}, \frac{7\pi}{4}, \frac{11\pi}{6}
            \}

        Then, the 12 special unit circle angles take the following form:

        .. math::

            \theta = k(2\pi)+t, k \in \mathbb{Z}

        Thus, to determine if a given angle :math:`\theta \in \mathbb{R}` is one of the 12 special
        unit circle angles, we check whether there exists a value :math:`k \in \mathbb{Z}` that
        satisfies the above equation. Solving for :math:`k`:

        .. math::

            k = \frac{\theta-t}{2\pi}

        Therefore, if :math:`k = \frac{\theta-t}{2\pi} \in \mathbb{Z}` is satsified, then the
        angle :math:`\theta` is a special unit circle angle.

        Alternatively, if the remainder of the operation :math:`\frac{\theta-t}{2\pi}` is
        :math:`0`, then :math:`\theta-t` is a multiple of :math:`2\pi` and thus
        :math:`k \in \mathbb{Z}`.

        Each of the 12 functions in the list computes the remainder of
        :math:`\frac{\theta-t}{2\pi}` for each of the 12 values of :math:`t` defined above. Each
        function takes the form

        .. code-block:: python

            lambda x: (x - t) % (2 * PI)

        where :math:`t` is an element of :py:attr:`UnitCircle.angles`.
        If the function call evalutes to :math:`0`, then it can be reasonably concluded that ``x``
        is a multiple of one of the 12 special unit circle angles.

    The below code snippet should successfully execute, with no exceptions raised:

    .. code-block:: python

        for theta, func in zip(UnitCircle.angles, UnitCircle._checks):
            assert func(theta) == 0

    .. _Unit Circle: https://en.wikipedia.org/wiki/Unit_circle
    """
    _ucircle_angles: typing.List[typing.Callable[[], D]] = [
        lambda: 0,              # θ = 0
        lambda: PI / 6,         # θ = π/6
        lambda: PI / 4,         # θ = π/4
        lambda: PI / 3,         # θ = π/3
        lambda: PI / 2,         # θ = π/2
        lambda: 2 * PI / 3,     # θ = 2π/3
        lambda: 3 * PI / 4,     # θ = 3π/4
        lambda: 5 * PI / 6,     # θ = 5π/6
        lambda: PI,             # θ = π
        lambda: 7 * PI / 6,     # θ = 7π/6
        lambda: 5 * PI / 4,     # θ = 5π/4
        lambda: 4 * PI / 3,     # θ = 4π/3
        lambda: 3 * PI / 2,     # θ = 3π/2
        lambda: 5 * PI / 3,     # θ = 5π/3
        lambda: 7 * PI / 4,     # θ = 7π/4
        lambda: 11 * PI / 6     # θ = 11π/6
    ]

    @classmethod
    def angles(cls, prec: int, *, only_axes: bool = False) -> typing.List[D]:
        r"""
        .. note::

            Angles are measured in radians.

        :param prec: The desired number of decimal places of precision
        :param only_axes: If ``True``, only the axis angles are included in the returned list
        :return: The special unit circle angles
        """
        with decimal.localcontext() as ctx:
            ctx.prec = prec

            angles = [x() for x in cls._ucircle_angles]
            return angles[::4] if only_axes else angles

    @classmethod
    def check_angle(
            cls, x: D, values: typing.List[D], prec: int, *, only_axes: bool = False
    ) -> typing.Optional[D]:
        r"""
        Checks whether the angle ``x`` is a multiple of one of the 12 special unit circle angles.

        .. note::

            Angles are measured in radians.

        Let :math:`n` equal ``prec``. If ``x`` is within ``9*10^{-(n-1)}`` radians of a special
        unit circle angle :math:`\theta`, then the evaluation of the given trigonometric function
        at ``x`` will be evaluated to the value of the given trigonometric function at
        :math:`\theta`.

        In other words, if :math:`T(\theta)` is the given trigonometric function,
        :math:`f(x)=\frac{\theta-t}{2\pi}` (where :math:`t` is in
        :py:attr:`UnitCircle.angles`), and :math:`|f(x)|<9*10^{-(n-1)}`, then
        :math:`x` is considered equal to :math:`k(2\pi)+t` and :math:`f(x)=f(k(2\pi)+t)`.

        :param x: The angle to check (in radians)
        :param values: The values of the function at the corresponding special unit circle angles
        :param prec:
        :param only_axes:
        :return:
        """
        angles = cls.angles(prec, only_axes=only_axes)

        if len(values) > len(angles):
            raise ValueError(
                f"Argument 'values' contains {len(values) - len(angles)} too many values"
            )
        elif len(values) < len(angles):
            raise ValueError(
                f"Argument 'values' contains {len(angles) - len(values)} too few values"
            )

        for angle, val in zip(angles, values):
            if abs(
                    ((x - angle) % (2 * PI)).quantize(D(10) ** -(prec - 1))
            ) < D(10) ** -(prec - 2):
                return val


@_precision
def sine(x: D, prec: int) -> D:
    r"""

    :param x:
    :param prec:
    :return:
    """
    # Unit circle values for x ∈ [0, 2π)
    ucircle_values = [
        0,                                                  # +x
        1 / D(2), D(2).sqrt() / 2, D(3).sqrt() / 2,         # QI
        1,                                                  # +y
        D(3).sqrt() / 2, D(2).sqrt() / 2, 1 / D(2),         # QII
        0,                                                  # -x
        -1 / D(2), -D(2).sqrt() / 2, -D(3).sqrt() / 2,      # QIII
        -1,                                                 # -y
        -D(3).sqrt() / 2, -D(2).sqrt() / 2, -1 / D(2),      # QIV
    ]

    res = UnitCircle.check_angle(x, ucircle_values, prec)
    return sum(_sine(x)) if res is None else res


@_precision
def cosine(x: D, prec: int) -> D:
    r"""

    :param x:
    :param prec:
    :return:
    """
    # Unit circle values for x ∈ [0, 2π)
    ucircle_values = [
        1,                                                  # +x
        D(3).sqrt() / 2, D(2).sqrt() / 2, 1 / D(2),         # QI
        0,                                                  # +y
        -1 / D(2), -D(2).sqrt() / 2, -D(3).sqrt() / 2,      # QII
        -1,                                                 # -x
        -D(3).sqrt() / 2, -D(2).sqrt() / 2, -1 / D(2),      # QIII
        0,                                                  # -y
        1 / D(2), D(2).sqrt() / 2, D(3).sqrt() / 2,         # QIV
    ]

    res = UnitCircle.check_angle(x, ucircle_values, prec)
    return sum(_cosine(x)) if res is None else res


@_precision
def tangent(x: D, prec: int) -> D:
    r"""

    :param x:
    :param prec:
    :return:
    """
    ucircle_values = [
        0, NAN, 0, NAN
    ]
    try:
        res = UnitCircle.check_angle(x, ucircle_values, prec, only_axes=True)
        return sine(x, prec) / cosine(x, prec) if res is None else res
    except ZeroDivisionError:
        return NAN


@_precision
def secant(x: D, prec: int) -> D:
    r"""

    :param x:
    :param prec:
    :return:
    """
    ucircle_values = [
        1, NAN, -1, NAN
    ]
    try:
        res = UnitCircle.check_angle(x, ucircle_values, prec, only_axes=True)
        return 1 / cosine(x) if res is None else res
    except ZeroDivisionError:
        return NAN


@_precision
def cosecant(x: D, prec: int) -> D:
    r"""

    :param x:
    :param prec:
    :return:
    """
    ucircle_values = [
        NAN, 1, NAN, -1
    ]
    try:
        res = UnitCircle.check_angle(x, ucircle_values, prec, only_axes=True)
        return 1 / sine(x) if res is None else res
    except ZeroDivisionError:
        return NAN


@_precision
def cotangent(x: D, prec: int) -> D:
    r"""

    :param x:
    :param prec:
    :return:
    """
    ucircle_values = [
        NAN, 0, NAN, 0
    ]
    try:
        res = UnitCircle.check_angle(x, ucircle_values, prec, only_axes=True)
        return cosine(x) / sine(x) if res is None else res
    except ZeroDivisionError:
        return NAN


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
        return PI / 2
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
        return D(0)
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
        return NAN


@_precision
def hyperbolic_cotangent(x: D) -> D:
    r"""

    :param x:
    :return:
    """
    try:
        return hyperbolic_cosine(x) / hyperbolic_sine(x)
    except ZeroDivisionError:
        return NAN


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
