r"""
**Pytrig** uses Maclaurin series expansions `[1]`_ `[2]`_ in conjunction with the
standard-library module :mod:`decimal` (which supports "correctly rounded decimal floating point
arithmetic") to rapidly approximate the trigonometric functions with high levels of precision. The
**pytrig** module offers higher levels of precision than does the :mod:`math` module:

.. doctest::
    :pyversion: >= 3.8

    >>> import math
    >>> import pytrig
    >>> math.pi
    3.141592653589793
    >>> pytrig.PI
    Decimal('3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706801')

The **pytrig** module supports computation of the following functions and classes of functions:

- Natural Logarithm `[3]`_ `[6]`_
- Trigonometric functions `[4]`_ `[7]`_
- Inverse trigonometric functions `[4]`_ `[8]`_
- Hyperbolic functions `[5]`_ `[9]`_
- Inverse hyperbolic functions `[5]`_ `[10]`_

.. _[1]: https://en.wikipedia.org/wiki/Taylor_series
.. _[2]: https://mathworld.wolfram.com/MaclaurinSeries.html
.. _[3]: https://en.wikipedia.org/wiki/Taylor_series#Natural_logarithm
.. _[4]: https://en.wikipedia.org/wiki/Taylor_series#Trigonometric_functions
.. _[5]: https://en.wikipedia.org/wiki/Taylor_series#Hyperbolic_functions
.. _[6]: https://en.wikipedia.org/wiki/Natural_logarithm#Series
.. _[7]: https://en.wikipedia.org/wiki/Trigonometric_functions#Power_series_expansion
.. _[8]: https://en.wikipedia.org/wiki/Inverse_trigonometric_functions#Infinite_series
.. _[9]: https://en.wikipedia.org/wiki/Hyperbolic_functions#Taylor_series_expressions
.. _[10]: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#Series_expansions
"""

import decimal
from decimal import Decimal
from math import comb, factorial
import typing


# ------------------------------------------- Constants -------------------------------------------


INF = Decimal("Infinity")
NAN = Decimal("NaN")


# ------------------------------------- Computation Precision -------------------------------------


# Default number of decimal places of precision
PRECISION = 100


# --------------------------------------- Angle Conversions ----------------------------------------


def to_degrees(theta: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Converts an angle measure from radians (:math:`rad`) to degrees (:math:`^{\circ}`).

    :param theta: The angle measure, in radians
    :param prec: The number of decimal places of precision
    :return: The angle measure, in degrees, to ``prec`` decimal places of precision
    """
    return theta * 180 / pi(prec)


def to_radians(theta: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Converts an angle measure from degrees (:math:`^{\circ}`) to radians (:math:`rad`).

    :param theta: The angle measure, in degrees
    :param prec: The number of decimal places of precision
    :return: The angle measure, in radians, to ``prec`` decimal places of precision
    """
    return theta * pi(prec) / 180


# --------------------------------------- Pi Approximations ---------------------------------------


def pi(prec: int = PRECISION) -> Decimal:
    r"""
    Uses the Chudnovsky algorithm to approximate the value of :math:`\pi` to ``prec`` decimal
    places of precision:

    .. math::

        \frac{1}{\pi}
        = 12 \sum_{q=0}^{\infty}
        \frac{{(-1)}^{q}(6q)!(545140134q+13591409)}{(3q)!{(q!)}^{3}{(640320)}^{3q+\frac{3}{2}}}

    `[11] <https://en.wikipedia.org/wiki/Chudnovsky_algorithm#Algorithm>`_.

    The above formula can be simplified to:

    .. math::

        \frac{{(640320)}^{\frac{3}{2}}}{12 \pi}
        = \frac{426880 \sqrt{10005}}{\pi}
        = \sum_{q=0}^{\infty}
        \frac{(6q)!(545140134q+13591409)}{(3q)!{(q!)}^{3}{(-262537412640768000)}^{q}}

    which can be generalized as the following:

    .. math::

        \pi = C{(\sum_{q=0}^{\infty} \frac{M_{q}L_{q}}{X_{q}})}^{-1}

    where:

    .. math::
        
        C = 426880 \sqrt{10005}

        M_{q} = \frac{(6q)!}{(3q)!{(q!)}^{3}}

        L_{q} = 545140134q + 13591409

        X_{q} = {(-262537412640768000)}^{q}

    This generalization is the method this function uses to compute approximations of :math:`\pi`.

    :param prec: The number of decimal places of precision
    :return: The value of :math:`\pi`, to ``prec`` decimal palces of precision
    """
    with decimal.localcontext() as ctx:
        ctx.prec = prec + 2

        # Initial conditions
        sum_ = Decimal(0)
        k: int = 0

        # Ramanujan-Sato series generalization
        term_c = 426880 * Decimal(10005).sqrt()

        def term_m(n: int) -> Decimal:
            return Decimal(factorial(6 * n)) / (
                Decimal(factorial(3 * n)) * Decimal(factorial(n)) ** 3
            )

        def term_l(n: int) -> Decimal:
            return Decimal(545140134 * n + 13591409)

        def term_x(n: int) -> Decimal:
            return Decimal(-262537412640768000) ** n

        while True:
            term = term_m(k) * term_l(k) / term_x(k)

            # Test for convergence
            if sum_ + term == sum_:
                return term_c * sum_ ** -1

            sum_ += term
            k += 1


PI = pi()


# ---------------------------------- Maclaurin Series Expansions ----------------------------------


def ms_natural_logarithm(n: int, x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    The Maclaurin series expansion for :math:`\ln(x)`:

    .. math::

        \ln(1+x) = \sum_{n=1}^{\infty} \frac{{(-1)}^{n+1}}{n} x^n, -1 < x \leq 1

    `[2]`_ `[3]`_ `[6]`_.

    .. note::

        The above formula can be rewritten as follows:

        .. math::

            \ln(x) = \sum_{n=0}^{\infty} \frac{{(-1)}^{n+2}}{n+1} {(x-1)}^{n+1}, 0 < x \leq 2

        This summation is 0-based and the series is evaluated at :math:`x` instead of :math:`1+x`.
        For these reasons, this alternative form of the series is used by this function.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :param prec: The number of decimal places of precision
    :return: The ``n``-th term of the Maclaurin series for :math:`\ln(x)`, evaluated at ``x``
    """
    with decimal.localcontext() as ctx:
        ctx.prec = prec + 4
        res = Decimal(-1) ** (n + 2) / Decimal(n + 1) * Decimal((x - 1) ** (n + 1))
        return res.quantize(Decimal(10) ** -prec).normalize()


def ms_sine(n: int, x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    The Maclaurin series expansion for :math:`\sin(x)`:

    .. math::

        \sin(x) = \sum_{n=0}^{\infty} \frac{{(-1)}^{n}}{(2n+1)!} {x}^{2n+1}, -\infty < x < \infty

    `[2]`_ `[4]`_ `[7]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :param prec: The number of decimal places of precision
    :return: The ``n``-th term of the Maclaurin series for :math:`\sin(x)`, evaluated at ``x``
    """
    with decimal.localcontext() as ctx:
        ctx.prec = prec + 4
        res = Decimal(-1) ** n / Decimal(factorial(2 * n + 1)) * Decimal(x ** (2 * n + 1))
        return res.quantize(Decimal(10) ** -prec).normalize()


def ms_cosine(n: int, x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    The Maclaurin series expansion for :math:`\cos(x)`:

    .. math::

        \cos(x) = \sum_{n=0}^{\infty} \frac{{(-1)}^{n}}{(2n)!} {x}^{2n}

    `[2]`_ `[4]`_ `[7]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :param prec: The number of decimal places of precision
    :return: The ``n``-th term of the Maclaurin series for :math:`\cos(x)`, evaluated at ``x``
    """
    with decimal.localcontext() as ctx:
        ctx.prec = prec + 4
        res = Decimal(-1) ** n / Decimal(factorial(2 * n)) * Decimal(x ** (2 * n))
        return res.quantize(Decimal(10) ** -prec).normalize()


def ms_arcsine(n: int, x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    The Maclaurin series expansion for :math:`\arcsin(x)`:

    .. math::

        \arcsin(x) = \sum_{n=0}^{\infty} {(\frac{1}{4})}^{n} \binom{2n}{n} \frac{{x}^{2n+1}}{2n+1}

    `[2]`_ `[4]`_ `[8]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :param prec: The number of decimal places of precision
    :return: The ``n``-th term of the Maclaurin series for :math:`\arcsin(x)`, evaluated at ``x``
    """
    with decimal.localcontext() as ctx:
        ctx.prec = prec + 4
        res = (1 / Decimal(4)) ** n * Decimal(comb(2 * n, n)) * (Decimal(x ** (2 * n + 1)) / Decimal(2 * n + 1))
        return res.quantize(Decimal(10) ** -prec).normalize()


def ms_arctangent(n: int, x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    The Maclaurin series expansion for :math:`\arctan(x)`:

    .. math::

        \arctan(x) = \sum_{n=0}^{\infty} \frac{{(-1)}^{n}}{2n+1} {x}^{2n+1}

    `[2]`_ `[4]`_ `[8]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :param prec: The number of decimal places of precision
    :return: The ``n``-th term of the Maclaurin series for :math:`\arctan(x)`, evaluated at ``x``
    """
    with decimal.localcontext() as ctx:
        ctx.prec = prec + 4
        res = Decimal(-1) ** n / Decimal(2 * n + 1) * Decimal(x ** (2 * n + 1))
        return res.quantize(Decimal(10) ** -prec).normalize()


def ms_hyperbolic_sine(n: int, x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    The Maclaurin series expansion for :math:`\sinh(x)`:

    .. math::

        \sinh(x) = \sum_{n=0}^{\infty} \frac{{x}^{2n+1}}{(2n+1)!}

    `[2]`_ `[5]`_ `[9]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :param prec: The number of decimal places of precision
    :return: The ``n``-th term of the Maclaurin series for :math:`\sinh(x)`, evaluated at ``x``
    """
    with decimal.localcontext() as ctx:
        ctx.prec = prec + 4
        res = Decimal(x ** (2 * n + 1)) / Decimal(factorial(2 * n + 1))
        return res.quantize(Decimal(10) ** -prec).normalize()


def ms_hyperbolic_cosine(n: int, x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    The Maclaurin series expansion for :math:`\cosh(x)`:

    .. math::

        \cosh(x) = \sum_{n=0}^{\infty} \frac{{x}^{2n}}{(2n)!}

    `[2]`_ `[5]`_ `[9]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :param prec: The number of decimal places of precision
    :return: The ``n``-th term of the Maclaurin series for :math:`\cosh(x)`, evaluated at ``x``
    """
    with decimal.localcontext() as ctx:
        ctx.prec = prec + 4
        res = Decimal(x ** (2 * n)) / Decimal(factorial(2 * n))
        return res.quantize(Decimal(10) ** -prec).normalize()


def ms_hyperbolic_arcsine(n: int, x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    The Maclaurin series expansion for :math:`\operatorname{arsinh}(x)`:

    .. math::

        \operatorname{arsinh}(x) = \sum_{n=0}^{\infty} {(-\frac{1}{4})}^{n} \binom{2n}{n} \frac{{x}^{2n+1}}{2n+1}

    `[2]`_ `[5]`_ `[10]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :param prec: The number of decimal places of precision
    :return: The ``n``-th term of the Maclaurin series for :math:`\operatorname{arsinh}(x)`, evaluated at ``x``
    """
    with decimal.localcontext() as ctx:
        ctx.prec = prec + 4
        res = (-1 / Decimal(4)) ** n * Decimal(comb(2 * n, n)) * (Decimal(x ** (2 * n + 1)) / Decimal(2 * n + 1))
        return res.quantize(Decimal(10) ** -prec).normalize()


def ms_hyperbolic_arctangent(n: int, x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    The Maclaurin series expansion for :math:`\operatorname{artanh}(x)`:

    .. math::

        \operatorname{artanh}(x) = \sum_{n=0}^{\infty} \frac{x^{2n+1}}{2n+1}

    `[2]`_ `[5]`_ `[10]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :param prec: The number of decimal places of precision
    :return: The ``n``-th term of the Maclaurin series for :math:`\operatorname{artanh}(x)`, evaluated at ``x``
    """
    with decimal.localcontext() as ctx:
        ctx.prec = prec + 4
        res = Decimal(x ** (2 * n + 1)) / Decimal(2 * n + 1)
        return res.quantize(Decimal(10) ** -prec).normalize()


class MaclaurinExpansion:
    """
    :param func: See :py:attr:`MaclaurinExpansion.func`

    .. py:attribute:: func

        A callable object that takes parameters ``n`` and ``x`` and returns the value of
        the :math:`n`-th term of the Maclaurin series expansion of the represented mathematical
        function, evaluated at :math:`x`.
    """
    def __init__(self, func: typing.Callable[[int, Decimal], Decimal]):
        self.func = func

    def __call__(self, x: Decimal, prec: int = PRECISION) -> Decimal:
        return sum(self.expand(x, prec))

    def expand(self, x: Decimal, prec: int = PRECISION) -> typing.Generator[Decimal, None, None]:
        """
        Returns a generator of the values of the terms of the Maclaurin series expansion of a
        function evaluated at ``x``. The generator stops when the next term of the series
        approximately equals :math:`0` to the given number of decimal places of precision.

        :param x: The :math:`x`-value at which the Maclaurin series is evaluated
        :param prec: The number of decimal places of precision
        :return: A generator of the values of the terms of the Maclaurin series expansion
        """
        n = 0

        with decimal.localcontext() as ctx:
            ctx.prec = prec

            while True:
                try:
                    term = self.func(n, x)
                except decimal.Overflow:
                    return

                if term + Decimal(1) == Decimal(1):
                    return

                yield term

                n += 1


_natural_logarithm = MaclaurinExpansion(ms_natural_logarithm)

_sine = MaclaurinExpansion(ms_sine)
_cosine = MaclaurinExpansion(ms_cosine)
_arcsine = MaclaurinExpansion(ms_arcsine)
_arctangent = MaclaurinExpansion(ms_arctangent)
_hyperbolic_sine = MaclaurinExpansion(ms_hyperbolic_sine)
_hyperbolic_cosine = MaclaurinExpansion(ms_hyperbolic_cosine)
_hyperbolic_arcsine = MaclaurinExpansion(ms_hyperbolic_arcsine)
_hyperbolic_arctangent = MaclaurinExpansion(ms_hyperbolic_arctangent)


# -------------------------------- Natural Logarithm Approximation --------------------------------


def natural_logarithm(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\ln(x), x \in (0, \infty)` to ``prec`` decimal places of precision.

    Let :math:`f(x)` be the Maclaurin series expansion (see :py:func:`ms_natural_logarithm`) of
    :math:`\ln(x)` evaluated at ``x``. This function approximates :math:`\ln(x)` using the
    following piecewise function:

    .. math::
        
        \ln(x) = \left \{
            \begin{array}{ll}
                f(x) & \quad 0 < x < 1

                0 & \quad x = 1

                -f(\frac{1}{x}) & \quad x > 1
            \end{array}
        \right.

    `[12] <https://mathworld.wolfram.com/NaturalLogarithm.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`ln`.

    :param x: The value at which to evaluate :math:`\ln(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\ln(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    :raise ValueError: The value of ``x`` is outside the domain of :math:`\ln`(x)
    """
    if 0 < x < 1:
        return _natural_logarithm(x, prec)
    if x == 1:
        return Decimal(0)
    if x > 1:
        return -natural_logarithm(1 / x, prec)
    raise ValueError("domain error")


ln = natural_logarithm


# ------------------------------------ Trigonometric Functions ------------------------------------


def sine(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\sin(x), x \in \mathbb{R}` to ``prec`` decimal places of precision.

    `[13] <https://mathworld.wolfram.com/Sine.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`sin`.

    :param x: The value at which to evaluate :math:`\sin(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\sin(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    """
    return _sine(x, prec)


def cosine(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\cos(x), x \in \mathbb{R}` to ``prec`` decimal places of precision.

    `[14] <https://mathworld.wolfram.com/Cosine.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`cos`.

    :param x: The value at which to evaluate :math:`\cos(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\cos(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    """
    return _cosine(x, prec)


def tangent(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\tan(x), x \in \{k | k \neq \frac{\pi}{2} + k \pi, k \in \mathbb{R}\}` to
    ``prec`` decimal places of precision.

    This function approximates :math:`\tan(x)` using the following definition:

    .. math::

        \tan(x) = \frac{\sin(x)}{\cos(x)}

    `[15] <https://mathworld.wolfram.com/Tangent.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`tan`.

    :param x: The value at which to evaluate :math:`\tan(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\tan(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    """
    try:
        return sine(x, prec) / cosine(x, prec)
    except ZeroDivisionError:
        return NAN


def secant(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\sec(x), x \in \{k | k \neq \frac{\pi}{2} + k \pi, k \in \mathbb{R}\}` to
    ``prec`` decimal places of precision.

    This function approximates :math:`\sec(x)` using the following definition:

    .. math::

        \sec(x) = {\cos(x)}^{-1}

    `[16] <https://mathworld.wolfram.com/Secant.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`sec`.

    :param x: The value at which to evaluate :math:`\sec(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\sec(x)`, to ``prec`` decimal palces of precision
    :rtype: decimal.Decimal
    """
    try:
        return cosine(x, prec) ** -1
    except ZeroDivisionError:
        return NAN


def cosecant(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\csc(x), x \in \{k | k \neq k \pi, k \in \mathbb{R}\}` to ``prec`` decimal
    places of precision.

    This function approximates :math:`\csc(x)` using the following definition:

    .. math::

        \csc(x) = {\sin(x)}^{-1}

    `[17] <https://mathworld.wolfram.com/Cosecant.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`csc`.

    :param x: The value at which to evaluate :math:`\csc(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\csc(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    """
    try:
        return sine(x, prec) ** -1
    except ZeroDivisionError:
        return NAN


def cotangent(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\cot(x), x \in \{k | k \neq k \pi, k \in \mathbb{R}\}` to ``prec`` decimal
    places of precision.

    This function approximates :math:`\cot(x)` using the following definition:

    .. math::

        \cot(x) = \frac{\cos(x)}{\sin(x)}

    `[18] <https://mathworld.wolfram.com/Cotangent.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`cot`.

    :param x: The value at which to evaluate :math:`\cot(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\cot(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    """
    try:
        return cosine(x, prec) / sine(x, prec)
    except ZeroDivisionError:
        return NAN


# Shorthands for trigonometric functions
sin, cos, tan = sine, cosine, tangent
sec, csc, cot = secant, cosecant, cotangent


# -------------------------------- Inverse Trigonometric Functions --------------------------------


def arcsine(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\arcsin(x), x \in [-1, 1]` to ``prec`` decimal places of precision.

    Let :math:`f(x)` be the Maclaurin series expansion of :math:`\arcsin(x)` (see
    :py:func:`ms_arcsine`) evaluated at ``x``. This function approximates :math:`\arcsin(x)`
    using the following piecewise function:

    .. math::

        \arcsin(x) = \left \{
            \begin{array}{ll}
                -\frac{\pi}{2} & \quad x = -1

                f(x) & \quad x \in (-1, 1)

                \frac{\pi}{2} & \quad x = 1
            \end{array}
        \right.

    `[19] <https://mathworld.wolfram.com/InverseSine.html>`_.

    Explicit values are returned for :math:`x \in \{-1, 1\}` because the Maclaurin series expansion
    of :math:`\arcsin(x)` is slow to converge for those values of :math:`x`.

    .. note::

        This function may be abbreviated to :py:func:`arcsin`.

    :param x: The value at which to evaluate :math:`\arcsin(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\arcsin(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    :raise ValueError: ``x`` is outside the domain of :math:`\arcsin(x)`
    """
    if x == -1:
        return -pi(prec) / 2
    if -1 < x < 1:
        return _arcsine(x, prec)
    if x == 1:
        return pi(prec) / 2
    raise ValueError("domain error")


def arccosine(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\arccos(x), x \in [-1, 1]` to ``prec`` decimal places of precision.

    Let :math:`f(x)` be the Maclaurin series expansion of :math:`\arcsin(x)` (see
    :py:func:`ms_arcsine`) evaluated at ``x``. This function approximates :math:`\arccos(x)`
    using the following piecewise function:

    .. math::

        \arccos(x) = \left \{
            \begin{array}{ll}
                \pi & \quad x = -1

                f(x) & \quad x \in (-1, 1)

                0 & \quad x = 1
            \end{array}
        \right.

    `[20] <https://mathworld.wolfram.com/InverseCosine.html>`_.

    Explicit values are returned for :math:`x \in \{-1, 1\}` because the Maclaurin series expansion
    of :math:`\arcsin(x)` is slow to converge for those values of :math:`x`.

    .. note::

        This function may be abbreviated to :py:func:`arccos`.

    :param x: The value at which to evaluate :math:`\arccos(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\arccos(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    :raise ValueError: ``x`` is outside the domain of :math:`\arccos(x)`
    """
    if x == -1:
        return pi(prec)
    if -1 < x < 1:
        return pi(prec) / 2 - _arcsine(x, prec)
    if x == 1:
        return Decimal(0)
    raise ValueError("domain error")


def arctangent(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\arctan(x), x \in \mathbb{R}` to ``prec`` decimal places of precision.

    Let :math:`f(x)` be the Maclaurin series expansion of :math:`\arctan(x)` (see
    :py:func:`ms_arctangent`) evaluated at ``x``. This function approximates :math:`\arctan(x)`
    using the following piecewise function:

    .. math::

        \arctan(x) = \left \{
            \begin{array}{ll}
                -\frac{\pi}{2} & \quad x = -\infty

                \arcsin(\frac{x}{\sqrt{1+{x}^{2}}}) & \quad x \in (-\infty -1]

                f(x) & \quad x \in (-1, 1)

                \arcsin(\frac{x}{\sqrt{1+{x}^{2}}}) & \quad x \in [1, \infty)
                
                \frac{\pi}{2} & \quad x = \infty
            \end{array}
        \right.

    `[21] <https://mathworld.wolfram.com/InverseTangent.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`arctan`.
    
    :param x: The value at which to evaluate :math:`\arctan(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\arctan(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    """
    if x == -INF:
        return -pi(prec) / 2
    if x == INF:
        return pi(prec) / 2
    if -1 < x < 1:
        return _arctangent(x, prec)
    return arcsine(x / (Decimal(1) + x ** 2).sqrt(), prec)


def arcsecant(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\operatorname{arcsec}(x), x \in \mathbb{R}` to ``prec`` decimal places of
    precision.

    This function approximates :math:`\operatorname{arcsec}(x)` using the following piecewise
    function:

    .. math::

        \operatorname{arcsec}(x) = \left \{
            \begin{array}{ll}
                \frac{\pi}{2} & \quad x = -\infty

                \arccos(\frac{1}{x}) & \quad x \in (-\infty, -1) \cup (1, \infty)

                \frac{\pi}{2} & \quad x = \infty
            \end{array}
        \right.

    `[22] <https://mathworld.wolfram.com/InverseSecant.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`arcsec`.

    :param x: The value at which to evaluate :math:`\operatorname{arcsec}(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\operatorname{arcsec}(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{arcsec}(x)`
    """
    if x == -INF:
        return pi(prec) / 2
    if x == INF:
        return pi(prec) / 2
    if x <= -1 or x >= 1:
        return arccosine(1 / x, prec)
    raise ValueError("domain error")


def arccosecant(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\operatorname{arccsc}(x), x \in \mathbb{R}` to ``prec`` decimal places of
    precision.

    This function approximates :math:`\operatorname{arccsc}(x)` using the following piecewise
    function:

    .. math::

        \operatorname{arccsc}(x) = \left \{
            \begin{array}{ll}
                0 & \quad x = -\infty

                \arcsin(\frac{1}{x}) & \quad x \in (-\infty, -1) \cup (1, \infty)

                0 & \quad x = \infty
            \end{array}
        \right.

    `[23] <https://mathworld.wolfram.com/InverseCosecant.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`arccsc`.

    :param x: The value at which to evaluate :math:`\operatorname{arccsc}(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\operatorname{arccsc}(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{arccsc}(x)`
    """
    if x == -INF:
        return Decimal(0)
    if x == INF:
        return Decimal(0)
    if x <= -1 or x >= 1:
        return arcsine(1 / x, prec)
    raise ValueError("domain error")


def arccotangent(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\operatorname{arccot}(x), x \in \mathbb{R}` to ``prec`` decimal places of
    precision.

    This function approximates :math:`\operatorname{arccot}(x)` using the following piecewise
    function:

    .. math::

        \operatorname{arccot}(x) = \left \{
            \begin{array}{ll}
                \pi & \quad x = -\infty

                \arctan(\frac{1}{x}) & \quad x \in (-\infty -1] \cup [1, \infty)

                0 & \quad x = \infty
            \end{array}
        \right.

    `[24] <https://mathworld.wolfram.com/InverseCotangent.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`arccot`.

    :param x: The value at which to evaluate :math:`\operatorname{arccot}(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\operatorname{arccot}(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    """
    if x == -INF:
        return pi(prec)
    if x == INF:
        return Decimal(0)
    return arctangent(1 / x, prec)


# Shorthands for inverse trigonometric functions
arcsin, arccos, arctan = arcsine, arccosine, arctangent
arcsec, arccsc, arccot = arcsecant, arccosecant, arccotangent


# -------------------------------------- Hyperbolic Functions -------------------------------------


def hyperbolic_sine(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\sinh(x), x \in \mathbb{R}` to ``prec`` decimal places of precision.

    `[25] <https://mathworld.wolfram.com/HyperbolicSine.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`sinh`.

    :param x: The value at which to evaluate :math:`\sinh(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\operatorname{sinh}(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    """
    return _hyperbolic_sine(x, prec)


def hyperbolic_cosine(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\cosh(x), x \in \mathbb{R}` to ``prec`` decimal places of precision.

    `[26] <https://mathworld.wolfram.com/HyperbolicCosine.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`cosh`.

    :param x: The value at which to evaluate :math:`\cosh(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\operatorname{cosh}(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    """
    return _hyperbolic_cosine(x, prec)


def hyperbolic_tangent(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\tanh(x), x \in \mathbb{R}` to ``prec`` decimal places of precision.

    This function approximates :math:`\tanh(x)` using the following definition:

    .. math::

        \tanh(x) = \frac{\sinh(x)}{\cosh(x)}.

    `[27] <https://mathworld.wolfram.com/HyperbolicTangent.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`tanh`.

    :param x: The value at which to evaluate :math:`\tanh(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\operatorname{tanh}(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    """
    return hyperbolic_sine(x, prec) / hyperbolic_cosine(x, prec)


def hyperbolic_secant(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\operatorname{sech}(x), x \in \mathbb{R}` to ``prec`` decimal places of
    precision.

    This function approximates :math:`\operatorname{sech}(x)` using the following definition:

    .. math::

        \operatorname{sech}(x) = {\cosh(x)}^{-1}.

    `[28] <https://mathworld.wolfram.com/HyperbolicSecant.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`sech`.

    :param x: The value at which to evaluate :math:`\operatorname{sech}(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\operatorname{sech}(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    """
    return hyperbolic_cosine(x, prec) ** -1


def hyperbolic_cosecant(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\operatorname{csch}(x), x \in (-\infty, 0) \cup (0, \infty)` to ``prec``
    decimal places of precision.

    This function approximates :math:`\operatorname{csch}(x)` using the following definition:

    .. math::

        \operatorname{csch}(x) = {\sinh(x)}^{-1}.

    `[29] <https://mathworld.wolfram.com/HyperbolicCosecant.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`csch`.

    :param x: The value at which to evaluate :math:`\operatorname{csch}(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\operatorname{csch}(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    """
    try:
        return hyperbolic_sine(x, prec) ** -1
    except ZeroDivisionError:
        return NAN


def hyperbolic_cotangent(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\coth(x), x \in (-\infty, 0) \cup (0, \infty)` to ``prec`` decimal places of
    precision.

    This function approximates :math:`\coth(x)` using the following definition:

    .. math::

        \coth(x) = \frac{\cosh(x)}{\sinh(x)}.

    `[30] <https://mathworld.wolfram.com/HyperbolicCotangent.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`coth`.

    :param x: The value at which to evaluate :math:`\coth(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\operatorname{coth}(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    """
    try:
        return hyperbolic_cosine(x, prec) / hyperbolic_sine(x, prec)
    except ZeroDivisionError:
        return NAN


# Shorthands for hyperbolic functions
sinh, cosh, tanh = hyperbolic_sine, hyperbolic_cosine, hyperbolic_tangent
sech, csch, coth = hyperbolic_secant, hyperbolic_cosecant, hyperbolic_cotangent


# ---------------------------------- Inverse Hyperbolic Functions ----------------------------------


def hyperbolic_arcsine(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\operatorname{arsinh}(x), x \in \mathbb{R}` to ``prec`` decimal places of
    precision.

    Let :math:`f(x)` be the Maclaurin series expansion of :math:`\operatorname{arsinh}(x)` (see
    :py:func:`ms_hyperbolic_arcsine`) evaluated at ``x``. This function approximates
    :math:`\operatorname{arsinh}(x)` using the following piecewise function:

    .. math::

        \operatorname{arsinh}(x) = \left \{
            \begin{array}{ll}
                \ln(x + \sqrt{{x}^{2} + 1}) & \quad x \in (-\infty, -0.95]

                f(x) & \quad x \in (-0.95, 0.95)

                \ln(x + \sqrt{{x}^{2} + 1}) & \quad x \in [0.95, \infty)
            \end{array}
        \right.

    `[31] <https://mathworld.wolfram.com/InverseHyperbolicSine.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`arsinh`.

    :param x: The value at which to evaluate :math:`\operatorname{arsinh}(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\operatorname{arsinh}(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    """
    if abs(x) >= 0.95:
        return ln(x + Decimal(x ** 2 + 1).sqrt(), prec)
    return _hyperbolic_arcsine(x)


def hyperbolic_arccosine(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\operatorname{arcosh}(x), x \in [1, \infty)` to ``prec`` decimal places of
    precision.

    This function approximates :math:`\operatorname{arsinh}(x)` using the following piecewise
    function:

    .. math::

        \operatorname{arcosh}(x) = \left \{
            \begin{array}{ll}
                \ln(x + \sqrt{{x}^{2} - 1)} & \quad x \in [1, \sqrt{1.95}]

                \operatorname{arsinh}(x) & \quad x \in (\sqrt{1.95}, \infty)
            \end{array}
        \right.

    `[32] <https://mathworld.wolfram.com/InverseHyperbolicCosine.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`arcosh`.
    
    :param x: The value at which to evaluate :math:`\operatorname{arcosh}(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\operatorname{arcosh}(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{arcosh}(x)`
    """
    if not x >= 1:
        raise ValueError("domain error")
    if x ** 2 > 1.95:
        return ln(x + Decimal(x ** 2 - 1).sqrt(), prec)
    return hyperbolic_arcsine(x ** 2 - 1, prec)


def hyperbolic_arctangent(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\operatorname{artanh}(x), x \in (-1, 1)` to ``prec`` decimal places of
    precision.

    `[33] <https://mathworld.wolfram.com/InverseHyperbolicTangent.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`artanh`.
    
    :param x: The value at which to evaluate :math:`\operatorname{artanh}(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\operatorname{artanh}(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{artanh}(x)`
    """
    if not abs(x) < 1:
        raise ValueError("domain error")
    return _hyperbolic_arctangent(x, prec)


def hyperbolic_arcsecant(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\operatorname{arsech}(x), x \in (0, 1]` to ``prec`` decimal places of
    precision.

    This function approximates :math:`\operatorname{arsech}(x)` using the following definition:

    .. math::

        \operatorname{arsech}(x) = \cosh(\frac{1}{x}).

    `[34] <https://mathworld.wolfram.com/InverseHyperbolicSecant.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`arsech`.
    
    :param x: The value at which to evaluate :math:`\operatorname{arsech}(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\operatorname{arsech}(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{arsech}(x)`
    """
    if not 0 < x <= 1:
        raise ValueError("domain error")
    return hyperbolic_cosine(1 / x, prec)


def hyperbolic_arccosecant(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\operatorname{arcsch}(x), x \in (-\infty, 0) \cup (0, \infty)` to ``prec``
    decimal places of precision.

    This function approximates :math:`\operatorname{arcsch}(x)` using the following definition:

    .. math::

        \operatorname{arcsch}(x) = \sinh(\frac{1}{x}).

    `[35] <https://mathworld.wolfram.com/InverseHyperbolicCosecant.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`arcsch`.
    
    :param x: The value at which to evaluate :math:`\operatorname{arcsch}(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\operatorname{arcsc}(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{arcsch}(x)`
    """
    try:
        return hyperbolic_arcsine(1 / x, prec)
    except ZeroDivisionError:
        return NAN


def hyperbolic_arccotangent(x: Decimal, prec: int = PRECISION) -> Decimal:
    r"""
    Evaluates :math:`\operatorname{arcoth}(x), x \in (-\infty, 1) \cup (1, \infty)` to ``prec``
    decimal places of precision.

    This function approximates :math:`\operatorname{arcoth}(x)` using the following definition:

    .. math::

        \operatorname{arcoth}(x) = \tanh(\frac{1}{x}).

    `[36] <https://mathworld.wolfram.com/InverseHyperbolicCotangent.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`arcoth`.
    
    :param x: The value at which to evaluate :math:`\operatorname{arcoth}(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\operatorname{arcoth}(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{arcoth}(x)`
    """
    if not abs(x) > 1:
        raise ValueError("domain error")
    return hyperbolic_tangent(1 / x, prec)


# Shorthands for inverse hyperbolic functions
arsinh, arcosh, artanh = hyperbolic_arcsine, hyperbolic_arccosine, hyperbolic_arctangent
arsech, arcsch, arcoth = hyperbolic_arcsecant, hyperbolic_arccosecant, hyperbolic_arccotangent
