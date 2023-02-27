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
import functools
from math import comb, factorial
import typing


# ------------------------------------------- Constants -------------------------------------------


INF = Decimal("Infinity")
NAN = Decimal("NaN")


# ------------------------------------- Computation Precision -------------------------------------


# Default number of decimal places of precision
PRECISION = 100


def precision(
        func: typing.Callable[[Decimal], Decimal]
    ) -> typing.Callable[[Decimal, int], Decimal]:
    """
    :param func:
    :return:
    """
    @functools.wraps(func)
    def wrapper(x: Decimal, prec: int = PRECISION) -> Decimal:
        """
        :param x:
        :param prec:
        :return:
        """
        with decimal.localcontext() as ctx:
            ctx.prec = prec + 4

            res = func(x)
            if res is NAN:
                return res

            return Decimal(res).quantize(Decimal(10) ** -prec).normalize()

    return wrapper


# --------------------------------------- Angle Conversions ----------------------------------------


@precision
def to_degrees(theta: Decimal) -> Decimal:
    r"""
    Converts an angle measure from radians to degrees.

    :param theta: The angle measure, in radians (:math:`rad`)
    :return: The angle measure, in degrees (:math:`^{\circ}`)
    """
    return theta * 180 / PI


@precision
def to_radians(theta: Decimal) -> Decimal:
    r"""
    Converts an angle measure from degrees to radians.

    :param theta: The angle measure, in degrees (:math:`^{\circ}`)
    :return: The angle measure, in radians (:math:`rad`)
    """
    return theta * PI / 180


# --------------------------------------- Pi Approximations ---------------------------------------


def chudnovsky_algorithm(prec: int = PRECISION) -> Decimal:
    r"""
    The Chudnovsky algorithm:

    .. math::

        \frac{1}{\pi} = 12 \sum_{q=0}^{\infty} \frac{{(-1)}^{q}(6q)!(545140134q+13591409)}{(3q)!{(q!)}^{3}{(640320)}^{3q+\frac{3}{2}}}

    `[11] <https://en.wikipedia.org/wiki/Chudnovsky_algorithm#Algorithm>`_.

    The above formula can be simplified to:

    .. math::

        \frac{{(640320)}^{\frac{3}{2}}}{12 \pi} = \frac{426880 \sqrt{10005}}{\pi} = \sum_{q=0}^{\infty} \frac{(6q)!(545140134q+13591409)}{(3q)!{(q!)}^{3}{(-262537412640768000)}^{q}}

    The result can then be generalized as the following:

    .. math::

        \pi = C{(\sum_{q=0}^{\infty} \frac{M_{q}L_{q}}{X_{q}})}^{-1}

    where:

    .. math::
        
        C = 426880 \sqrt{10005}

        M_{q} = \frac{(6q)!}{(3q)!{(q!)}^{3}}

        L_{q} = 545140134q + 13591409

        X_{q} = {(-262537412640768000)}^{q}

    This generalization is the method this function uses to compute approximations of :math:`\pi`.

    The time complexity of this algorithm is :math:`O(n{(\log n)}^{3})`
    `[12] <http://www.numberworld.org/y-cruncher/internals/formulas.html>`_.

    :param prec: The number of decimal places of precision
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


PI = chudnovsky_algorithm()


# ---------------------------------- Maclaurin Series Expansions ----------------------------------


def ms_natural_logarithm(n: int, x: Decimal) -> Decimal:
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
    :return: The ``n``-th term of the Maclaurin series for :math:`\ln(x)`, evaluated at ``x``
    """
    return Decimal(-1) ** (n + 2) / Decimal(n + 1) * Decimal((x - 1) ** (n + 1))


def ms_sine(n: int, x: Decimal) -> Decimal:
    r"""
    The Maclaurin series expansion for :math:`\sin(x)`:

    .. math::

        \sin(x) = \sum_{n=0}^{\infty} \frac{{(-1)}^{n}}{(2n+1)!} {x}^{2n+1}, -\infty < x < \infty

    `[2]`_ `[4]`_ `[7]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :return: The ``n``-th term of the Maclaurin series for :math:`\sin(x)`, evaluated at ``x``
    """
    return Decimal(-1) ** n / Decimal(factorial(2 * n + 1)) * Decimal(x ** (2 * n + 1))


def ms_cosine(n: int, x: Decimal) -> Decimal:
    r"""
    The Maclaurin series expansion for :math:`\cos(x)`:

    .. math::

        \cos(x) = \sum_{n=0}^{\infty} \frac{{(-1)}^{n}}{(2n)!} {x}^{2n}

    `[2]`_ `[4]`_ `[7]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :return: The ``n``-th term of the Maclaurin series for :math:`\cos(x)`, evaluated at ``x``
    """
    return Decimal(-1) ** n / Decimal(factorial(2 * n)) * Decimal(x ** (2 * n))


def ms_arcsine(n: int, x: Decimal) -> Decimal:
    r"""
    The Maclaurin series expansion for :math:`\arcsin(x)`:

    .. math::

        \arcsin(x) = \sum_{n=0}^{\infty} {(\frac{1}{4})}^{n} \binom{2n}{n} \frac{{x}^{2n+1}}{2n+1}

    `[2]`_ `[4]`_ `[8]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :return: The ``n``-th term of the Maclaurin series for :math:`\arcsin(x)`, evaluated at ``x``
    """
    return (1 / Decimal(4)) ** n * Decimal(comb(2 * n, n)) * (Decimal(x ** (2 * n + 1)) / Decimal(2 * n + 1))


def ms_arctangent(n: int, x: Decimal) -> Decimal:
    r"""
    The Maclaurin series expansion for :math:`\arctan(x)`:

    .. math::

        \arctan(x) = \sum_{n=0}^{\infty} \frac{{(-1)}^{n}}{2n+1} {x}^{2n+1}

    `[2]`_ `[4]`_ `[8]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :return: The ``n``-th term of the Maclaurin series for :math:`\arctan(x)`, evaluated at ``x``
    """
    return Decimal(-1) ** n / Decimal(2 * n + 1) * Decimal(x ** (2 * n + 1))


def ms_hyperbolic_sine(n: int, x: Decimal) -> Decimal:
    r"""
    The Maclaurin series expansion for :math:`\sinh(x)`:

    .. math::

        \sinh(x) = \sum_{n=0}^{\infty} \frac{{x}^{2n+1}}{(2n+1)!}

    `[2]`_ `[5]`_ `[9]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :return: The ``n``-th term of the Maclaurin series for :math:`\sinh(x)`, evaluated at ``x``
    """
    return Decimal(x ** (2 * n + 1)) / Decimal(factorial(2 * n + 1))


def ms_hyperbolic_cosine(n: int, x: Decimal) -> Decimal:
    r"""
    The Maclaurin series expansion for :math:`\cosh(x)`:

    .. math::

        \cosh(x) = \sum_{n=0}^{\infty} \frac{{x}^{2n}}{(2n)!}

    `[2]`_ `[5]`_ `[9]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :return: The ``n``-th term of the Maclaurin series for :math:`\cosh(x)`, evaluated at ``x``
    """
    return Decimal(x ** (2 * n)) / Decimal(factorial(2 * n))


def ms_hyperbolic_arcsine(n: int, x: Decimal) -> Decimal:
    r"""
    The Maclaurin series expansion for :math:`\operatorname{arsinh}(x)`:

    .. math::

        \operatorname{arsinh}(x) = \sum_{n=0}^{\infty} {(\frac{-1}{4})}^{n} \binom{2n}{n} \frac{{x}^{2n+1}}{2n+1}

    `[2]`_ `[5]`_ `[10]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :return: The ``n``-th term of the Maclaurin series for :math:`\operatorname{arsinh}(x)`, evaluated at ``x``
    """
    return (Decimal(-1) / Decimal(4)) ** n * Decimal(comb(2 * n, n)) * (Decimal(x ** (2 * n + 1)) / Decimal(2 * n + 1))


def ms_hyperbolic_arctangent(n: int, x: Decimal) -> Decimal:
    r"""
    The Maclaurin series expansion for :math:`\operatorname{artanh}(x)`:

    .. math::

        \operatorname{artanh}(x) = \sum_{n=0}^{\infty} \frac{x^{2n+1}}{2n+1}

    `[2]`_ `[5]`_ `[10]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :return: The ``n``-th term of the Maclaurin series for :math:`\operatorname{artanh}(x)`, evaluated at ``x``
    """
    return Decimal(x ** (2 * n + 1)) / Decimal(2 * n + 1)


class MaclaurinExpansion:
    """
    :param func:
    """
    def __init__(self, func: typing.Callable[[int, Decimal], Decimal]):
        self._func = func

    def __call__(self, x: Decimal) -> Decimal:
        return sum(self.expand(x))

    @property
    def func(self) -> typing.Callable[[int, Decimal], Decimal]:
        """
        A callable object that takes parameters ``n`` and ``x`` and returns the value of the
        :math:`n`-th term of the Maclaurin series expansion of the represented mathematical function
        evaluated at :math:`x`.
        """
        return self._func

    def expand(self, x: Decimal) -> typing.Generator[Decimal, None, None]:
        """
        Returns a generator of the values of the terms of the Maclaurin series expansion of a
        function evaluated at ``x``. The generator stops when the next term of the series
        approximately equals :math:`0` to the given number of decimal places of precision.

        :param x: The :math:`x`-value at which the Maclaurin series is evaluated
        :return: A generator of the values of the terms of the Maclaurin series expansion
        """
        n = 0
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


@precision
def natural_logarithm(x: Decimal) -> Decimal:
    r"""
    Let :math:`f(x)` be the Maclaurin series expansion (see :py:func:`ms_natural_logarithm`) of
    :math:`\ln(x)` evaluated at ``x``. This function approximates :math:`\ln(x)` using the
    following piecewise function:

    .. math::
        
        \ln(x) = \left \{
            \begin{array}{ll}
                f(x) & \quad 0 < x < 1

                0 & \quad x = 1

                f(\frac{1}{x}) & \quad x > 1
            \end{array}
        \right.

    `[] <https://mathworld.wolfram.com/NaturalLogarithm.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`ln`.

    :param x: The value at which to evaluate :math:`\ln(x)`
    :param prec: The number of decimal places of precision
    :return: The value of :math:`\ln(x)`, to ``prec`` decimal places of precision
    :rtype: decimal.Decimal
    :raise ValueError: The value of 'x' is outside the domain of :math:`\ln`(x)
    """
    if 0 < x < 1:
        return _natural_logarithm(x)
    if x == 1:
        return Decimal(0)
    if x > 1:
        return -natural_logarithm(1 / x)
    raise ValueError("domain error")


ln = natural_logarithm


# ------------------------------------ Trigonometric Functions ------------------------------------


@precision
def sine(x: Decimal) -> Decimal:
    r"""
    Evaluates :math:`\sin(x)` to ``prec`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/Sine.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`sin`.

    :param x: The value at which to evaluate :math:`\sin(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    """
    return _sine(x)


@precision
def cosine(x: Decimal) -> Decimal:
    r"""
    Evaluates :math:`\cos(x)` to ``prec`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/Cosine.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`cos`.

    :param x: The value at which to evaluate :math:`\cos(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    """
    return _cosine(x)


@precision
def tangent(x: Decimal) -> Decimal:
    r"""
    Evaluates :math:`\tan(x)` to ``prec`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/Tangent.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`tan`.

    :param x: The value at which to evaluate :math:`\tan(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    """
    try:
        return sine(x) / cosine(x)
    except ZeroDivisionError:
        return NAN


@precision
def secant(x: Decimal) -> Decimal:
    r"""
    Evaluates :math:`\sec(x)` to ``prec`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/Secant.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`sec`.

    :param x: The value at which to evaluate :math:`\sec(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    """
    try:
        return 1 / cosine(x)
    except ZeroDivisionError:
        return NAN


@precision
def cosecant(x: Decimal) -> Decimal:
    r"""
    Evaluates :math:`\csc(x)` to ``prec`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/Cosecant.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`csc`.

    :param x: The value at which to evaluate :math:`\csc(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    """
    try:
        return 1 / sine(x)
    except ZeroDivisionError:
        return NAN


@precision
def cotangent(x: Decimal) -> Decimal:
    r"""
    Evaluates :math:`\cot(x)` to ``prec`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/Cotangent.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`cot`.

    :param x: The value at which to evaluate :math:`\cot(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    """
    try:
        return cosine(x) / sine(x)
    except ZeroDivisionError:
        return NAN


# Shorthands for trigonometric functions
sin, cos, tan = sine, cosine, tangent
sec, csc, cot = secant, cosecant, cotangent


# -------------------------------- Inverse Trigonometric Functions --------------------------------


@precision
def arcsine(x: Decimal) -> Decimal:
    r"""
    Let :math:`f(x)` be the Maclaurin series expansion (see :py:func:`ms_arcsine`) of
    :math:`\arcsin(x)` evaluated at ``x``. This function approximates :math:`\arcsin(x)` using the
    following piecewise function:

    .. math::

        \arcsin(x) = \left \{
            \begin{array}{ll}
                -\frac{\pi}{2} & \quad x = -1

                f(x) & \quad x \in (-1, 1)

                \frac{\pi}{2} & \quad x = 1
            \end{array}
        \right.

    `[] <https://mathworld.wolfram.com/InverseSine.html>`_.

    Explicit values are returned for :math:`x \in \{-1, 1\}` because the Maclaurin series expansion
    of :math:`\arcsin(x)` is slow to converge for those values of :math:`x`.

    .. note::

        This function may be abbreviated to :py:func:`arcsin`.

    :param x: The value at which to evaluate :math:`\arcsin(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    :raise ValueError: ``x`` is outside the domain of :math:`\arcsin(x)`
    """
    if x == -1:
        return -PI / 2
    if -1 < x < 1:
        return _arcsine(x)
    if x == 1:
        return PI / 2
    raise ValueError("domain error")


@precision
def arccosine(x: Decimal) -> Decimal:
    r"""
    Let :math:`f(x)` be the Maclaurin series expansion (see :py:func:`ms_arcsine`) of
    :math:`\arcsin(x)` evaluated at ``x``. This function approximates :math:`\arccos(x)` using the
    following piecewise function:

    .. math::

        \arccos(x) = \left \{
            \begin{array}{ll}
                \pi & \quad x = -1

                f(x) & \quad x \in (-1, 1)

                0 & \quad x = 1
            \end{array}
        \right.

    `[] <https://mathworld.wolfram.com/InverseCosine.html>`_.

    Explicit values are returned for :math:`x \in \{-1, 1\}` because the Maclaurin series expansion
    of :math:`\arcsin(x)` is slow to converge for those values of :math:`x`.

    .. note::

        This function may be abbreviated to :py:func:`arccos`.

    :param x: The value at which to evaluate :math:`\arccos(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    :raise ValueError: ``x`` is outside the domain of :math:`\arccos(x)`
    """
    if x == -1:
        return PI
    if -1 < x < 1:
        return PI / 2 - _arcsine(x)
    if x == 1:
        return Decimal(0)
    raise ValueError("domain error")


@precision
def arctangent(x: Decimal) -> Decimal:
    r"""
    Let :math:`f(x)` be the Maclaurin series expansion (see :py:func:`ms_arctangent`) of
    :math:`\arctan(x)` evaluated at ``x``. This function approximates :math:`\arctan(x)` using the
    following piecewise function:

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

    `[] <https://mathworld.wolfram.com/InverseTangent.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`arctan`.
    
    :param x: The value at which to evaluate :math:`\arctan(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    """
    if x == -INF:
        return -PI / 2
    if x == INF:
        return PI / 2
    if -1 < x < 1:
        return _arctangent(x)
    return arcsine(x / (Decimal(1) + x ** 2).sqrt())


@precision
def arcsecant(x: Decimal) -> Decimal:
    r"""
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

    `[] <https://mathworld.wolfram.com/InverseSecant.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`arcsec`.

    :param x: The value at which to evaluate :math:`\operatorname{arcsec}(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{arcsec}(x)`
    """
    if x == -INF:
        return PI / 2
    if x == INF:
        return PI / 2
    if x <= -1 or x >= 1:
        return arccosine(1 / x)
    raise ValueError("domain error")


@precision
def arccosecant(x: Decimal) -> Decimal:
    r"""
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

    `[] <https://mathworld.wolfram.com/InverseCosecant.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`arccsc`.

    :param x: The value at which to evaluate :math:`\operatorname{arccsc}(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{arccsc}(x)`
    """
    if x == -INF:
        return Decimal(0)
    if x == INF:
        return Decimal(0)
    if x <= -1 or x >= 1:
        return arcsine(1 / x)
    raise ValueError("domain error")


@precision
def arccotangent(x: Decimal) -> Decimal:
    r"""
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

    `[] <https://mathworld.wolfram.com/InverseCotangent.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`arccot`.

    :param x: The value at which to evaluate :math:`\operatorname{arccot}(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    """
    if x == -INF:
        return PI
    if x == INF:
        return Decimal(0)
    return arctangent(1 / x)


# Shorthands for inverse trigonometric functions
arcsin, arccos, arctan = arcsine, arccosine, arctangent
arcsec, arccsc, arccot = arcsecant, arccosecant, arccotangent


# -------------------------------------- Hyperbolic Functions -------------------------------------


@precision
def hyperbolic_sine(x: Decimal) -> Decimal:
    r"""
    Evaluates :math:`\sinh(x)` to ``prec`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/HyperbolicSine.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`sinh`.

    :param x: The value at which to evaluate :math:`\sinh(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    """
    return _hyperbolic_sine(x)


@precision
def hyperbolic_cosine(x: Decimal) -> Decimal:
    r"""
    Evaluates :math:`\cosh(x)` to ``prec`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/HyperbolicCosine.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`cosh`.

    :param x: The value at which to evaluate :math:`\cosh(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    """
    return _hyperbolic_cosine(x)


@precision
def hyperbolic_tangent(x: Decimal) -> Decimal:
    r"""
    Evaluates :math:`\tanh(x)` to ``prec`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/HyperbolicTangent.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`tanh`.

    :param x: The value at which to evaluate :math:`\tanh(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    """
    return hyperbolic_sine(x) / hyperbolic_cosine(x)


@precision
def hyperbolic_secant(x: Decimal) -> Decimal:
    r"""
    Evaluates :math:`\operatorname{sech}(x)` to ``prec`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/HyperbolicSecant.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`sech`.

    :param x: The value at which to evaluate :math:`\operatorname{sech}(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    """
    return 1 / hyperbolic_cosine(x)


@precision
def hyperbolic_cosecant(x: Decimal) -> Decimal:
    r"""
    Evaluates :math:`\operatorname{csch}(x)` to ``prec`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/HyperbolicCosecant.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`csch`.

    :param x: The value at which to evaluate :math:`\operatorname{csch}(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    """
    try:
        return 1 / hyperbolic_sine(x)
    except ZeroDivisionError:
        return NAN


@precision
def hyperbolic_cotangent(x: Decimal) -> Decimal:
    r"""
    Evaluates :math:`\coth(x)` to ``prec`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/HyperbolicCotangent.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`coth`.

    :param x: The value at which to evaluate :math:`\coth(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    """
    try:
        return hyperbolic_cosine(x) / hyperbolic_sine(x)
    except ZeroDivisionError:
        return NAN


# Shorthands for hyperbolic functions
sinh, cosh, tanh = hyperbolic_sine, hyperbolic_cosine, hyperbolic_tangent
sech, csch, coth = hyperbolic_secant, hyperbolic_cosecant, hyperbolic_cotangent


# ---------------------------------- Inverse Hyperbolic Functions ----------------------------------


@precision
def hyperbolic_arcsine(x: Decimal) -> Decimal:
    r"""
    Evaluates :math:`\operatorname{arsinh}(x)` to ``prec`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/InverseHyperbolicSine.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`arsinh`.

    :param x: The value at which to evaluate :math:`\operatorname{arsinh}(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    """
    if abs(x) >= 0.95:
        return ln(x + Decimal(x ** 2 + 1).sqrt())
    return _hyperbolic_arcsine(x)


@precision
def hyperbolic_arccosine(x: Decimal) -> Decimal:
    r"""
    Evaluates :math:`\operatorname{arcosh}(x)` to ``prec`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/InverseHyperbolicCosine.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`arcosh`.
    
    :param x: The value at which to evaluate :math:`\operatorname{arcosh}(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{arcosh}(x)`
    """
    if not abs(x) >= 1:
        raise ValueError("domain error")
    if x ** 2 > 1.95:
        return ln(x + Decimal(x ** 2 - 1).sqrt())
    return hyperbolic_arcsine(x ** 2 - 1)


@precision
def hyperbolic_arctangent(x: Decimal) -> Decimal:
    r"""
    Evaluates :math:`\operatorname{artanh}(x)` to ``prec`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/InverseHyperbolicTangent.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`artanh`.
    
    :param x: The value at which to evaluate :math:`\operatorname{artanh}(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{artanh}(x)`
    """
    if not abs(x) < 1:
        raise ValueError("domain error")
    return _hyperbolic_arctangent(x)


@precision
def hyperbolic_arcsecant(x: Decimal) -> Decimal:
    r"""
    Evaluates :math:`\operatorname{arsech}(x)` to ``prec`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/InverseHyperbolicSecant.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`arsech`.
    
    :param x: The value at which to evaluate :math:`\operatorname{arsech}(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{arsech}(x)`
    """
    if not 0 < x <= 1:
        raise ValueError("domain error")
    return hyperbolic_cosine(1 / x)


@precision
def hyperbolic_arccosecant(x: Decimal) -> Decimal:
    r"""
    Evaluates :math:`\operatorname{arcsch}(x)` to ``prec`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/InverseHyperbolicCosecant.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`arcsch`.
    
    :param x: The value at which to evaluate :math:`\operatorname{arcsch}(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{arcsch}(x)`
    """
    try:
        return hyperbolic_arcsine(1 / x)
    except ZeroDivisionError:
        return NAN


@precision
def hyperbolic_arccotangent(x: Decimal) -> Decimal:
    r"""
    Evaluates :math:`\operatorname{arcoth}(x)` to ``prec`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/InverseHyperbolicCotangent.html>`_.

    .. note::

        This function may be abbreviated to :py:func:`arcoth`.
    
    :param x: The value at which to evaluate :math:`\operatorname{arcoth}(x)`
    :param prec: The number of decimal places of precision
    :return:
    :rtype: decimal.Decimal
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{arcoth}(x)`
    """
    if not abs(x) > 1:
        raise ValueError("domain error")
    return hyperbolic_tangent(1 / x)


# Shorthands for inverse hyperbolic functions
arsinh, arcosh, artanh = hyperbolic_arcsine, hyperbolic_arccosine, hyperbolic_arctangent
arsech, arcsch, arcoth = hyperbolic_arcsecant, hyperbolic_arccosecant, hyperbolic_arccotangent
