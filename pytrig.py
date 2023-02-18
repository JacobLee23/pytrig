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
import functools
from math import comb, factorial
import typing


D = decimal.Decimal


INF = D("Infinity")
NAN = D("NaN")


# ------------------------------------- Computation Precision -------------------------------------


# Default number of decimal places of precision
PRECISION = 100


def precision(func: typing.Callable[[D], D]) -> typing.Callable[[D, int], D]:
    """
    :param func:
    :return:
    """
    @functools.wraps(func)
    def wrapper(x: D, prec: int = PRECISION) -> D:
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

            return D(res).quantize(D(10) ** -prec).normalize()

    return wrapper


# --------------------------------------- Pi Approximations ---------------------------------------


def chudnovsky_algorithm(prec: int = PRECISION) -> D:
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
        sum_ = D(0)
        k: int = 0

        # Ramanujan-Sato series generalization
        c = 426880 * D(10005).sqrt()
        m = lambda n: D(factorial(6 * n)) / (D(factorial(3 * n)) * D(factorial(n)) ** 3)
        l = lambda n: D(545140134 * n + 13591409)
        x = lambda n: D(-262537412640768000) ** n

        while True:
            term = m(k) * l(k) / x(k)

            # Test for convergence
            if sum_ + term == sum_:
                return c * sum_ ** -1

            sum_ += term
            k += 1


PI = chudnovsky_algorithm()


# ---------------------------------- Maclaurin Series Expansions ----------------------------------


def ms_natural_logarithm(n: int, x: D) -> D:
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
    return D(-1) ** (n + 2) / D(n + 1) * D((x - 1) ** (n + 1))


def ms_sine(n: int, x: D) -> D:
    r"""
    The Maclaurin series expansion for :math:`\sin(x)`:

    .. math::

        \sin(x) = \sum_{n=0}^{\infty} \frac{{(-1)}^{n}}{(2n+1)!} {x}^{2n+1}, -\infty < x < \infty

    `[2]`_ `[4]`_ `[7]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :return: The ``n``-th term of the Maclaurin series for :math:`\sin(x)`, evaluated at ``x``
    """
    return D(-1) ** n / D(factorial(2 * n + 1)) * D(x ** (2 * n + 1))


def ms_cosine(n: int, x: D) -> D:
    r"""
    The Maclaurin series expansion for :math:`\cos(x)`:

    .. math::

        \cos(x) = \sum_{n=0}^{\infty} \frac{{(-1)}^{n}}{(2n)!} {x}^{2n}

    `[2]`_ `[4]`_ `[7]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :return: The ``n``-th term of the Maclaurin series for :math:`\cos(x)`, evaluated at ``x``
    """
    return D(-1) ** n / D(factorial(2 * n)) * D(x ** (2 * n))


def ms_arcsine(n: int, x: D) -> D:
    r"""
    The Maclaurin series expansion for :math:`\arcsin(x)`:

    .. math::

        \arcsin(x) = \sum_{n=0}^{\infty} {(\frac{1}{4})}^{n} \binom{2n}{n} \frac{{x}^{2n+1}}{2n+1}

    `[2]`_ `[4]`_ `[8]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :return: The ``n``-th term of the Maclaurin series for :math:`\arcsin(x)`, evaluated at ``x``
    """
    return (1 / D(4)) ** n * D(comb(2 * n, n)) * (D(x ** (2 * n + 1)) / D(2 * n + 1))


def ms_arctangent(n: int, x: D) -> D:
    r"""
    The Maclaurin series expansion for :math:`\arctan(x)`:

    .. math::

        \arctan(x) = \sum_{n=0}^{\infty} \frac{{(-1)}^{n}}{2n+1} {x}^{2n+1}

    `[2]`_ `[4]`_ `[8]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :return: The ``n``-th term of the Maclaurin series for :math:`\arctan(x)`, evaluated at ``x``
    """
    return D(-1) ** n / D(2 * n + 1) * D(x ** (2 * n + 1))


def ms_hyperbolic_sine(n: int, x: D) -> D:
    r"""
    The Maclaurin series expansion for :math:`\sinh(x)`:

    .. math::

        \sinh(x) = \sum_{n=0}^{\infty} \frac{{x}^{2n+1}}{(2n+1)!}

    `[2]`_ `[5]`_ `[9]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :return: The ``n``-th term of the Maclaurin series for :math:`\sinh(x)`, evaluated at ``x``
    """
    return D(x ** (2 * n + 1)) / D(factorial(2 * n + 1))


def ms_hyperbolic_cosine(n: int, x: D) -> D:
    r"""
    The Maclaurin series expansion for :math:`\cosh(x)`:

    .. math::

        \cosh(x) = \sum_{n=0}^{\infty} \frac{{x}^{2n}}{(2n)!}

    `[2]`_ `[5]`_ `[9]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :return: The ``n``-th term of the Maclaurin series for :math:`\cosh(x)`, evaluated at ``x``
    """
    return D(x ** (2 * n)) / D(factorial(2 * n))


def ms_hyperbolic_arcsine(n: int, x: D) -> D:
    r"""
    The Maclaurin series expansion for :math:`\arsinh(x)`:

    .. math::

        \operatorname{arsinh}(x) = \sum_{n=0}^{\infty} {(\frac{-1}{4})}^{n} \binom{2n}{n} \frac{{x}^{2n+1}}{2n+1}

    `[2]`_ `[5]`_ `[10]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :return: The ``n``-th term of the Maclaurin series for :math:`\operatorname{arsinh}(x)`, evaluated at ``x``
    """
    return (D(-1) / D(4)) ** n * D(comb(2 * n, n)) * (D(x ** (2 * n + 1)) / D(2 * n + 1))


def ms_hyperbolic_arctangent(n: int, x: D) -> D:
    r"""
    The Maclaurin series expansion for :math:`\operatorname{artanh}(x)`:

    .. math::

        \operatorname{artanh}(x) = \sum_{n=0}^{\infty} \frac{x^{2n+1}}{2n+1}

    `[2]`_ `[5]`_ `[10]`_.

    :param n: The 0-based index of the series term to compute
    :param x: The value at which to evaluate the series term
    :return: The ``n``-th term of the Maclaurin series for :math:`\operatorname{artanh}(x)`, evaluated at ``x``
    """
    return D(x ** (2 * n + 1)) / D(2 * n + 1)


class MaclaurinExpansion:
    """
    :param func:
    """
    def __init__(self, func: typing.Callable[[int, D], D]):
        self._func = func

    def __call__(self, x: D) -> D:
        return sum(self.expand(x))

    @property
    def func(self) -> typing.Callable[[int, D], D]:
        """
        A callable object that takes parameters ``n`` and ``x`` and returns the value of the
        :math:`n`-th term of the Maclaurin series expansion of the represented mathematical function
        evaluated at :math:`x`.
        """
        return self._func

    def expand(self, x: D) -> typing.Generator[D, None, None]:
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

            if term + D(1) == D(1):
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
def natural_logarithm(x: D) -> D:
    r"""
    Evaluates :math:`\ln(x)` to ``n`` decimal places of precision.

    :param x:
    :param n:
    :raise ValueError: The value of 'x' is outside the domain of ln(x)
    """
    if not x > 0:
        raise ValueError("domain error")

    return _natural_logarithm(x) if 0 < x < 1 else -natural_logarithm(1 / x)


ln = natural_logarithm


# ------------------------------------- Unit Circle Evaluation ------------------------------------


class UnitCircle:
    r"""
    .. code-block:: python

        >>> def func(x: D) -> D:
        ...     ...
        ...
        >>> ucircle = UnitCircle(
        ...     axis_values={
        ...         "posx": f(0), "negx": f(PI), "posy": f(PI / 2), "negy": f(3 * PI / 2)
        ...     },
        ...     quadrant_values = {
        ...         "q1": (f(PI / 6), f(PI / 4), f(PI / 3)),
        ...         "q2": (f(2 * PI / 3), f(3 * PI / 4), f(5 * PI / 6)),
        ...         "q3": (f(7 * PI / 6), f(5 * PI / 4), f(4 * PI / 3)),
        ...         "q4": (f(5 * PI / 3), f(7 * PI / 4), f(11 * PI / 6)),
        ...     }
        ... )

    :param axis_values: The values of the trigonometric function along the +x, +y, -x, and -y axes
    :param quadrant_values: The value of the trigonometric functions in QI, QII, QIII, and QIV
    """
    keys = ("posx", "negx", "posy", "negy", "q1", "q2", "q3", "q4")

    def __init__(
        self, *,
        axis_values: typing.Optional[typing.Dict[str, D]] = None,
        quadrant_values: typing.Optional[typing.Dict[str, typing.Tuple[D]]] = None
    ):
        if axis_values is not None:
            assert len(axis_values) == 4
        self._axis_values = axis_values
        
        if quadrant_values is not None:
            assert len(quadrant_values) == 4
            assert all(len(v) == 3 for v in quadrant_values.values())
        self._quadrant_values = quadrant_values

        self._angles = (
            self["posx"], *self["q1"], self["posy"], *self["q2"],
            self["negx"], *self["q3"], self["negy"], *self["q4"]
        )
        self._axes = {x: self[x] for x in ("posx", "negx", "posy", "negy")}
        self._quadrants = {x: self[x] for x in ("q1", "q2", "q3", "q4")}
        self._ucircle_angles = {**self.axes, **self.quadrants}

    def __getitem__(self, key: str) -> typing.Union[D, typing.Tuple[D]]:
        if key == "posx":
            return 0
        elif key == "q1":
            return PI / 6, PI / 4, PI / 3
        elif key == "posy":
            return PI / 2
        elif key == "q2":
            return 2 * PI / 3, 3 * PI / 4, 5 * PI / 6
        elif key == "negx":
            return PI
        elif key == "q3":
            return 7 * PI / 6, 5 * PI / 4, 4 * PI / 3
        elif key == "negy":
            return 3 * PI / 2
        elif key == "q4":
            return 5 * PI / 3, 7 * PI / 4, 11 * PI / 6
        else:
            raise KeyError(f"{key} not in {self.keys}")

    @property
    def angles(self) -> typing.Tuple[D]:
        """
        The 12 unit circle angles
        """
        return self._angles

    @property
    def axes(self) -> typing.Union[typing.Dict[str, D], None]:
        """
        """
        return self._axes

    @property
    def quadrants(self) -> typing.Union[typing.Dict[str, typing.Tuple[D]], None]:
        """
        """
        return self._quadrants

    @property
    def ucircle_angles(self) -> typing.Dict[str, typing.Union[D, typing.Tuple[D]]]:
        """
        """
        return self._ucircle_angles

    @property
    def axis_values(self) -> typing.Dict[str, D]:
        """
        """
        return self._axis_values

    @property
    def quadrant_values(self) -> typing.Dict[str, typing.Tuple[D]]:
        """
        """
        return self._quadrant_values

    def approximate_angle(self, x: D) -> typing.Optional[D]:
        """
        Attempts to approximate the value of the trigonometric function at :math:`x` using the unit
        circle angles. If :math:`x` is not 

        :param x:
        :return:
        """
        with decimal.localcontext() as ctx:
            tolerance = D(10) ** -(ctx.prec - 2)

            for axis, theta in self.axes.items():
                error = ((x - theta) % (2 * PI)).quantize(D(10) ** -(ctx.prec - 1))
                if abs(error) <= tolerance:
                    return self.axis_values[axis]

            for quadrant, angles in self.quadrants.items():
                for i, theta in enumerate(angles):
                    error = ((x - theta) % (2 * PI)).quantize(D(10) ** -(ctx.prec - 1))
                    if abs(error) <= tolerance:
                        return self.quadrant_values[quadrant][i]

            return None


# ------------------------------------ Trigonometric Functions ------------------------------------


@precision
def sine(x: D) -> D:
    r"""
    Evaluates :math:`\sin(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/Sine.html>`_.

    :param x: The value at which to evaluate :math:`\sin(x)`
    :param n: The number of decimal places of precision
    """
    ucircle = UnitCircle(
        axis_values={"posx": 0, "negx": 0, "posy": 1, "negy": -1},
        quadrant_values={
            "q1": (1 / D(2), D(2).sqrt() / 2, D(3).sqrt() / 2),
            "q2": (D(3).sqrt() / 2, D(2).sqrt() / 2, 1 / D(2)),
            "q3": (-1 / D(2), -D(2).sqrt() / 2, -D(3).sqrt() / 2),
            "q4": (-D(3).sqrt() / 2, -D(2).sqrt() / 2, -1 / D(2))
        }
    )

    res = ucircle.approximate_angle(x)
    return _sine(x) if res is None else res


@precision
def cosine(x: D) -> D:
    r"""
    Evaluates :math:`\cos(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/Cosine.html>`_.

    :param x: The value at which to evaluate :math:`\cos(x)`.
    :param n: The number of decimal places of precision
    """
    ucircle = UnitCircle(
        axis_values={"posx": 1, "negx": -1, "posy": 0, "negy": 0},
        quadrant_values={
            "q1": (D(3).sqrt() / 2, D(2).sqrt() / 2, 1 / D(2)),
            "q2": (-1 / D(2), -D(2).sqrt() / 2, -D(3).sqrt() / 2),
            "q3": (-D(3).sqrt() / 2, -D(2).sqrt() / 2, -1 / D(2)),
            "q4": (1 / D(2), D(2).sqrt() / 2, D(3).sqrt() / 2)
        }
    )

    res = ucircle.approximate_angle(x)
    return _cosine(x) if res is None else res


@precision
def tangent(x: D) -> D:
    r"""
    Evaluates :math:`\tan(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/Tangent.html>`_.

    :param x: The value at which to evaluate :math:`\tan(x)`.
    :param n: The number of decimal places of precision
    """
    ucircle = UnitCircle(axis_values={"posx": 0, "negx": 0, "posy": NAN, "negy": NAN})

    try:
        res = ucircle.approximate_angle(x)
        return sine(x) / cosine(x) if res is None else res
    except ZeroDivisionError:
        return NAN


@precision
def secant(x: D) -> D:
    r"""
    Evaluates :math:`\sec(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/Secant.html>`_.

    :param x: The value at which to evaluate :math:`\sec(x)`.
    :param n: The number of decimal places of precision
    """
    ucircle = UnitCircle(axis_values={"posx": 1, "negx": -1, "posy": NAN, "negy": NAN})
    
    try:
        res = ucircle.approximate_angle(x)
        return 1 / cosine(x) if res is None else res
    except ZeroDivisionError:
        return NAN


@precision
def cosecant(x: D) -> D:
    r"""
    Evaluates :math:`\csc(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/Cosecant.html>`_.

    :param x: The value at which to evaluate :math:`\csc(x)`.
    :param n: The number of decimal places of precision
    """
    ucircle = UnitCircle(axis_values={"posx": NAN, "negx": NAN, "posy": 1, "negy": -1})
    
    try:
        res = ucircle.approximate_angle(x)
        return 1 / sine(x) if res is None else res
    except ZeroDivisionError:
        return NAN


@precision
def cotangent(x: D) -> D:
    r"""
    Evaluates :math:`\cot(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/Cotangent.html>`_.

    :param x: The value at which to evaluate :math:`\cot(x)`.
    :param n: The number of decimal places of precision
    """
    ucircle = UnitCircle(axis_values={"posx": NAN, "negx": NAN, "posy": 0, "negy": 0})

    try:
        res = ucircle.approximate_angle(x)
        return cosine(x) / sine(x) if res is None else res
    except ZeroDivisionError:
        return NAN


# Shorthands for trigonometric functions
sin, cos, tan = sine, cosine, tangent
sec, csc, cot = secant, cosecant, cotangent


# -------------------------------- Inverse Trigonometric Functions --------------------------------


@precision
def arcsine(x: D) -> D:
    r"""
    Evaluates :math:`\arcsin(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/InverseSine.html>`_.

    :param x: The value at which to evaluate :math:`\arcsin(x)`.
    :param n: The number of decimal places of precision
    :raise ValueError: ``x`` is outside the domain of :math:`\arcsin(x)`
    """
    if not abs(x) <= 1:
        raise ValueError("domain error")

    with decimal.localcontext():
        return (PI / 2 if x == 1 else -PI / 2) if abs(x) == 1 else _arcsine(x)


@precision
def arccosine(x: D) -> D:
    r"""
    Evaluates :math:`\arccos(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/InverseCosine.html>`_.

    :param x: The value at which to evaluate :math:`\arccos(x)`.
    :param n: The number of decimal places of precision
    :raise ValueError: ``x`` is outside the domain of :math:`\arccos(x)`
    """
    if not abs(x) <= 1:
        raise ValueError("domain error")

    with decimal.localcontext():
        return (D(0) if x == 1 else PI) if abs(x) == 1 else PI / 2 - _arcsine(x)


@precision
def arctangent(x: D) -> D:
    r"""
    Evaluates :math:`\arctan(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/InverseTangent.html>`_.
    
    :param x: The value at which to evaluate :math:`\arctan(x)`.
    :param n: The number of decimal places of precision
    """
    if abs(x) is INF:
        return PI / 2 if x is INF else -PI / 2

    with decimal.localcontext():
        return _arctangent(x) if -1 < x < 1 else arcsine(x / (D(1) + x ** 2).sqrt())


@precision
def arcsecant(x: D) -> D:
    r"""
    Evaluates :math:`\operatorname{arcsec}(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/InverseSecant.html>`_.

    :param x: The value at which to evaluate :math:`\operatorname{arcsec}(x)`.
    :param n: The number of decimal places of precision
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{arcsec}(x)`
    """
    if not abs(x) >= 1:
        raise ValueError("domain error")

    with decimal.localcontext():
        return PI / 2 if abs(x) is INF else arccosine(1 / x)


@precision
def arccosecant(x: D) -> D:
    r"""
    Evaluates :math:`\operatorname{arccsc}(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/InverseCosecant.html>`_.

    :param x: The value at which to evaluate :math:`\operatorname{arccsc}(x)`.
    :param n: The number of decimal places of precision
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{arccsc}(x)`
    """
    if not abs(x) >= 1:
        raise ValueError("domain error")

    with decimal.localcontext():
        return D(0) if abs(x) is INF else arcsine(1 / x)


@precision
def arccotangent(x: D) -> D:
    r"""
    Evaluates :math:`\operatorname{arccot}(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/InverseCotangent.html>`_.

    :param x: The value at which to evaluate :math:`\arccot(x)`.
    :param n: The number of decimal places of precision
    """
    with decimal.localcontext():
        return (D(0) if x is INF else PI) if abs(x) is INF else arctangent(1 / x)


# Shorthands for inverse trigonometric functions
arcsin, arccos, arctan = arcsine, arccosine, arctangent
arcsec, arccsc, arccot = arcsecant, arccosecant, arccotangent


# -------------------------------------- Hyperbolic Functions -------------------------------------


@precision
def hyperbolic_sine(x: D) -> D:
    r"""
    Evaluates :math:`\sinh(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/HyperbolicSine.html>`_.

    :param x: The value at which to evaluate :math:`\sinh(x)`.
    :param n: The number of decimal places of precision
    """
    return _hyperbolic_sine(x)


@precision
def hyperbolic_cosine(x: D) -> D:
    r"""
    Evaluates :math:`\cosh(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/HyperbolicCosine.html>`_.

    :param x: The value at which to evaluate :math:`\cosh(x)`.
    :param n: The number of decimal places of precision
    """
    return _hyperbolic_cosine(x)


@precision
def hyperbolic_tangent(x: D) -> D:
    r"""
    Evaluates :math:`\tanh(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/HyperbolicTangent.html>`_.

    :param x: The value at which to evaluate :math:`\tanh(x)`.
    :param n: The number of decimal places of precision
    """
    return hyperbolic_sine(x) / hyperbolic_cosine(x)


@precision
def hyperbolic_secant(x: D) -> D:
    r"""
    Evaluates :math:`\operatorname{sech}(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/HyperbolicSecant.html>`_.

    :param x: The value at which to evaluate :math:`\operatorname{sech}(x)`.
    :param n: The number of decimal places of precision
    """
    return 1 / hyperbolic_cosine(x)


@precision
def hyperbolic_cosecant(x: D) -> D:
    r"""
    Evaluates :math:`\operatorname{csch}(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/HyperbolicCosecant.html>`_.

    :param x: The value at which to evaluate :math:`\operatorname{csch}(x)`.
    :param n: The number of decimal places of precision
    """
    try:
        return 1 / hyperbolic_sine(x)
    except ZeroDivisionError:
        return NAN


@precision
def hyperbolic_cotangent(x: D) -> D:
    r"""
    Evaluates :math:`\coth(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/HyperbolicCotangent.html>`_.

    :param x: The value at which to evaluate :math:`\coth(x)`.
    :param n: The number of decimal places of precision
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
def hyperbolic_arcsine(x: D) -> D:
    r"""
    Evaluates :math:`\operatorname{arsinh}(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/InverseHyperbolicSine.html>`_.

    :param x: The value at which to evaluate :math:`\operatorname{arsinh}(x)`.
    :param n: The number of decimal places of precision
    """
    return ln(x + D(x ** 2 + 1).sqrt()) if abs(x) >= 0.95 else _hyperbolic_arcsine(x)


@precision
def hyperbolic_arccosine(x: D) -> D:
    r"""
    Evaluates :math:`\operatorname{arcosh}(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/InverseHyperbolicCosine.html>`_.
    
    :param x: The value at which to evaluate :math:`\operatorname{arcosh}(x)`.
    :param n: The number of decimal places of precision
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{arcosh}(x)`
    """
    if not abs(x) >= 1:
        raise ValueError("domain error")

    return ln(x + D(x ** 2 - 1).sqrt()) if x ** 2 > 1.95 else hyperbolic_arcsine(x ** 2 - 1)


@precision
def hyperbolic_arctangent(x: D) -> D:
    r"""
    Evaluates :math:`\operatorname{artanh}(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/InverseHyperbolicTangent.html>`_.
    
    :param x: The value at which to evaluate :math:`\operatorname{artanh}(x)`.
    :param n: The number of decimal places of precision
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{artanh}(x)`
    """
    if not abs(x) < 1:
        raise ValueError("domain error")
        
    return _hyperbolic_arctangent(x)


@precision
def hyperbolic_arcsecant(x: D) -> D:
    r"""
    Evaluates :math:`\operatorname{arsech}(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/InverseHyperbolicSecant.html>`_.
    
    :param x: The value at which to evaluate :math:`\operatorname{arsech}(x)`.
    :param n: The number of decimal places of precision
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{arsech}(x)`
    """
    if not 0 < x <= 1:
        raise ValueError("domain error")

    return hyperbolic_cosine(1 / x)


@precision
def hyperbolic_arccosecant(x: D) -> D:
    r"""
    Evaluates :math:`\operatorname{arcsch}(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/InverseHyperbolicCosecant.html>`_.
    
    :param x: The value at which to evaluate :math:`\operatorname{arcsch}(x)`.
    :param n: The number of decimal places of precision
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{arcsch}(x)`
    """
    try:
        return hyperbolic_arcsine(1 / x)
    except ZeroDivisionError:
        return NAN


@precision
def hyperbolic_arccotangent(x: D) -> D:
    r"""
    Evaluates :math:`\operatorname{arcoth}(x)` to ``n`` decimal places of precision.

    `[] <https://mathworld.wolfram.com/InverseHyperbolicCotangent.html>`_.
    
    :param x: The value at which to evaluate :math:`\operatorname{arcoth}(x)`.
    :param n: The number of decimal places of precision
    :raise ValueError: ``x`` is outside the domain of :math:`\operatorname{arcoth}(x)`
    """
    if not abs(x) > 1:
        raise ValueError("domain error")

    return hyperbolic_tangent(1 / x)


# Shorthands for inverse hyperbolic functions
arsinh, arcosh, artanh = hyperbolic_arcsine, hyperbolic_arccosine, hyperbolic_arctangent
arsech, arcsch, arcoth = hyperbolic_arcsecant, hyperbolic_arccosecant, hyperbolic_arccotangent
