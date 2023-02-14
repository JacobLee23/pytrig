"""
"""

import decimal
from math import comb, factorial
import typing


D = decimal.Decimal


PRECISION = 100


INF = D("Infinity")
NAN = D("NaN")


def _precision(func: typing.Callable[[D, int], D]) -> typing.Callable[[D, int], D]:
    """
    :param func:
    :return:
    """
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


def chudnovsky_algorithm(prec: int = PRECISION) -> D:
    r"""
    :param prec:
    :return:
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


# -------------------------------- Natural Logarithm Approximation --------------------------------


def log_natural(x: D, prec: int = PRECISION) -> D:
    """
    :param x:
    :param prec:
    :return: The value of 'x' is outside the domain of ln(x)
    """
    if x <= 0:
        raise ValueError("domain error")

    with decimal.localcontext() as ctx:
        ctx.prec = prec + 2

        res = None
        a = D(1)
        while True:
            res_ = a * (x ** (1 / a)) - a

            if res is not None and res == res_:
                return res

            res = res_
            a *= 10


# ---------------------------------- Maclaurin Series Expansions ----------------------------------


def ms_sine(n: int, x: D) -> D:
    return D(-1) ** n / D(factorial(2 * n + 1)) * (x ** (2 * n + 1))


def ms_cosine(n: int, x: D) -> D:
    return D(-1) ** n / D(factorial(2 * n)) * (x ** (2 * n))


def ms_arcsine(n: int, x: D) -> D:
    return (1 / D(4)) ** n * comb(2 * n, n) * (x ** (2 * n + 1) / (2 * n + 1))


def ms_arctangent(n: int, x: D) -> D:
    return D(-1) ** n / D(2 * n + 1) * (x ** (2 * n + 1))


def ms_hyperbolic_sine(n: int, x: D) -> D:
    return x ** D(2 * n + 1) / D(factorial(2 * n + 1))


def ms_hyperbolic_cosine(n: int, x: D) -> D:
    return x ** (2 * n) / D(factorial(2 * n))


def ms_hyperbolic_arcsine(n: int, x: D) -> D:
    return (D(-1) / D(4)) ** n * comb(2 * n, n) * (x ** (2 * n + 1) / (2 * n + 1))


def ms_hyperbolic_tangent(n: int, x: D) -> D:
    return x ** (2 * n + 1) / (2 * n + 1)


class MaclaurinExpansion:
    """
    :param term:
    """
    def __init__(self, term: typing.Callable[[int, D], D]):
        self._term = term

    def __call__(self, x: D) -> D:
        return sum(self.expand(x))

    @property
    def term(self) -> typing.Callable[[int, D], D]:
        """
        :return:
        """
        return self._term

    def expand(self, x: D) -> typing.Generator[D, None, None]:
        """
        :param x:
        :return:
        """
        n = 0
        while True:
            try:
                term = self.term(n, x)
            except decimal.Overflow:
                return

            if term + D(1) == D(1):
                return

            yield term

            n += 1


_sine = MaclaurinExpansion(ms_sine)
_cosine = MaclaurinExpansion(ms_cosine)
_arcsine = MaclaurinExpansion(ms_arcsine)
_arctangent = MaclaurinExpansion(ms_arctangent)
_hyperbolic_sine = MaclaurinExpansion(ms_hyperbolic_sine)
_hyperbolic_cosine = MaclaurinExpansion(ms_hyperbolic_cosine)
_hyperbolic_arcsine = MaclaurinExpansion(ms_hyperbolic_arcsine)
_hyperbolic_arctangent = MaclaurinExpansion(ms_hyperbolic_tangent) 


# ------------------------------------- Unit Circle Evaluation ------------------------------------


class UnitCircle:
    r"""
    :param ctx:
    :param axis_values: The values of the trigonometric function at the +x, +y, -x, and -y axes
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
        :return:
        """
        return self._angles

    @property
    def axes(self) -> typing.Union[typing.Dict[str, D], None]:
        """
        :return:
        """
        return self._axes

    @property
    def quadrants(self) -> typing.Union[typing.Dict[str, typing.Tuple[D]], None]:
        """
        :return:
        """
        return self._quadrants

    @property
    def ucircle_angles(self) -> typing.Dict[str, typing.Union[D, typing.Tuple[D]]]:
        """
        :return:
        """
        return self._ucircle_angles

    @property
    def axis_values(self) -> typing.Dict[str, D]:
        """
        :return:
        """
        return self._axis_values

    @property
    def quadrant_values(self) -> typing.Dict[str, typing.Tuple[D]]:
        """
        :return:
        """
        return self._quadrant_values

    def approximate_angle(self, x: D) -> typing.Optional[D]:
        """
        :param x:
        :param prec:
        :return:
        """
        with decimal.localcontext() as ctx:
            tolerance = D(10) ** -(ctx.prec - 2)

            for axis, theta in self.axes.items():
                error = ((x - theta) % (2 * PI)).quantize(D(10) ** -(ctx.prec - 1))
                if abs(error - tolerance) < 1:
                    return self.axis_values[axis]

            for quadrant, angles in self.quadrants.items():
                for i, theta in enumerate(angles):
                    error = ((x - theta) % (2 * PI)).quantize(D(10) ** -(ctx.prec - 1))
                    if abs(error - tolerance) < 1:
                        return self.quadrant_values[quadrant][i]

            return None


# ------------------------------------ Trigonometric Functions ------------------------------------


@_precision
def sine(x: D) -> D:
    r"""
    :param x:
    :return:
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


@_precision
def cosine(x: D) -> D:
    r"""
    :param x:
    :return:
    """
    ucircle = UnitCircle(
        axis_values = {"posx": 1, "negx": -1, "posy": 0, "negy": 0},
        quadrant_values={
            "q1": (D(3).sqrt() / 2, D(2).sqrt() / 2, 1 / D(2)),
            "q2": (-1 / D(2), -D(2).sqrt() / 2, -D(3).sqrt() / 2),
            "q3": (-D(3).sqrt() / 2, -D(2).sqrt() / 2, -1 / D(2)),
            "q4": (1 / D(2), D(2).sqrt() / 2, D(3).sqrt() / 2)
        }
    )

    res = ucircle.approximate_angle(x)
    return _cosine(x) if res is None else res


@_precision
def tangent(x: D) -> D:
    r"""
    :param x:
    :return:
    """
    ucircle = UnitCircle(axis_values={"posx": 0, "negx": 0, "posy": NAN, "negy": NAN})

    try:
        res = ucircle.approximate_angle(x)
        return sine(x) / cosine(x) if res is None else res
    except ZeroDivisionError:
        return NAN


@_precision
def secant(x: D) -> D:
    r"""
    :param x:
    :return:
    """
    ucircle = UnitCircle(axis_values={"posx": 1, "negx": -1, "posy": NAN, "negy": NAN})
    
    try:
        res = ucircle.approximate_angle(x)
        return 1 / cosine(x) if res is None else res
    except ZeroDivisionError:
        return NAN


@_precision
def cosecant(x: D) -> D:
    r"""
    :param x:
    :return:
    """
    ucircle = UnitCircle(axis_values={"posx": NAN, "negx": NAN, "posy": 1, "negy": -1})
    
    try:
        res = ucircle.approximate_angle(x)
        return 1 / sine(x) if res is None else res
    except ZeroDivisionError:
        return NAN


@_precision
def cotangent(x: D) -> D:
    r"""
    :param x:
    :return:
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


@_precision
def arcsine(x: D) -> D:
    r"""
    :param x:
    :return:
    :raise ValueError: The value of 'x' is outside the domain of arcsin(x)
    """
    if not abs(x) <= 1:
        raise ValueError("domain error")

    with decimal.localcontext():
        return (PI / 2 if x == 1 else -PI / 2) if abs(x) == 1 else _arcsine(x)


@_precision
def arccosine(x: D) -> D:
    r"""
    :param x:
    :return:
    :raise ValueError: The value of 'x' is outside the domain of arccos(x)
    """
    if not abs(x) <= 1:
        raise ValueError("domain error")

    with decimal.localcontext():
        return (D(0) if x == 1 else PI) if abs(x) == 1 else PI / 2 - arcsine(x)


@_precision
def arctangent(x: D) -> D:
    r"""
    :param x:
    :return:
    """
    if abs(x) is INF:
        return PI / 2 if x is INF else -PI / 2

    with decimal.localcontext():
        return _arctangent(x) if -1 < x < 1 else arcsine(x / (D(1) + x ** 2).sqrt())


@_precision
def arcsecant(x: D) -> D:
    r"""
    :param x:
    :return:
    :raise ValueError: The value of 'x' is outside the domain of arcsec(x)
    """
    if not abs(x) >= 1:
        raise ValueError("domain error")

    with decimal.localcontext():
        return PI / 2 if abs(x) is INF else arccosine(1 / x)


@_precision
def arccosecant(x: D) -> D:
    r"""
    :param x:
    :return:
    :raise ValueError: The value of 'x' is outside the domain of arccsc(x)
    """
    if not abs(x) >= 1:
        raise ValueError("domain error")

    with decimal.localcontext():
        return D(0) if abs(x) is INF else arcsine(1 / x)


@_precision
def arccotangent(x: D) -> D:
    r"""
    :param x:
    :return:
    """
    with decimal.localcontext():
        return (D(0) if x is INF else PI) if abs(x) is INF else arctangent(1 / x)


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
    return _hyperbolic_sine(x)


@_precision
def hyperbolic_cosine(x: D) -> D:
    r"""
    :param x:
    :return:
    """
    return _hyperbolic_cosine(x)


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


# Shorthands for hyperbolic functions
sinh, cosh, tanh = hyperbolic_sine, hyperbolic_cosine, hyperbolic_tangent
sech, csch, coth = hyperbolic_secant, hyperbolic_cosecant, hyperbolic_cotangent


# ---------------------------------- Inverse Hyperbolic Functions ----------------------------------


@_precision
def hyperbolic_arcsine(x: D) -> D:
    r"""
    :param x:
    :return:
    """
    return log_natural(x + D(x ** 2 + 1).sqrt()) if abs(x) >= 0.95 else _hyperbolic_arcsine(x)


@_precision
def hyperbolic_arccosine(x: D) -> D:
    r"""
    :param x:
    :return:
    :raise ValueError: Value of 'x' is outside the domain of arcosh(x)
    """
    if not abs(x) > 1:
        raise ValueError("domain error")

    return hyperbolic_arcsine(x ** 2 - 1)


@_precision
def hyperbolic_arctangent(x: D) -> D:
    r"""
    :param x:
    :return:
    :raise ValueError: Value of 'x' is outside the domain of artanh(x)
    """
    if not abs(x) < 1:
        raise ValueError("domain error")
        
    return _hyperbolic_arctangent(x)


@_precision
def hyperbolic_arcsecant(x: D) -> D:
    r"""
    :param x:
    :return:
    :raise ValueError: Value of 'x' is outside the domain of arsech(x)
    """
    if not 0 < x <= 1:
        raise ValueError("domain error")

    return hyperbolic_cosine(1 / x)


@_precision
def hyperbolic_arccosecant(x: D) -> D:
    r"""
    :param x:
    :return:
    :raise ValueError: Value of 'x' is outside the domain of arcsch(x)
    """
    if not abs(x) > 1:
        raise ValueError("domain error")

    return hyperbolic_arcsine(1 / x)


@_precision
def hyperbolic_arccotangent(x: D) -> D:
    r"""
    :param x:
    :return:
    :raise ValueError: Value of 'x' is outside the domain of artanh(x)
    """
    if not abs(x) > 1:
        raise ValueError("domain error")

    return hyperbolic_tangent(1 / x)


# Shorthands for inverse hyperbolic functions
arsinh, arcosh, artanh = hyperbolic_arcsine, hyperbolic_arccosine, hyperbolic_arctangent
arsech, arcsch, arcoth = hyperbolic_arcsecant, hyperbolic_arccosecant, hyperbolic_arccotangent
