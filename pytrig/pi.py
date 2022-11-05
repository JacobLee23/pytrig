"""
`Wikipedia`_

Vestermark, H. (2022). Practical Implementation of π Algorithms. Retrieved November 4, 2022.

.. _Wikipedia: https://en.wikipedia.org/wiki/Approximations_of_%CF%80
"""

import decimal
from math import factorial
import typing

from .constants import PRECISION
from .maclaurin_series import arctangent

D = decimal.Decimal


def _precision(func: typing.Callable[[decimal.Context], D]) -> typing.Callable:
    """

    :param func:
    :return:
    """
    def wrapper(precision: int = PRECISION) -> D:
        """

        :param precision:
        :return:
        """
        with decimal.localcontext() as ctx:
            ctx.prec = precision + 3

            res = func(ctx)

        with decimal.localcontext() as ctx:
            ctx.prec = precision + 1

            return +res

    return wrapper


def _summation(func: typing.Callable[[int], D], ctx: decimal.Context) -> D:
    """

    :param func:
    :param ctx:
    :return:
    """
    with decimal.localcontext(ctx):
        # Initial conditions
        sum_ = D(0)
        k: int = 0

        while True:
            term = func(k)

            # Test for convergence
            if sum_ + term == sum_:
                return sum_

            sum_ += term
            k += 1


def _product(func: typing.Callable[[int], D], ctx: decimal.Context) -> D:
    """

    :param func:
    :param ctx:
    :return:
    """
    with decimal.localcontext(ctx):
        # Initial conditions
        product_ = D(1)
        k: int = 1

        while True:
            term = func(k)

            # Test for convergence
            if product_ * term == product_:
                return product_

            product_ *= term
            k += 1


@_precision
def bbp(ctx: decimal.Context) -> D:
    r"""

    :param ctx:
    :return:
    """
    # Initial conditions
    sum_ = D(0)
    k: int = 0

    while True:
        term = 1 / D(16) ** D(k) * sum([
            4 / D(8 * k + 1),
            -2 / D(8 * k + 4),
            -1 / D(8 * k + 5),
            -1 / D(8 * k + 6)
        ])

        # Test for convergence
        if sum_ + term == sum_:
            return sum_

        sum_ += term
        k += 1


class BorweinAlgorithm:
    r"""
    `Wikipedia`_

    .. _Wikipedia: https://en.wikipedia.org/wiki/Borwein%27s_algorithm
    """
    @staticmethod
    @_precision
    def quadratic(ctx: decimal.Context) -> D:
        """

        :param ctx:
        :return:
        """
        with decimal.localcontext(ctx):
            # Initial conditions
            a = D(2).sqrt()
            b = D(0)
            p = D(2) + D(2).sqrt()

            while True:
                a_ = (a.sqrt() + 1 / a.sqrt()) / 2
                b_ = (1 + b) * a.sqrt() / (a + b)
                p_ = (1 + a_) * p * b_ / (1 + b_)

                # Test for convergence
                if p == p_:
                    return p_

                a, b, p = a_, b_, p_

    @staticmethod
    @_precision
    def cubic(ctx: decimal.Context) -> D:
        """

        :param ctx:
        :return:
        """
        with decimal.localcontext(ctx):
            # Initial conditions
            k = 0
            a = 1 / D(3)
            s = (D(3).sqrt() - 1) / 2

            while True:
                r_ = 3 / (1 + 2 * (1 - s ** 3) ** (1 / D(3)))
                s_ = (r_ - 1) / 2
                a_ = r_ ** 2 * a - 3 ** k * (r_ ** 2 - 1)

                # Test for convergence
                if a == a_:
                    return 1 / a_

                k += 1
                a, s = a_, s_

    @staticmethod
    @_precision
    def quartic(ctx: decimal.Context) -> D:
        r"""

        :param ctx:
        :return:
        """
        with decimal.localcontext(ctx):
            # Initial conditions
            k = 0
            a = 2 * D(D(2).sqrt() - 1) ** 2
            y = D(D(2).sqrt() - 1)

            while True:
                y_ = (1 - (1 - y ** 4) ** (1 / D(4))) / (1 + (1 - y ** 4) ** (1 / D(4)))
                a_ = a * (1 + y_) ** 4 - 2 ** (2 * k + 3) * y_ * (1 + y_ + y_ ** 2)

                # Test for convergence
                if a == a_:
                    return 1 / a_

                k += 1
                a, y = a_, y_

    @staticmethod
    @_precision
    def quintic(ctx: decimal.Context) -> D:
        r"""

        :param ctx:
        :return:
        """

        with decimal.localcontext(ctx):
            # Initial conditions
            k = 0
            a = 1 / D(2)
            s = 5 * (D(5).sqrt() - 2)

            while True:
                x_ = 5 / s - 1
                y_ = (x_ - 1) ** 2 + 7
                z_ = (x_ / 2 * (y_ + (y_ ** 2 - 4 * x_ ** 3).sqrt())) ** (1 / D(5))
                a_ = s ** 2 * a - 5 ** k * ((s ** 2 - 5) / 2 + (s * (s ** 2 - 2 * s + 5)).sqrt())
                s_ = 25 / (((z_ + x_ / z_ + 1) ** 2) * s)

                # Test for convergence
                if a == a_:
                    return 1 / a_

                k += 1
                a, s = a_, s_

    @staticmethod
    @_precision
    def nonic(ctx: decimal.Context) -> D:
        r"""

        :param ctx:
        :return:
        """
        with decimal.localcontext(ctx):
            # Initial conditions
            k = 0
            a = 1 / D(3)
            r = (D(3).sqrt() - 1) / 2
            s = (1 - r ** 3) ** (1 / D(3))

            while True:
                t_ = 1 + 2 * r
                u_ = (9 * r * (1 + r + r ** 2)) ** (1 / D(3))
                v_ = t_ ** 2 + t_ * u_ + u_ ** 2
                w_ = 27 * (1 + s + s ** 2) / v_
                a_ = w_ * a + 3 ** D(2 * k - 1) * (1 - w_)
                s_ = (1 - r) ** 3 / ((t_ + 2 * u_) * v_)
                r_ = (1 - s_ ** 3) ** (1 / D(3))

                # Test for convergence
                if k != 1 and a == a_:
                    return 1 / a_

                k += 1
                a, r, s = a_, r_, s_


@_precision
def brent_salamin_method(ctx: decimal.Context) -> D:
    r"""

    :param ctx:
    :return:
    """
    with decimal.localcontext(ctx):
        # Initial conditions
        k = 0
        a = D(1)
        b = 1 / D(2).sqrt()
        c = 1 / D(2)

        while True:
            a_ = (a + b) / 2
            b_ = (a * b).sqrt()
            c_ = c - 2 ** (k + 1) * (a_ - b) ** 2

            # Test for convergence
            if a_ + b_ == 2 * a_ == 2 * b_:
                return 2 * a_ ** 2 / c_

            a, b, c = a_, b_, c_
            k += 1


@_precision
def chudnovsky_algorithm(ctx: decimal.Context) -> D:
    r"""
    `Wikipedia`_

    .. _Wikipedia: https://en.wikipedia.org/wiki/Chudnovsky_algorithm

    :param ctx:
    :return:
    """
    with decimal.localcontext(ctx):
        # Initial conditions
        sum_ = D(0)
        k: int = 0

        # Ramanujan-Sato series generalization
        c = 426880 * D(10005).sqrt()
        m = lambda n: D(factorial(6 * k)) / (D(factorial(3 * k)) * D(factorial(k)) ** 3)
        l = lambda n: D(545140134 * n + 13591409)
        x = lambda n: D(-262537412640768000) ** n

        while True:
            term = m(k) * l(k) / x(k)

            # Test for convergence
            if sum_ + term == sum_:
                return c * sum_ ** -1

            sum_ += term
            k += 1


@_precision
def euler_formula(ctx: decimal.Context) -> D:
    r"""

    :param ctx:
    :return:
    """
    with decimal.localcontext(ctx):
        return sum([
            20 * sum(arctangent(1 / D(7))),
            8 * sum(arctangent(D(3) / D(79)))
        ])


@_precision
def euler_product(ctx: decimal.Context) -> D:
    r"""

    :param ctx:
    :return:
    """
    def odd_primes() -> typing.Generator[int, None, None]:
        r"""
        Creates an interator that returns odd prime number in the interval :math:`[3, \infinity)`.

        :return: A generator of odd prime numbers
        """
        n: int = 3
        yield n

        while True:
            n += 1
            prime = True

            for i in range(2, int(D(n).sqrt()) + 1):
                # Test for divisibility
                if n % i == 0:
                    prime = False
                    break

            if prime:
                yield n

    with decimal.localcontext(ctx):
        # A generator of odd prime numbers
        primes = odd_primes()

        # Initial conditions
        product_ = D(1)

        for x in primes:
            term = D(x) / D((x + 1) if x % 4 == 3 else (x - 1))

            # Test for convergence
            if product_ * term == product_:
                return 4 * product_

            product_ *= term


@_precision
def gauss_legendre(ctx: decimal.Context) -> D:
    """
    `Wikipedia`_

    .. _Wikipedia: https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_algorithm

    :param ctx:
    :return:
    """
    with decimal.localcontext(ctx):
        # Initial conditions
        a = D(1)
        b = 1 / D(2).sqrt()
        t = 1 / D(4)
        p = D(1)

        while True:
            a_ = (a + b) / 2
            b_ = (a * b).sqrt()
            t_ = t - p * (a - a_) ** 2
            p_ = 2 * p

            # Test for convergence
            if a_ + b_ == 2 * a_ == 2 * b_:
                return (a_ + b_) ** 2 / (4 * t_)

            a, b, t, p = a_, b_, t_, p_


@_precision
def leibniz_formula(ctx: decimal.Context) -> D:
    """
    `Wikipedia`_

    .. _Wikipedia: https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80

    :param ctx:
    :return:
    """
    with decimal.localcontext(ctx):
        sum_ = _summation(
            lambda k: (1 / D(4 * k + 1)) - (1 / D(4 * k + 3)),
            ctx
        )

        return 4 * sum_


@_precision
def madhava_series(ctx: decimal.Context) -> D:
    r"""
    `Madhava Series`_

    .. _Madhava Series: https://en.wikipedia.org/wiki/Madhava_series

    :param ctx:
    :return:
    """
    with decimal.localcontext(ctx):
        sum_ = _summation(
            lambda k: D(-3) ** -k / (2 * k + 1),
            ctx
        )

        return D(12).sqrt() * sum_


@_precision
def newton_formula(ctx: decimal.Context) -> D:
    r"""

    :param ctx:
    :return:
    """
    with decimal.localcontext(ctx):
        sum_ = _summation(
            lambda k: D(2) ** k * D(factorial(k)) ** 2 / D(factorial(2 * k + 1)),
            ctx
        )

        return 2 * sum_


@_precision
def nilakantha_formula(ctx: decimal.Context) -> D:
    r"""

    :param ctx:
    :return:
    """
    with decimal.localcontext(ctx):
        sum_ = _summation(
            lambda k: 1 / D((2 * k + 2) * (2 * k + 3) * (2 * k + 4)) * D(-1) ** k,
            ctx
        )

        return 4 * sum_ + D(3)


@_precision
def ramanujan_formula(ctx: decimal.Context) -> D:
    """

    :param ctx:
    :return:
    """
    with decimal.localcontext(ctx):
        # Ramanujan-Sato series generalization
        a, b, c = D(26390), D(1103), D(396)

        sum_ = _summation(
            lambda k: D(factorial(4 * k)) / (D(factorial(k)) ** 4) * (a * k + b) / (c ** (4 * k)),
            ctx
        )

        return 1 / ((2 * D(2).sqrt() / D(99) ** 2) * sum_)


@_precision
def stormer_formula(ctx: decimal.Context) -> D:
    r"""

    :param ctx:
    :return:
    """
    with decimal.localcontext(ctx):
        return 4 * sum([
            44 * sum(arctangent(1 / D(57))),
            7 * sum(arctangent(1 / D(239))),
            -12 * sum(arctangent(1 / D(682))),
            24 * sum(arctangent(1 / D(12943)))
        ])


@_precision
def takano_formula(ctx: decimal.Context) -> D:
    r"""

    :param ctx:
    :return:
    """
    with decimal.localcontext(ctx):
        return 4 * sum([
            12 * sum(arctangent(1 / D(49))),
            32 * sum(arctangent(1 / D(57))),
            -5 * sum(arctangent(1 / D(239))),
            12 * sum(arctangent(1 / D(110443)))
        ])


@_precision
def viete_formula(ctx: decimal.Context) -> D:
    r"""
    `Wikipedia`_

    .. _Wikipedia: https://en.wikipedia.org/wiki/Vi%C3%A8te%27s_formula

    :param ctx:
    :return:
    """
    with decimal.localcontext(ctx):
        # Initial conditions
        product_ = D(1)
        term = D(0)

        while True:
            term = (D(2) + 2 * term).sqrt() / 2

            # Test for convergence
            if product_ * term == product_:
                return 2 / product_

            product_ *= term


@_precision
def wallis_product(ctx: decimal.Context) -> D:
    r"""
    `Wikipedia`_

    .. _Wikipedia: https://en.wikipedia.org/wiki/Wallis_product

    :param ctx:
    :return:
    """
    with decimal.localcontext(ctx):
        product_ = _product(
            lambda k: D(4 * k ** 2) / D(4 * k ** 2 - 1),
            ctx
        )

        return 2 * product_
