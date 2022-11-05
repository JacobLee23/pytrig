"""
`Wikipedia`_

.. _Wikipedia: https://en.wikipedia.org/wiki/Approximations_of_%CF%80
"""

import decimal
from math import factorial
import typing

from .constants import PRECISION

D = decimal.Decimal


def _summation(func: typing.Callable[[int], D], ctx: decimal.Context) -> D:
    """

    :param func:
    :param ctx:
    :return:
    """
    with decimal.localcontext(ctx):
        sum_ = D(0)
        k: int = 0

        while True:
            term = func(k)

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
        product_ = D(1)
        k: int = 1

        while True:
            term = func(k)

            if product_ * term == product_:
                return product_

            product_ *= term
            k += 1


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
            ctx.prec = precision + 2

            res = func(ctx)

        with decimal.localcontext() as ctx:
            ctx.prec = precision + 1

            return +res

    return wrapper


class BorweinAlgorithm:
    r"""
    `Wikipedia`_

    .. _Wikipedia: https://en.wikipedia.org/wiki/Borwein%27s_algorithm

    """
    @staticmethod
    def quadratic_convergence(*, precision: int = PRECISION) -> D:
        """

        :param precision:
        :return:
        """
        with decimal.localcontext() as ctx:
            ctx.precision = precision + 2

            # Initial conditions
            a = D(2).sqrt()
            b = D(0)
            p = D(2) + D(2).sqrt()

            while True:
                a_ = (a.sqrt() + 1 / a.sqrt()) / 2
                b_ = (1 + b) * a.sqrt() / (a + b)
                p_ = (1 + a_) * p * b_ / (1 + b_)

                if p == p_:
                    break

                a, b, p = a_, b_, p_

            return p_

    @staticmethod
    def cubic_convergence(*, precision: int = PRECISION) -> D:
        """

        :param precision:
        :return:
        """
        with decimal.localcontext() as ctx:
            ctx.prec = precision + 2

            # Initial conditions
            k = 0
            a = 1 / D(3)
            s = (D(3).sqrt() - 1) / 2

            while True:
                r_ = 3 / (1 + 2 * (1 - s ** 3) ** (1 / D(3)))
                s_ = (r_ - 1) / 2
                a_ = r_ ** 2 * a - 3 ** k * (r_ ** 2 - 1)

                if a + D(1) == a_ + D(1):
                    break

                a, s = a_, s_

            return 1 / a_


def chudnovsky_algorithm(*, precision: int = PRECISION) -> D:
    r"""
    `Wikipedia`_

    .. _Wikipedia: https://en.wikipedia.org/wiki/Chudnovsky_algorithm

    :param precision:
    :return:
    """
    with decimal.localcontext() as ctx:
        ctx.prec = precision + 2

        # Initial conditions
        sum_ = D(0)
        k: int = 0

        while True:
            term = (
                (
                        D(-1) ** k
                        * D(factorial(6 * k))
                        * (545140134 * k + 13591409)
                )
                / (
                        D(factorial(3 * k))
                        * D(factorial(k)) ** 3
                        * D(640320) ** D(3 * k + 3 / 2)
                )
            )

            if term + D(1) == D(1):
                break

            sum_ += term
            k += 1

    return +(1 / (12 * sum_))


def euler_formula(*, precision: int = PRECISION) -> D:
    r"""

    :param precision:
    :return:
    """
    with decimal.localcontext() as ctx:
        ctx.prec = precision + 10

        sum_ = _summation(
            lambda k: 1 / D(k) ** 2,
            ctx
        )

    return +(6 * sum_).sqrt()


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

            if a_ + b_ == 2 * a_ == 2 * b_:
                return (a_ + b_) ** 2 / (4 * t_)

            a, b, t, p = a_, b_, t_, p_


def leibniz_formula(*, precision: int = PRECISION) -> D:
    """
    `Wikipedia`_

    .. _Wikipedia: https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80

    :param precision:
    :return:
    """
    with decimal.localcontext() as ctx:
        ctx.prec = precision + 10

        sum_ = _summation(
            lambda k: (1 / D(2 * k + 1)) * D(-1) ** k,
            ctx
        )

    return +(4 * sum_)


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


def newton_formula(*, precision: int = PRECISION) -> D:
    r"""

    :param precision:
    :return:
    """
    with decimal.localcontext() as ctx:
        ctx.prec = precision + 10

        sum_ = _summation(
            lambda k: D(2) ** k * D(factorial(k)) ** 2 / D(factorial(2 * k + 1)),
            ctx
        )

    return +(2 * sum_)


def nilakantha_formula(*, precision: int = PRECISION) -> D:
    r"""

    :param precision:
    :return:
    """
    with decimal.localcontext() as ctx:
        ctx.prec = precision + 10

        sum_ = _summation(
            lambda k: 1 / D((2 * k + 2) * (2 * k + 3) * (2 * k + 4)) * D(-1) ** k,
            ctx
        )
        res = 4 * sum_ + D(3)

    with decimal.localcontext() as ctx:
        ctx.prec = precision + 1

        return +res


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
        term = 0

        while True:
            term = (D(2) + 2 * term).sqrt() / 2

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
