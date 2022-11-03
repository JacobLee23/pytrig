"""

"""

import decimal
from decimal import Decimal
from math import factorial

from .constants import PRECISION


def chudnovsky_algorithm(*, precision: int = PRECISION) -> Decimal:
    r"""
    `Wikipedia`_

    .. _Wikipedia: https://en.wikipedia.org/wiki/Chudnovsky_algorithm

    :param precision:
    :return:
    """
    with decimal.localcontext() as ctx:
        ctx.prec = precision + 2

        sum_ = Decimal(0)
        k: int = 0

        while True:
            term = (
                (
                        Decimal(-1) ** k
                        * Decimal(factorial(6 * k))
                        * (545140134 * k + 13591409)
                )
                / (
                        Decimal(factorial(3 * k))
                        * Decimal(factorial(k)) ** 3
                        * Decimal(640320) ** Decimal(3 * k + 3 / 2)
                )
            )

            if term + Decimal(1) == Decimal(1):
                break

            sum_ += term
            k += 1

        return 1 / (12 * sum_)


def euler_formula(*, precision: int = PRECISION) -> Decimal:
    r"""

    :param precision:
    :return:
    """
    with decimal.localcontext() as ctx:
        ctx.prec = precision + 2

        sum_ = Decimal(0)
        k: int = 1

        while True:
            term = 1 / Decimal(k) ** 2
            print(term)

            if term + Decimal(1) == Decimal(1):
                break

            sum_ += term
            k += 1

        return (6 * sum_).sqrt()


def leibniz_formula(*, precision: int = PRECISION) -> Decimal:
    """
    `Wikipedia`_

    .. _Wikipedia: https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80

    :param precision:
    :return:
    """
    with decimal.localcontext() as ctx:
        ctx.prec = precision + 2

        sum_ = Decimal(0)
        k: int = 0
        while True:
            term = (1 / Decimal(2 * k + 1)) * Decimal(-1) ** k
            print(term)

            if term + Decimal(1) == Decimal(1):
                break

            sum_ += term
            k += 1

        return 4 * sum_


def madhava_series(*, precision: int = PRECISION) -> Decimal:
    r"""
    `Madhava Series`_

    .. _Madhava Series: https://en.wikipedia.org/wiki/Madhava_series

    :param precision:
    :return:
    """
    with decimal.localcontext() as ctx:
        ctx.prec = precision + 2

        # Initial conditions
        sum_ = Decimal(0)
        k: int = 0

        while True:
            term = (-1 / Decimal(3)) ** k / (2 * k + 1)

            if term + Decimal(1) == Decimal(1):
                break

            sum_ += term
            k += 1

        return Decimal(12).sqrt() * sum_


def newton_formula(*, precision: int = PRECISION) -> Decimal:
    r"""

    :param precision:
    :return:
    """
    with decimal.localcontext() as ctx:
        ctx.prec = precision + 2

        sum_ = Decimal(0)
        k: int = 0

        while True:
            term = Decimal(2) ** k * Decimal(factorial(k)) ** 2 / Decimal(factorial(2 * k + 1))

            if term + Decimal(1) == Decimal(1):
                break

            sum_ += term
            k += 1

        return 2 * sum_


def nilakantha_formula(*, precision: int = PRECISION) -> Decimal:
    r"""

    :param precision:
    :return:
    """
    with decimal.localcontext() as ctx:
        ctx.prec = precision + 2

        sum_ = Decimal(0)
        k: int = 0

        while True:
            term = 1 / Decimal((2 * k + 2) * (2 * k + 3) * (2 * k + 4)) * Decimal(-1) ** k

            if term + Decimal(1) == Decimal(1):
                break

            sum_ += term
            k += 1

        return 4 * sum_ + Decimal(3)


def ramanujan_formula(*, precision: int) -> Decimal:
    """

    :param precision:
    :return:
    """
    with decimal.localcontext() as ctx:
        ctx.prec = precision + 2

        sum_ = Decimal(0)
        k: int = 0

        while True:
            term = (
                (Decimal(factorial(4 * k)) * Decimal(1103 + 26390 * k))
                / (Decimal(factorial(k)) ** 4 * Decimal(396) ** (4 * k))
            )

            if term + Decimal(1) == Decimal(1):
                break

            sum_ += term
            k += 1

        return 1 / (2 * Decimal(2).sqrt() / 9801 * sum_)


def viete_formula(*, precision: int = PRECISION) -> Decimal:
    r"""
    `Wikipedia`_

    .. _Wikipedia: https://en.wikipedia.org/wiki/Vi%C3%A8te%27s_formula

    :param precision:
    :return:
    """
    with decimal.localcontext() as ctx:
        ctx.prec = precision + 2

        # Initial conditions
        product_ = Decimal(1)
        a = Decimal(2).sqrt()

        while True:
            product_ *= a / 2

            # Test for convergence
            if 2 / a == Decimal(1):
                break

            a = (Decimal(2) + a).sqrt()

        return 2 / product_


def wallis_product(*, precision: int = PRECISION) -> Decimal:
    r"""
    `Wikipedia`_

    .. _Wikipedia: https://en.wikipedia.org/wiki/Wallis_product

    :param precision:
    :return:
    """
    with decimal.localcontext() as ctx:
        ctx.prec = precision + 2

        product_ = Decimal(1)
        k: int = 1
        while True:
            term = Decimal(4 * k) / Decimal(4 * k ** 2 - 1)

            if 2 / term == Decimal(1):
                break

            product_ *= term
            k += 1

        return 2 / product_
