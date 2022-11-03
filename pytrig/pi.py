"""

"""

import decimal
from decimal import Decimal

from .constants import PRECISION


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
