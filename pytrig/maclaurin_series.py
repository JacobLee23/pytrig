"""

"""

import decimal
from decimal import Decimal
import math
import typing


def maclaurin_expansion(func: typing.Callable[[int, Decimal, int], Decimal]) -> typing.Callable:
    """

    :param func:
    :return:
    """

    def wrapper(x: Decimal, *, precision: int = 100) -> typing.Generator[Decimal, None, None]:
        """

        :param x:
        :param precision:
        :return:
        """
        with decimal.localcontext() as ctx:
            ctx.prec = precision + 2

            n = 0
            while True:
                try:
                    term = func(n, x, precision)
                except decimal.Overflow:
                    return

                # Test for convergence
                if term + Decimal(1) == Decimal(1):
                    return

                yield term

                n += 1

    return wrapper


@maclaurin_expansion
def sine(n: int, x: Decimal, precision: int) -> Decimal:
    r"""

    :param n:
    :param x:
    :param precision:
    :return:
    """
    with decimal.localcontext() as ctx:
        ctx.prec = precision + 2

        return Decimal(
            (-1) ** n
        ) / Decimal(
            math.factorial(2 * n + 1)
        ) * (x ** (2 * n + 1))


@maclaurin_expansion
def cosine(n: int, x: Decimal, precision: int) -> Decimal:
    r"""

    :param n:
    :param x:
    :param precision:
    :return:
    """
    with decimal.localcontext() as ctx:
        ctx.prec = precision + 2

        return Decimal(
            (-1) ** n
        ) / Decimal(
            math.factorial(2 * n)
        ) * (x ** (2 * n))


@maclaurin_expansion
def arcsine(n: int, x: Decimal, precision: int) -> Decimal:
    r"""

    :param n:
    :param x:
    :param precision:
    :return:
    """
    with decimal.localcontext() as ctx:
        ctx.prec = precision + 2

        return Decimal(
            math.factorial(2 * n)
        ) / (
                Decimal(4 ** n) * Decimal(math.factorial(n)) ** 2 * Decimal(2 * n + 1)
        ) * (x ** (2 * n + 1))


@maclaurin_expansion
def arctangent(n: int, x: Decimal, precision: int) -> Decimal:
    r"""

    :param n:
    :param x:
    :param precision:
    :return:
    """
    with decimal.localcontext() as ctx:
        ctx.prec = precision + 2

        return Decimal(
            (-1) ** n
        ) / Decimal(
            2 * n + 1
        ) * (x ** (2 * n + 1))
