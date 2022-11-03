"""

"""

import decimal
from decimal import Decimal

from .constants import PRECISION


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
