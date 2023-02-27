.. _api:

API Reference
=============

.. automodule:: pytrig

-----

Constants
---------

.. py:data:: INF

    A :class:`decimal.Decimal` representation of infinity.
    Equivalent to ``decimal.Decimal("Infinity")``:

    .. doctest::
        :pyversion: >= 3.8

        >>> import decimal
        >>> import pytrig
        >>> pytrig.INF == decimal.Decimal("Infinity")
        True

    :type: decimal.Decimal

.. py:data:: NAN

    A :class:`decimal.Decimal` representation for a nonexistent value.
    Equivalent to ``decimal.Decimal("NaN")``:

    .. doctest::
        :pyversion: >= 3.8

        >>> import decimal
        >>> import pytrig
        >>> pytrig.NAN == decimal.Decimal("NaN")
        True

    :type: decimal.Decimal

Computation Precision
---------------------

.. py:data:: PRECISION

    The default number of decimal places of precision used by computations.

    :type: int
    :value: 100

.. autofunction:: precision

Angle Conversions
-----------------

.. autofunction:: to_degrees(theta: decimal.Decimal, prec: int)
.. autofunction:: to_radians(theta: decimal.Decimal, prec: int)

Pi Approximations
-----------------

.. autofunction:: chudnovsky_algorithm

.. py:data:: PI

    An approximation of :math:`\pi` as computed by :py:func:`chudnovsky_algorithm`, to :py:data:`PRECISION` decimal places of precision.

    :type: decimal.Decimal

Maclaurin Series Expansions
---------------------------

.. autofunction:: ms_natural_logarithm
.. autofunction:: ms_sine
.. autofunction:: ms_cosine
.. autofunction:: ms_arcsine
.. autofunction:: ms_arctangent
.. autofunction:: ms_hyperbolic_sine
.. autofunction:: ms_hyperbolic_cosine
.. autofunction:: ms_hyperbolic_arcsine
.. autofunction:: ms_hyperbolic_arctangent

.. autoclass:: MaclaurinExpansion
    :members:

Natural Logarithm Approximations
--------------------------------

.. autofunction:: natural_logarithm(x: decimal.Decimal, prec: int)


Trigonometric Functions
-----------------------

.. autofunction:: sine(x: decimal.Decimal, prec: int)
.. autofunction:: cosine(x: decimal.Decimal, prec: int)
.. autofunction:: tangent(x: decimal.Decimal, prec: int)
.. autofunction:: secant(x: decimal.Decimal, prec: int)
.. autofunction:: cosecant(x: decimal.Decimal, prec: int)
.. autofunction:: cotangent(x: decimal.Decimal, prec: int)

Inverse Trigonometric Functions
-------------------------------

.. autofunction:: arcsine(x: decimal.Decimal, prec: int)
.. autofunction:: arccosine(x: decimal.Decimal, prec: int)
.. autofunction:: arctangent(x: decimal.Decimal, prec: int)
.. autofunction:: arcsecant(x: decimal.Decimal, prec: int)
.. autofunction:: arccosecant(x: decimal.Decimal, prec: int)
.. autofunction:: arccotangent(x: decimal.Decimal, prec: int)

Hyperbolic Functions
--------------------

.. autofunction:: hyperbolic_sine(x: decimal.Decimal, prec: int)
.. autofunction:: hyperbolic_cosine(x: decimal.Decimal, prec: int)
.. autofunction:: hyperbolic_tangent(x: decimal.Decimal, prec: int)
.. autofunction:: hyperbolic_secant(x: decimal.Decimal, prec: int)
.. autofunction:: hyperbolic_cosecant(x: decimal.Decimal, prec: int)
.. autofunction:: hyperbolic_cotangent(x: decimal.Decimal, prec: int)

Inverse Hyperbolic Functions
----------------------------

.. autofunction:: hyperbolic_arcsine(x: decimal.Decimal, prec: int)
.. autofunction:: hyperbolic_arccosine(x: decimal.Decimal, prec: int)
.. autofunction:: hyperbolic_arctangent(x: decimal.Decimal, prec: int)
.. autofunction:: hyperbolic_arcsecant(x: decimal.Decimal, prec: int)
.. autofunction:: hyperbolic_arccosecant(x: decimal.Decimal, prec: int)
.. autofunction:: hyperbolic_arccotangent(x: decimal.Decimal, prec: int)
