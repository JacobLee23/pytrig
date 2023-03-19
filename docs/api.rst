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

Angle Conversions
-----------------

.. autofunction:: to_degrees(theta: decimal.Decimal, prec: int)
.. autofunction:: to_radians(theta: decimal.Decimal, prec: int)

Pi Approximations
-----------------

.. autofunction:: pi

.. py:data:: PI

    An approximation of :math:`\pi` as computed by :py:func:`pi`, to :py:data:`PRECISION` decimal places of precision.

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

.. autofunction:: natural_logarithm


Trigonometric Functions
-----------------------

.. autofunction:: sine
.. autofunction:: cosine
.. autofunction:: tangent
.. autofunction:: secant
.. autofunction:: cosecant
.. autofunction:: cotangent

Inverse Trigonometric Functions
-------------------------------

.. autofunction:: arcsine
.. autofunction:: arccosine
.. autofunction:: arctangent
.. autofunction:: arcsecant
.. autofunction:: arccosecant
.. autofunction:: arccotangent

Hyperbolic Functions
--------------------

.. autofunction:: hyperbolic_sine
.. autofunction:: hyperbolic_cosine
.. autofunction:: hyperbolic_tangent
.. autofunction:: hyperbolic_secant
.. autofunction:: hyperbolic_cosecant
.. autofunction:: hyperbolic_cotangent

Inverse Hyperbolic Functions
----------------------------

.. autofunction:: hyperbolic_arcsine
.. autofunction:: hyperbolic_arccosine
.. autofunction:: hyperbolic_arctangent
.. autofunction:: hyperbolic_arcsecant
.. autofunction:: hyperbolic_arccosecant
.. autofunction:: hyperbolic_arccotangent
