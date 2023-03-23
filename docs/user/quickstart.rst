.. _quickstart:

Quickstart
==========

1. Install **pytrig**

To use **pytrig**, you'll need to install it first:

.. code-block:: console

    $ pip install pytrig

See the installation guide for additional help installing **pytrig**.

2. Import **pytrig**

Once you have **pytrig** installed, you can start using it in your Python scripts.
To do this, simply import it at the top of your script:

.. code-block:: python

    import pytrig

3. Using **pytrig** functions

The **pytrig** module provides a variety of trigonometric and hyperbolic functions that you can use in your scripts.

For example, to calculate the sine of an angle in radians, you can use the :py:func:`pytrig.sine` (or :py:func:`pytrig.sin`) function:

.. doctest::
    :pyversion: >= 3.8

    >>> angle = 0.5
    >>> sin_angle = pytrig.sine(angle)      # OR sin_angle = pytrig.sin(angle)
    >>> sin_angle
    Decimal('0.4794255386042030002732879352155713880818033679406006751886166131255350002878148322096312746843482692')

Below are additional example usages of **pytrig** functions.

*****

Computing :math:`\pi` Approximations
------------------------------------

.. doctest::
    :pyversion: >= 3.8

    >>> import pytrig
    >>> pytrig.PI
    Decimal('3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706801')
    >>> pytrig.pi(50)
    Decimal('3.141592653589793238462643383279502884197169399375105')

*****

Computing Approximations of Special Functions
---------------------------------------------

Natural Logarithm
^^^^^^^^^^^^^^^^^

+---------------------------------------+-------------------+
| Function                              | Shorthand         |
+=======================================+===================+
| :py:func:`pytrig.natural_logarithm`   | ``pytrig.ln()``   |
+---------------------------------------+-------------------+

.. doctest::
    :pyversion: >= 3.8

    >>> import pytrig
    >>> pytrig.natural_logarithm(2, 10)     # OR: pytrig.ln(2, 10)
    Decimal('0.6931471807')

Trigonometric Functions
^^^^^^^^^^^^^^^^^^^^^^^

+---------------------------+-------------------+-----------------------------------+-------------------+
| Trigonometric Function    | Shorthand         | Reciprocal Trigonometric Function | Shorthand         |
+===========================+===================+===================================+===================+
| :py:func:`pytrig.sine`    | ``pytrig.sin()``  | :py:func:`pytrig.cosecant`        | ``pytrig.csc()``  |
+---------------------------+-------------------+-----------------------------------+-------------------+
| :py:func:`pytrig.cosine`  | ``pytrig.cos()``  | :py:func:`pytrig.secant`          | ``pytrig.sec()``  |
+---------------------------+-------------------+-----------------------------------+-------------------+
| :py:func:`pytrig.tangent` | ``pytrig.tan()``  | :py:func:`pytrig.cotangent`       | ``pytrig.cot()``  |
+---------------------------+-------------------+-----------------------------------+-------------------+

.. doctest::
    :pyversion: >= 3.8

    >>> import pytrig
    >>> pytrig.sine(1, 10)          # OR: pytrig.sin(1, 10)
    Decimal('0.8414709845')
    >>> pytrig.cosine(1, 25)        # OR: pytrig.cos(1, 25)
    Decimal('0.5403023058681397174009367')
    >>> pytrig.tangent(1, 50)       # OR: pytrig.tan(1, 50)
    Decimal('1.5574077246549022305069748074583601730872507723814')

Inverse Trigonometric Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------------------------------+-----------------------+-------------------------------------------+-----------------------+
| Inverse Trigonometric Function    | Shorthand             | Inverse Reciprocal Trigonometric Function | Shorthand             |
+===================================+=======================+===========================================+=======================+
| :py:func:`pytrig.arcsine`         | ``pytrig.arcsin()``   | :py:func:`pytrig.arccosecant`             | ``pytrig.arccsc()``   |
+-----------------------------------+-----------------------+-------------------------------------------+-----------------------+
| :py:func:`pytrig.arccosine`       | ``pytrig.arccos()``   | :py:func:`pytrig.arcsecant`               | ``pytrig.arcsec()``   |
+-----------------------------------+-----------------------+-------------------------------------------+-----------------------+
| :py:func:`pytrig.arctangent`      | ``pytrig.arctan()``   | :py:func:`pytrig.arccotangent`            | ``pytrig.arccot()``   |
+-----------------------------------+-----------------------+-------------------------------------------+-----------------------+

.. doctest::
    :pyversion: >= 3.8

    >>> import pytrig
    >>> pytrig.arcsine(0.5, 10)         # OR: pytrig.arcsin(0.5, 10)
    Decimal('0.5235987753')
    >>> pytrig.arccosine(0.5, 25)       # OR: pytrig.arccos(0.5, 25)
    Decimal('1.04719755119659774615421479')
    >>> pytrig.arctangent(0.5, 50)      # OR: pytrig.arctan(0.5, 50)
    Decimal('0.46364760900080611621425623146121440202853705428609')

Hyperbolic Functions
^^^^^^^^^^^^^^^^^^^^

+---------------------------------------+-------------------+-------------------------------------------+-------------------+
| Hyperbolic Function                   | Shorthand         | Reciprocal Hyperbolic Function            | Shorthand         |  
+=======================================+===================+===========================================+===================+
| :py:func:`pytrig.hyperbolic_sine`     | ``pytrig.sinh()`` | :py:func:`pytrig.hyperbolic_cosecant`     | ``pytrig.csch()`` |
+---------------------------------------+-------------------+-------------------------------------------+-------------------+
| :py:func:`pytrig.hyperbolic_cosine`   | ``pytrig.cosh()`` | :py:func:`pytrig.hyperbolic_secant`       | ``pytrig.sech()`` |
+---------------------------------------+-------------------+-------------------------------------------+-------------------+
| :py:func:`pytrig.hyperbolic_tangent`  | ``pytrig.tanh()`` | :py:func:`pytrig.hyperbolic_cotangent`    | ``pytrig.coth()`` |
+---------------------------------------+-------------------+-------------------------------------------+-------------------+

.. doctest::
    :pyversion: >= 3.8
    
    >>> import pytrig
    >>> pytrig.hyperbolic_sine(1, 10)           # OR: pytrig.sinh(1, 50)
    Decimal('1.175201194')
    >>> pytrig.hyperbolic_cosine(1, 25)         # OR: pytrig.cosh(1, 50)
    Decimal('1.543080634815243778477908')
    >>> pytrig.hyperbolic_tangent(1, 50)        # OR: pytrig.tanh(1, 50)
    Decimal('0.76159415595576488811945828260479359041276859725791')

Inverse Hyperbolic Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-------------------------------------------+-----------------------+-------------------------------------------+-----------------------+
| Inverse Hyperbolic Function               | Shorthand             | Inverse Reciprocal Hyperbolic Function    | Shorthand             |
+===========================================+=======================+===========================================+=======================+
| :py:func:`pytrig.hyperbolic_arcsine`      | ``pytrig.arsinh()``   | :py:func:`pytrig.hyperbolic_arccosecant`  | ``pytrig.arcsch()``   |
+-------------------------------------------+-----------------------+-------------------------------------------+-----------------------+
| :py:func:`pytrig.hyperbolic_arccosine`    | ``pytrig.arcosh()``   | :py:func:`pytrig.hyperbolic_arcsecant`    | ``pytrig.arsch()``    |
+-------------------------------------------+-----------------------+-------------------------------------------+-----------------------+
| :py:func:`pytrig.hyperbolic_arctangent`   | ``pytrig.artanh()``   | :py:func:`pytrig.hyperbolic_arccotangent` | ``pytrig.arcoth()``   |
+-------------------------------------------+-----------------------+-------------------------------------------+-----------------------+


.. doctest::
    :pyversion: >= 3.8

    >>> import pytrig
    >>> pytrig.hyperbolic_arcsine(1, 10)            # OR: pytrig.arsinh(1, 10)
    Decimal('0.8813735872')
    >>> pytrig.hyperbolic_arccosine(2, 25)          # OR: pytrig.arcosh(2, 25)
    Decimal('1.316957896924816708625051')
    >>> pytrig.hyperbolic_arctangent(0.5, 50)       # OR: pytrig.artanh(0.5, 50)
    Decimal('0.54930614433405484569762261846126285232374527891133')