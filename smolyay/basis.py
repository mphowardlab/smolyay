import abc
import collections.abc

import numpy
import scipy.special


class BasisFunction(abc.ABC):
    """Basis function for interpolating data.

    A one-dimensional basis function is defined on some given natural
    domain. The function defines the :attr:`points` at which it should
    be sampled within this interval for interpolation. The function also
    has an associated :meth:`__call__` method for evaluating it at a
    point within its domain. Moreover, the first derivative of the function
    can be evaluated via :meth:`derivative`.
    """

    @property
    @abc.abstractmethod
    def domain(self):
        """numpy.ndarray: Domain the sample points come from."""
        pass

    def __call__(self, x):
        """Evaluate the basis function.

        Parameters
        ----------
        x : float
            One-dimensional point.

        Returns
        -------
        float
            Value of basis function.
        """
        if not numpy.all(self.in_domain(x)):
            raise ValueError("Input is outside the domain " + str(self.domain))
        return self._function(x)

    def derivative(self, x):
        """Evaluate the first derivative of the basis function.

        Parameters
        ----------
        x : float
            one-dimensional point.

        Returns
        -------
        float
            Value of the derivative of the basis function.
        """
        if not numpy.all(self.in_domain(x)):
            raise ValueError("Input is outside the domain")
        return self._derivative(x)

    def in_domain(self, x):
        """Check if the input is within the natural domain.

        Parameters
        ----------
        x : float, numpy:ndarray
            One-dimensional points.

        Returns
        -------
        bool
            True if input was outside domain, False otherwise
        """
        return numpy.logical_and(
            numpy.greater_equal(x, self.domain[0]), numpy.less_equal(x, self.domain[1])
        )

    @abc.abstractmethod
    def _function(self, x):
        """Evaluate the basis function.

        Parameters
        ----------
        x : float
            One-dimensional point.

        Returns
        -------
        float
            Value of basis function.
        """
        pass

    @abc.abstractmethod
    def _derivative(self, x):
        """Evaluate the first derivative of the basis function.

        Parameters
        ----------
        x : float
            one-dimensional point.

        Returns
        -------
        float
            Value of the derivative of the basis function.
        """
        pass


class ChebyshevFirstKind(BasisFunction):
    r"""Chebyshev polynomial of the first kind.

    The Chebyshev polynomial :math:`T_n` of degree *n* is defined by the
    recursive relationship:

    .. math::

        T_0(x) = 1
        T_1(x) = x
        T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)

    Their domain is defined to be :math:`-1 \le x \le 1`. The degree *n* is
    represented by property `degree`.

    Parameters
    ----------
    degree : int
        Degree of the Chebyshev polynomial.
    """

    def __init__(self, degree):
        super().__init__()
        self.degree = degree

    @property
    def domain(self):
        """numpy.ndarray: Domain the sample points come from."""
        return numpy.array([-1, 1])

    @property
    def is_complex(self):
        """bool: if basis function returns complex values."""
        return False

    @property
    def degree(self):
        """int: Degree of polynomial."""
        return self._degree

    @degree.setter
    def degree(self, value):
        self._degree = int(value)

    def _function(self, x):
        r"""Evaluate the basis function.

        The Chebyshev polynomial is evaluated using the combinatorial formula:

        .. math::

            T_n(x) = \sum_{k=0}^{\lfloor n/2 \rfloor} {n \choose 2k} (x^2-1)^k x^{n-2k}

        for :math:`n \ge 2`, and by the direct formula for the other values of *n*.

        Parameters
        ----------
        x : float
            One-dimensional point on :math:`[-1, 1]`.

        Returns
        -------
        float
            Value of Chebyshev polynomial of the first kind.

        Raises
        ------
        ValueError
            if input is outside the domain [-1, 1]

        """
        return scipy.special.eval_chebyt(self.degree, x)

    def _derivative(self, x):
        """Evaluate the derivative of ChebyshevFirstKind.

        The first derivative of Chebyshev polynomials of first kind is
        evaluated using the relation between Chebyshev polynomial of
        first kind and second kind.

        ..math::
            T_n'(x) = nU_{n-1}(x)

        Parameters
        ----------
        x: float
            input in [-1, 1] domain.

        Returns
        -------
        float
            Value of the derivative of Chebyshev polynomials of first kind.

        Raises
        ------
        ValueError
            if input is outside the domain [-1, 1].
        """
        return self.degree * scipy.special.eval_chebyu(self.degree - 1, x)


class ChebyshevSecondKind(BasisFunction):
    r"""Chebyshev polynomial of the second kind.

    The Chebyshev polynomial :math:`U_n` of degree *n* is defined by the
    recursive relationship:

    .. math::

        U_0(x) = 1
        U_1(x) = 2x
        U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x)

    Their domain is defined to be :math:`-1 \le x \le 1`. The degree *n* is
    represented by property `degree`.

    Parameters
    ----------
    degree : int
        Degree of the Chebyshev polynomial.
    """

    def __init__(self, degree):
        super().__init__()
        self.degree = degree

    @property
    def domain(self):
        """numpy.ndarray: Domain the sample points come from."""
        return numpy.array([-1, 1])
    
    @property
    def is_complex(self):
        """bool: if basis function returns complex values."""
        return False
    
    @property
    def degree(self):
        """int: Degree of polynomial."""
        return self._degree

    @degree.setter
    def degree(self, value):
        self._degree = int(value)

    def _function(self, x):
        r"""Evaluate the basis function.

        The Chebyshev polynomial is evaluated using the combinatorial formula:

        .. math::

            U_n(x) = \sum_{k=0}^{\lfloor n/2 \rfloor} {n+1 \choose 2k+1} (x^2-1)^k x^{n-2k}

        for :math:`n \ge 2`, and by the direct formula for the other values of *n*.

        Parameters
        ----------
        x : float
            One-dimensional point on :math:`[-1, 1]`.

        Returns
        -------
        float
            Value of Chebyshev polynomial of the second kind.

        Raises
        -------
        float
            Value of Chebyshev polynomial of the second kind.

        Raises
        ------
        ValueError
            if input is outside the domain [-1, 1]

        """
        return scipy.special.eval_chebyu(self.degree, x)

    def _derivative(self, x):
        r"""Evaluate the derivative of Chebyshev Second Kind.

        The first derivative of Chebyshev polynomials of second kind is
        evaluated using the connection between Chebyshev polynomial of
        first kind and second kind.

        ..math::
            U_n'(x) = \frac{(n+1)T_{n+1}(x)-xU_n(x)}{x^{2}-1}

        The above equation does not converge for :math:x={-1, 1}.
        Derivative can be found using L'Hôpital's rule.
        ..math::
            \lim_{x \to 1} U_n'(x) = \frac{n(n+1)(n+2)}{3}
            \lim_{x \to -1} U_n'(x) = (-1)^{n+1} \frac{n(n+1)(n+2)}{3}

        Parameters
        ----------
        x: float
            input in [-1, 1] domain.

        Returns
        -------
        float
            Value of the derivative of Chebyshev polynomials of second kind.

        Raises
        ------
        ValueError
            if input is outside the domain [-1, 1].
        """
        x = numpy.asarray(x)
        y = numpy.zeros(x.shape)
        u_limit = self.degree * (self.degree + 1) * (self.degree + 2) / 3
        flag_upper = x == 1
        y[flag_upper] = u_limit

        flag_lower = x == -1
        y[flag_lower] = (-1) ** (self.degree + 1) * u_limit

        flag = ~(flag_upper | flag_lower)
        y[flag] = (
            (self.degree + 1) * scipy.special.eval_chebyt(self.degree + 1, x[flag])
            - x[flag] * scipy.special.eval_chebyu(self.degree, x[flag])
        ) / (x[flag] ** 2 - 1)
        if y.ndim == 0:
            y = y.item()
        return y


class Trigonometric(BasisFunction):
    r"""Trigonometric basis functions.

    The Trigonometric polynomials represents periodic functions
    as sums of sine and cosine terms, where *n* is the frequency
    of trigonometric polynomial and is any integer.

    .. math::

        \phi_n(x) = \exp(xi * n)

    Parameters
    ----------
    frequency : int
        Degree of trigonometric polynomial.

    """

    def __init__(self, frequency):
        super().__init__()
        self.frequency = frequency

    @property
    def domain(self):
        """numpy.ndarray: Domain the sample points come from."""
        return numpy.array([0, 2 * numpy.pi])

    @property
    def frequency(self):
        """int: frequency of polynomial."""
        return self._frequency
    
    @property
    def is_complex(self):
        """bool: if basis function returns complex values."""
        return True
    
    @frequency.setter
    def frequency(self, value):
        self._frequency = int(value)

    def _function(self, x):
        r"""Evaluate the basis function.

        The Trigonometric polynomial is evaluated using the following formula:

        .. math::

            \phi_n(x) = \exp(xi * n)

        where *n* is the frequency of the trigonometric polynomial and
        is any integer.

        Parameters
        ----------
        x : float
            One-dimensional point on :math:`[0, 2\pi]`.

        Returns
        -------
        float
            Value of Trigonometric polynomial.

        Raises
        ------
        ValueError
            If input is outside the domain `[0, 2\pi]`
        """
        x = numpy.asarray(x)
        return numpy.exp(x * self.frequency * 1j)

    def _derivative(self, x):
        r"""Evaluate the derivetive of the trigonometric polynomials.

        Parameters
        ----------
        x : float
            One-dimensional point on :math:`[0, 2\pi]`.

        Returns
        -------
        float
            Value of the derivative of Trigonometric polynomial.

        Raises
        ------
        ValueError
            If input is outside the domain `[0, 2\pi]`
        """
        x = numpy.asarray(x)
        return self.frequency * 1j * numpy.exp(x * self.frequency * 1j)


class BasisFunctionSet(collections.abc.Sequence):
    """Set of basis functions and sample points.

    Parameters
    ----------
    basis_functions : list
        Basis functions in set.

    """

    def __init__(self, basis_functions):
        self._basis_functions = basis_functions

    @property
    def basis_functions(self):
        """list: Basis functions."""
        return self._basis_functions
    
    @property
    def is_complex(self):
        """bool: if any basis function return complex values"""
        return any(bf.is_complex for bf in self._basis_functions)
    
    def __len__(self):
        return len(self.basis_functions)

    def __getitem__(self, key):
        return self.basis_functions[key]


class NestedBasisFunctionSet(BasisFunctionSet):
    """Set of nested basis functions and sample points.

    Parameters
    ----------
    basis_functions : list
        Basis functions in set.

    num_per_level : list
        number of unique functions per level

    Raises
    ------
    IndexError
        number of basis function does not match functions in each level
    """

    def __init__(self, basis_functions, num_per_level):
        super().__init__(basis_functions)
        if sum(num_per_level) != len(basis_functions):
            raise IndexError(
                str(sum(num_per_level))
                + " total functions in levels, "
                + str(len(basis_functions))
                + " functions given."
            )

        self._num_per_level = numpy.array(num_per_level, dtype=int)
        self._end_level = numpy.cumsum(self._num_per_level)
        self._start_level = self._end_level - self._num_per_level

    @property
    def num_per_level(self):
        """numpy.ndarray: number of points per level."""
        return self._num_per_level

    @property
    def num_levels(self):
        """int: number of levels."""
        return len(self.num_per_level)

    @property
    def start_level(self):
        """numpy.ndarray: the starting index of each level."""
        return self._start_level

    @property
    def end_level(self):
        """numpy.ndarray: the ending index of each level."""
        return self._end_level

    def level(self, index):
        """list of :class:BasisFunction: Functions in a level"""
        return self.basis_functions[self.start_level[index] : self.end_level[index]]
