import abc
import collections.abc

import numpy
import scipy.special

from smolyay import _growth


class BasisFunction(abc.ABC):
    """Basis function for interpolating data.

    A one-dimensional basis function is defined on some given natural
    domain. The function defines the :attr:`points` at which it should
    be sampled within this interval for interpolation. The function also
    has an associated :meth:`__call__` method for evaluating it at a
    point within its domain. Moreover, the first derivative of the function
    can be evaluated via :meth:`derivative`.
    """

    _is_complex = False

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

    def derivative(self, x, n=1):
        """Evaluate the first derivative of the basis function.

        Parameters
        ----------
        x : float
            one-dimensional point.

        n : int, optional
            order of derivative. Default is one.

        Returns
        -------
        float
            Value of the derivative of the basis function.
        """
        if not numpy.all(self.in_domain(x)):
            raise ValueError("Input is outside the domain")
        return self._derivative(x, n)

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

    def scale_to_domain(self, points, old_domain):
        """Scale points from a domain to BasisFunction domain

        Parameters
        ----------
        points: numeric or array-like
            points to be shifted to new domain.

        old_domain: ndarray of shape (2,)
            upper and lower bounds of points.

        Returns
        -------
        numeric or array-like
            points shifted to BasisFunction domain"""
        new_points = numpy.asarray(
            self.domain[0]
            + (self.domain[1] - self.domain[0])
            * ((points - old_domain[0]) / (old_domain[1] - old_domain[0]))
        )
        numpy.clip(new_points, self.domain[0], self.domain[1], out=new_points)
        if new_points.ndim == 0:
            new_points = new_points.item()
        return new_points

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

    def _derivative(self, x, n=1):
        """Evaluate the derivative of ChebyshevFirstKind.

        The derivative of Chebyshev polynomials of first kind is
        evaluated using the relation between Chebyshev polynomial of
        first kind and second kind.

        The 1st and 2nd derivative are supported. Higher order
        derivative will raise an error.

        ..math::
            T_n'(x) = nU_{n-1}(x)

        Parameters
        ----------
        x: float
            input in [-1, 1] domain.

        n : int, optional
            order of derivative. Default is 1.

        Returns
        -------
        float
            Value of the derivative of Chebyshev polynomials of first kind.

        Raises
        ------
        ValueError
            if input is outside the domain [-1, 1].
        NotImplementedError
            Order of derivative outside supported range (1, 2).
        """
        if n == 1:
            return self.degree * scipy.special.eval_chebyu(self.degree - 1, x)
        elif n == 2:
            x = numpy.asarray(x)
            y = numpy.zeros(x.shape)
            u_limit = (self.degree - 1) * (self.degree) * (self.degree + 1) / 3
            flag_upper = x == 1
            y[flag_upper] = u_limit * self.degree
            flag_lower = x == -1
            y[flag_lower] = (-1) ** (self.degree) * u_limit * self.degree

            flag = ~(flag_upper | flag_lower)
            y[flag] = (
                (
                    (self.degree) * scipy.special.eval_chebyt(self.degree, x[flag])
                    - x[flag] * scipy.special.eval_chebyu(self.degree - 1, x[flag])
                )
                / (x[flag] ** 2 - 1)
                * self.degree
            )
            if y.ndim == 0:
                y = y.item()
            return y
        else:
            raise NotImplementedError("nth derivative outside supported range (1, 2).")


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

    def _derivative(self, x, n=1):
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

        n : int, optional
            order of derivative. Default is 1.

        Returns
        -------
        float
            Value of the derivative of Chebyshev polynomials of second kind.

        Raises
        ------
        ValueError
            if input is outside the domain [-1, 1].

        NotImplementedError
            Order of derivative outside supported range (1).
        """
        if n != 1:
            raise NotImplementedError("Only first derivative is supported.")
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

    _is_complex = True

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

    def _derivative(self, x, n=1):
        r"""Evaluate the derivetive of the trigonometric polynomials.

        Parameters
        ----------
        x : float
            One-dimensional point on :math:`[0, 2\pi]`.

        Returns
        -------
        float
            Value of the derivative of Trigonometric polynomial.

        n : int, optional
            order of derivative. Default is 1.

        Raises
        ------
        ValueError
            If input is outside the domain `[0, 2\pi]`
        """
        x = numpy.asarray(x)
        return numpy.exp(x * self.frequency * 1j) * (self.frequency * 1j) ** n


class BasisFunctionSet(collections.abc.Sequence):
    """Set of basis functions and sample points."""

    def __init__(self, basis_functions= None):
        self._basis_functions = None
        pass

    @property
    def basis_functions(self):
        """list: Basis functions."""
        return self._basis_functions

    @property
    def domain(self):
        """numpy.ndarray: Domain of the `basis_functions`"""
        return self.basis_functions[0].domain

    def __len__(self):
        return len(self._basis_functions)

    def __getitem__(self, key):
        return self.basis_functions[key]

    def scale_to_domain(self, points, old_domain):
        return self.basis_functions[0].scale_to_domain(points, old_domain)

    def __call__(self, X, domain=None):
        """Evaluate all the basis functions in the set

        Calls all the basis function(s) at index and evaluates at X.

        Parameters
        ----------
        X : array-like
            the points to evaluate.

        X_domain : numpy array of shape (2,)
            the lower and upper bounds of X.

        Returns
        -------
        scalar or ndarray
            the values of the basis functions.
        """
        if not domain is None:
            new_X = self._scale_to_domain(numpy.asarray(X), domain)
        else:
            new_X = numpy.asarray(X)
        y = numpy.zeros(
            [len(self)] + list(new_X.shape),
            dtype=complex if any(bf._is_complex for bf in self) else float
        )
        for i, bf in enumerate(self._basis_functions):
            y[i] = bf(new_X)
        return y

    def derivative(self, X, X_domain, n=1):
        """Evaluate all the derivative of basis functions in the set

        Calls the derivative for all the basis function(s) and
        evaluates at X.

        Parameters
        ----------
        X : array-like
            the points to evaluate.

        X_domain : numpy array of shape (2,)
            the lower and upper bounds of X.

        n : int, optional
            order of derivative. Default is 1.

        Returns
        -------
        scalar or ndarray
            the values of the basis functions.
        """
        new_X = self._scale_to_domain(numpy.asarray(X), X_domain)
        if any(bf._is_complex for bf in self):
            y = numpy.zeros([len(self)] + list(new_X.shape), dtype="complex_")
        else:
            y = numpy.zeros([len(self)] + list(new_X.shape))
        for i in range(len(self)):
            y[i, :] = self[i].derivative(new_X, n)
        y *= ((self.domain[1] - self.domain[0]) / (X_domain[1] - X_domain[0])) ** n
        return y


class CustomBasisFunctionSet(BasisFunctionSet):
    """Set of basis functions

    Parameters
    ----------
    basis_functions : list
        Basis functions in set.

    Raises
    ------
    IndexError
        Must have at least one basis function
    ValueError
        Basis functions must have the same domain.
    """

    def __init__(self, basis_functions):
        if len(basis_functions) == 0:
            raise IndexError("Must have at least one basis function.")
        domain = basis_functions[0].domain
        if any(not numpy.array_equal(domain, b.domain) for b in basis_functions[1:]):
            raise ValueError("Basis functions must have the same domain.")
        self._basis_functions = basis_functions


class MutableBasisFunctionSet(BasisFunctionSet):
    """Set of basis functions where number of functions is mutable.

    Parameters
    ----------
    num_terms : int
        the number of terms in the set.
    """

    def __init__(self, num_terms):
        self._num_terms = None
        self._basis_functions = None

        self.num_terms = num_terms

    @property
    def basis_functions(self):
        """list: Basis functions."""
        if self._basis_functions is None:
            self._create()
        return self._basis_functions

    @property
    def num_terms(self):
        """int: the number of terms in the set."""
        return self._num_terms

    @num_terms.setter
    def num_terms(self, value):
        num_terms = int(value)
        if num_terms <= 0:
            raise ValueError("Must have at least one term.")
        if num_terms != self._num_terms:
            self._num_terms = num_terms
            self._basis_functions = None

    @abc.abstractmethod
    def _create(self):
        """Create the basis functions in the set."""
        pass


class ChebyshevFirstKindBasisFunctionSet(MutableBasisFunctionSet):
    """Set of Chebyshev polynomials of the first kind.

    Parameters
    ----------
    num_terms : int
        the number of terms in the set.
    """

    def _create(self):
        """Create the basis functions in the set."""
        self._basis_functions = [ChebyshevFirstKind(f) for f in range(self._num_terms)]


class ChebyshevSecondKindBasisFunctionSet(MutableBasisFunctionSet):
    """Set of Chebyshev polynomials of the second kind.

    Parameters
    ----------
    num_terms : int
        the number of terms in the set.
    """

    def _create(self):
        """Create the basis functions in the set."""
        self._basis_functions = [ChebyshevSecondKind(f) for f in range(self._num_terms)]


class TrigonometricBasisFunctionSet(MutableBasisFunctionSet):
    """Set of Trigonmetric equations.

    Parameters
    ----------
    num_terms : int
        the number of terms in the set.
    """

    def _create(self):
        """Create the basis functions in the set."""
        index_trig = numpy.arange(self.num_terms, dtype=int)
        frequencies = numpy.where(
            index_trig % 2 == 1, (1 + index_trig) // 2, -index_trig // 2
        )
        self._basis_functions = [Trigonometric(f) for f in frequencies]


class NestedBasisFunctionSet(BasisFunctionSet):
    """Set of nested basis functions and sample points."""

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


class NestedCustomBasisFunctionSet(NestedBasisFunctionSet):
    """Set of nested basis functions and sample points.

    Parameters
    ----------
    basis_functions : list
        Basis functions in set.

    num_per_level : list
        number of unique functions per level.

    Raises
    ------
    IndexError
        Must have at least one basis function
    ValueError
        Basis functions must have the same domain.
    IndexError
        number of basis function does not match functions in each level.
    """

    def __init__(self, basis_functions, num_per_level):
        if sum(num_per_level) != len(basis_functions):
            raise IndexError(
                str(sum(num_per_level))
                + " total functions in levels, "
                + str(len(basis_functions))
                + " functions given."
            )
        if len(basis_functions) == 0:
            raise IndexError("Must have at least one term.")
        domain = basis_functions[0].domain
        if any(not numpy.array_equal(domain, b.domain) for b in basis_functions[1:]):
            raise ValueError("Basis functions must have the same domain.")
        self._basis_functions = basis_functions
        self._num_per_level = numpy.array(num_per_level, dtype=int)
        self._end_level = numpy.cumsum(self._num_per_level)
        self._start_level = self._end_level - self._num_per_level


class MutableNestedBasisFunctionSet(NestedBasisFunctionSet):
    """Set of basis functions where number of functions is mutable.

    Parameters
    ----------
    num_terms : int
        the number of terms in the set.
    """

    def __init__(self, num_levels):
        self._num_levels = None
        self._basis_functions = None
        self._start_level = None
        self._end_level = None
        self._num_per_level = None

        self.num_levels = num_levels

    @property
    def num_levels(self):
        """int: number of levels."""
        return self._num_levels

    @num_levels.setter
    def num_levels(self, value):
        num_levels = int(value)
        if num_levels <= 0:
            raise ValueError("Must have at least one level.")
        if num_levels != self._num_levels:
            self._create_levels(num_levels)
            self._create()
            self._num_levels = num_levels

    @abc.abstractmethod
    def _create(self):
        """Create the basis functions in the set."""
        pass


class NestedClenshawCurtisBasisFunctionSet(MutableNestedBasisFunctionSet
):
    """Nested Clenshaw Curtis basis function set

    Parameters
    ----------
    num_levels : int
        The number of levels. Must be 1 or greater.

    Raises
    ------
    ValueError
        Must have at least one level.
    """

    def _create(self):
        """Create the basis functions in the set."""
        num_terms = self._end_level[-1]
        self._basis_functions = [ChebyshevFirstKind(i) for i in range(num_terms)]


class SlowNestedClenshawCurtisBasisFunctionSet(MutableNestedBasisFunctionSet
):
    """Nested Clenshaw Curtis basis function set using slow exponential growth.

    Parameters
    ----------
    num_levels : int
        The number of levels. Must be 1 or greater.

    Raises
    ------
    ValueError
        Must have at least one level.
    """

    def _create(self):
        """Create the basis functions in the set."""
        num_terms = self._end_level[-1]
        self._basis_functions = [ChebyshevFirstKind(i) for i in range(num_terms)]


class NestedTrigonometricBasisFunctionSet(MutableNestedBasisFunctionSet
):
    """Nested Trigonometric basis function set.

    Parameters
    ----------
    num_levels : int
        The number of levels. Must be 1 or greater.

    Raises
    ------
    ValueError
        Must have at least one level.
    """

    def _create(self):
        """Create the basis functions in the set."""
        num_terms = self._end_level[-1]
        index_trig = numpy.arange(num_terms, dtype=int)
        frequencies = numpy.where(
            index_trig % 2 == 1, (1 + index_trig) // 2, -index_trig // 2
        )
        self._basis_functions = [Trigonometric(f) for f in frequencies]
