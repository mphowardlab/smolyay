import abc
import itertools
import warnings

import numpy
import sklearn.linear_model

from . import samples


class Surrogate:
    r"""Create a surrogate to approximate a complex function.

    Depending on the dimensionality (number of independent variables),
    sampling method, and basis functions, a surrogate model can be generated
    that approximates a set of data.
    ``domain`` is the domain of the function to be approximated.

    :attr:`num_dimensions` is the number of dimensionns/independent variables.
    :meth:`fit` computes the coefficients based on a set of data
    (function values at transformed grid points).
    Once the surrogate is constructed, one can evaluate the surrogate
    through :meth:`predict(x)`.

    Parameters
    ----------
    domain: numpy.ndarray
        the domain of the surrgoate
    """

    def __init__(self, domain):
        self._domain = None
        self._needs_fit = False

        self.domain = domain

    @property
    def domain(self):
        """domain: numpy.ndarray the domain of the surrogate."""
        return self._domain

    @domain.setter
    def domain(self, value):
        domain = numpy.sort(numpy.array(value, ndmin=2), axis=1)
        if not numpy.array_equal(self._domain, domain):
            self._domain = domain
            self._needs_fit = False

    @property
    def num_dimensions(self):
        """int: number of independent variables."""
        return self.domain.shape[0]

    def _assert_in_domain(self, X):
        """Check if input is in surrogate's domain.

        Parameters
        ----------
        X : numpy.ndarray with shape (n_samples, num_dimensions)
            input
        """
        if X.shape[1] != self.num_dimensions:
            raise IndexError("Must be 2D array with shape (n_samples, num_dimensions).")

        oob = any(
            numpy.any(X[:, i] < self.domain[i][0])
            or numpy.any(X[:, i] > self.domain[i][1])
            for i in range(self.num_dimensions)
        )
        if oob:
            raise ValueError("X must lie in domain of surrogate.")

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit surrogate's components (basis functions) to data.

        Parameters
        ----------
        X : array-like, UnidimensionalPointSet, or MultidimensionalPointSet of shape (n_samples, num_dimensions)
            points that are sampled

        y : array-like of shape (n_samples,)
            function at grid points.

        Returns
        -------
        self : object
            BaseSurrogate class instance

        Raises
        ------
        ValueError
            Input must lie in domain of surrogate.
        """

    @abc.abstractmethod
    def predict(self, X):
        """Evaluate surrogate at a given input.

        Parameters
        ----------
        X: array-like with shape (n_samples, num_dimensions) or list of object
            Points at which the model is evaluated

        Returns
        -------
        ndarray of shape (n_samples,) or (n_samples, num_dimensions)
            Surrogate output at x.

        Raises
        ------
        RuntimeError
            For surrogate to be evaluated, function needs to be fit.
        ValueError
            Input must lie in domain of surrogate.
        """


class SetProductSurrogate(Surrogate):
    r"""Create a surrogate to approximate a complex function.

    Depending on the dimensionality (number of independent variables),
    sampling method, and combination of basis functions, a surrogate
    model can be generated that approximates a set of data.
    ``domain`` is the domain of the function to be approximated.
    ``basis_sets`` is a list of :class:BasisFunctionSets that describe
    the basis functions used to construct terms in the surrogate function,
    with the :class:BasisFunctionSet at index i in the list being the set of
    basis functions used for the ith dimensional variable.
    ``regularization`` is the method of regularization used in the event that the
    number of terms does not match the number of sample points. Ridge
    regularization and lasso regularization are available, as well as finding
    the solution of least squares. If the basis functions return complex
    values, then solving for least squares will be used instead of ridge
    or lasso.

    :attr:`index_combinations` describes the combination of basis function
    used to construct the terms of the surrogate, where each row is a term
    represented by a list of size ``num_dimensions`` that give the index of
    the basis functions from the :class:BasisFunctionSets in `basis_sets`
    that make up a given term.
    :attr:`coefficients` is the list of coefficients for each term in the
    surrogate equation. These cefficients are determined in :meth:`fit`.
    :meth:`fit` computes the coefficients based on a set of data
    (function values at transformed grid points).
    Once the surrogate is constructed, one can evaluate the surrogate
    through :meth:`predict(x)` and its gradient can be evaluated
    through :meth:`gradient(x)`.
    :meth:`_create_terms`, is an abstract method for generating the terms
    described in :attr:`index_combinations`.

    Parameters
    ----------
    domain: numpy.ndarray
        the domain of the surrgoate

    basis_sets: list of :class:BasisFunctionSet
        the set of basis functions used to combine terms

    regularization : [:class:L1Regularization, :class:L2Regularization, None], default None
        the regularization method for determining the coefficients
    """

    def __init__(self, domain, basis_sets, regularization=None):
        super().__init__(domain)
        self._basis_sets = list(basis_sets)
        self._regularization = None
        self._index_combinations = None
        self._coefficients = None
        self._integration_constant = 0
        self._needs_terms_constructed = True

        self.regularization = regularization

    @property
    def regularization(self):
        """:class:RegularizationMethod: constant of the L2 and L1 term"""
        return self._regularization

    @regularization.setter
    def regularization(self, value):
        if value is not None and not isinstance(value, RegularizationMethod):
            raise ValueError("Regression must be a RegularizationMethod")
        if self.regularization != value:
            self._regularization = value
            self._needs_fit = False

    @property
    def basis_sets(self):
        """list of BasisFunctionSet: the set of basis functions for the terms."""
        return self._basis_sets

    def predict(self, X):
        """Evaluate surrogate at a given input.

        Parameters
        ----------
        X: array-like with shape (n_samples, num_dimensions) or list of object
            Points at which the model is evaluated.

        Returns
        -------
        ndarray of shape (n_samples,) or (n_samples, num_dimensions)
            Surrogate output at x.

        Raises
        ------
        RuntimeError
            For surrogate to be evaluated, function needs to be trained.
        IndexError
            Input must be 2D array with shape (n_samples, num_dimensions).
        ValueError
            Input must lie in domain of surrogate.

        """
        # validate inputs
        if not self._needs_fit:
            raise RuntimeError("Model must be fit!")
        X = numpy.array(X, ndmin=2, copy=None)
        self._assert_in_domain(X)

        # create lookup table and solve for all the basis functions
        lookup_table = [
            self.basis_sets[dim](X[:, dim], self.domain[dim])
            for dim in range(self.num_dimensions)
        ]

        # use lookup table to combine terms
        answer = numpy.ones(len(X)) * self._integration_constant
        for ic, coeff in zip(self._index_combinations, self._coefficients):
            answer += numpy.real(
                coeff
                * numpy.prod(
                    [lookup_table[dim][ic[dim], :] for dim in range(len(ic))], axis=0
                )
            )

        # return results
        if len(X) == 1:
            return answer[0]
        else:
            return answer

    def predict_gradient(self, X):
        """Evaluate gradient or Jacobian of the surrogate at a given input.

        Parameters
        ----------
        X: array-like with shape (n_samples, num_dimensions) or list of object
            Points at which the model is evaluated

        Returns
        -------
        ndarray of shape (n_samples,) or (n_samples, num_dimensions)
            Gradient output at x.

        Raises
        ------
        RuntimeError
            For surrogate to be evaluated, function needs to be fit.
        IndexError
            Input must be 2D array with shape (n_samples, num_dimensions).
        ValueError
            Input must lie in domain of surrogate.
        """
        # validate inputs
        if not self._needs_fit:
            raise RuntimeError("Model must be fit!")
        X = numpy.array(X, ndmin=2, copy=None)
        self._assert_in_domain(X)

        # create lookup table and solve for all the basis functions
        lookup_table = []
        lookup_table_derivative = []
        for dim in range(self.num_dimensions):
            lookup_table.append(self.basis_sets[dim](X[:, dim], self.domain[dim]))
            lookup_table_derivative.append(
                self.basis_sets[dim].derivative(X[:, dim], domain=self.domain[dim])
            )

        # use lookup table to combine terms
        answer = numpy.zeros((len(X), self.num_dimensions))
        for d in range(self.num_dimensions):
            for ic, coeff in zip(self._index_combinations, self._coefficients):
                answer[:, d] = answer[:, d] + numpy.real(
                    coeff
                    * numpy.prod(
                        [
                            (
                                lookup_table_derivative[dim][ic[dim], :]
                                if dim == d
                                else lookup_table[dim][ic[dim], :]
                            )
                            for dim in range(len(ic))
                        ],
                        axis=0,
                    )
                )

        # return results
        answer.reshape(X.shape)
        if all(x == 1 for x in X.shape):
            return answer.item(0)
        else:
            return answer

    def predict_hessian(self, X):
        """Evaluate the Hessian matrix, or 2nd order derivatives, of the surrogate.

        Parameters
        ----------
        X: array-like with shape (n_samples, num_dimensions) or list of object
            Points at which the model is evaluated

        Returns
        -------
        ndarray of shape (n_samples,) or (n_samples, num_dimensions, num_dimensions)
            Surrogate output at x.

        Raises
        ------
        RuntimeError
            For surrogate to be evaluated, function needs to be fit.
        IndexError
            Input must be 2D array with shape (n_samples, num_dimensions).
        ValueError
            Input must lie in domain of surrogate.
        """
        # validate inputs
        if not self._needs_fit:
            raise RuntimeError("Model must be fit!")
        X = numpy.array(X, ndmin=2, copy=None)
        self._assert_in_domain(X)

        # create lookup table and solve for all the basis functions
        lookup_table = []
        lookup_table_derivative = []
        lookup_table_2nd_derivative = []
        for dim in range(self.num_dimensions):
            lookup_table.append(self.basis_sets[dim](X[:, dim], self.domain[dim]))
            lookup_table_derivative.append(
                self.basis_sets[dim].derivative(X[:, dim], domain=self.domain[dim])
            )
            lookup_table_2nd_derivative.append(
                self.basis_sets[dim].derivative(X[:, dim], n=2, domain=self.domain[dim])
            )

        # use lookup table to combine terms
        answer = numpy.zeros((len(X), self.num_dimensions, self.num_dimensions))
        for dx in range(self.num_dimensions):
            for dy in range(dx, self.num_dimensions):
                for ic, coeff in zip(self._index_combinations, self._coefficients):
                    answer[:, dx, dy] = answer[:, dy, dx] = answer[
                        :, dx, dy
                    ] + numpy.real(
                        coeff
                        * numpy.prod(
                            [
                                (
                                    lookup_table_2nd_derivative[dim][ic[dim], :]
                                    if dim == dx and dim == dy
                                    else (
                                        lookup_table_derivative[dim][ic[dim], :]
                                        if dim == dx or dim == dy
                                        else lookup_table[dim][ic[dim], :]
                                    )
                                )
                                for dim in range(len(ic))
                            ],
                            axis=0,
                        )
                    )

        # return results
        answer.reshape(list(X.shape) + [self.num_dimensions])
        if all(x == 1 for x in X.shape):
            return answer.item(0)
        else:
            return answer

    def fit(self, X, y):
        """Fit surrogate's components (basis functions) to data.

        Parameters
        ----------
        X : array-like, UnidimensionalPointSet, or MultidimensionalPointSet of shape (n_samples, num_dimensions)
            points that are sampled

        y : array-like of shape (n_samples,)
            function at grid points.

        Returns
        -------
        self : object
            BaseSurrogate class instance

        Raises
        ------
        IndexError
            X must be 2D array with shape (n_samples, num_dimensions).
        IndexError
            y must be 1D array with shape (n_samples,).
        ValueError
            X must lie in domain of surrogate.
        """
        # reset constant
        self._integration_constant = 0
        if self._needs_terms_constructed:
            self._create_terms()
            self._needs_terms_constructed = False

        # get points
        if isinstance(
            X,
            (
                samples.UnidimensionalPointSet,
                samples.MultidimensionalPointSet,
            ),
        ):
            X = X.points

        # validate data inputs
        X = numpy.array(X, ndmin=2, copy=None)
        self._assert_in_domain(X)

        y = numpy.asarray(y)
        if y.shape != (X.shape[0],) and y.shape != (X.shape[0], 1):
            raise IndexError("Must be 1D array with shape (n_samples,).")

        # create lookup table and solve for all the basis functions
        lookup_table = [
            self.basis_sets[dim](X[:, dim], self.domain[dim])
            for dim in range(self.num_dimensions)
        ]

        # create basis matrix
        basis_matrix = numpy.zeros(
            (len(X), len(self._index_combinations)),
            dtype=(
                complex
                if any(
                    any(bf._is_complex for bf in basis_set)
                    for basis_set in self.basis_sets
                )
                else float
            ),
        )

        # use lookup table to solve for each term
        for term, ic in enumerate(self._index_combinations):
            basis_matrix[:, term] = numpy.prod(
                [lookup_table[dim][ic[dim], :] for dim in range(len(ic))], axis=0
            )

        # solve for coefficients
        if self.regularization is None or numpy.iscomplexobj(basis_matrix):
            if basis_matrix.shape[0] == basis_matrix.shape[1]:
                try:
                    self._coefficients = numpy.linalg.solve(basis_matrix, y)
                except numpy.linalg.LinAlgError:
                    self._coefficients = numpy.linalg.lstsq(
                        basis_matrix, y, rcond=None
                    )[0]
            else:
                self._coefficients = numpy.linalg.lstsq(basis_matrix, y, rcond=None)[0]
        elif isinstance(self.regularization, L2Regularization):
            regressor = regressor = sklearn.linear_model.Ridge(alpha=self.regularization.alpha, fit_intercept=False)
            self._coefficients = numpy.squeeze(regressor.fit(basis_matrix, y).coef_)
        elif isinstance(self.regularization, L1Regularization):
            regressor = regressor = sklearn.linear_model.Lasso(alpha=self.regularization.alpha, fit_intercept=False)
            self._coefficients = numpy.squeeze(regressor.fit(basis_matrix, y).coef_)
        else:
            self._coefficients = numpy.linalg.lstsq(basis_matrix, y, rcond=None)[0]
        self._needs_fit = True
        return self

    def fit_gradient(self, X, y, X0=None, y0=None):
        """Fit surrogate's components (basis functions) to gradient.

        Parameters
        ----------
        X : array-like, UnidimensionalPointSet, or MultidimensionalPointSet of shape (n_samples, num_dimensions)
            points that are sampled

        y : array-like of shape (n_samples, num_dimensions)
            gradient function at grid points.

        X0 : array-like of shape (num_dimensions) or None, optional
            a point where the function has a specified value to solve
            the integration constant. If None, assumed to be the lower domain.

        y0 : numeric or None, optional
            the value at X0

        Returns
        -------
        self : object
            BaseSurrogate class instance

        Raises
        ------
        IndexError
            X must be 2D array with shape (n_samples, num_dimensions).
        IndexError
            y must be 1D array with shape (n_samples,).
        ValueError
            X must lie in domain of surrogate.
        IndexError
            X0 must be 1D array with shape (n_samples,) if not None.
        """
        # reset constant
        self._integration_constant = 0
        if self._needs_terms_constructed:
            self._create_terms()
            self._needs_terms_constructed = False

        # validate inputs
        if isinstance(
            X,
            (
                samples.UnidimensionalPointSet,
                samples.MultidimensionalPointSet,
            ),
        ):
            X = X.points

        # validate data inputs
        X = numpy.array(X, ndmin=2, copy=None)
        self._assert_in_domain(X)
        y = numpy.array(y, ndmin=2, copy=None)
        if y.shape != X.shape:
            raise IndexError(
                "y must be 2D array with shape (n_samples, num_dimensions)."
            )

        # create lookup table and solve for all the basis functions
        lookup_table = []
        lookup_table_derivative = []
        for dim in range(self.num_dimensions):
            lookup_table.append(self.basis_sets[dim](X[:, dim], self.domain[dim]))
            lookup_table_derivative.append(
                self.basis_sets[dim].derivative(X[:, dim], domain=self.domain[dim])
            )

        # create basis matrix
        basis_matrix = numpy.zeros(
            (len(X) * self.num_dimensions, len(self._index_combinations)),
            dtype=(
                complex
                if any(
                    any(bf._is_complex for bf in basis_set)
                    for basis_set in self.basis_sets
                )
                else float
            ),
        )

        # use lookup table to solve for each term
        for d in range(self.num_dimensions):
            for term, ic in enumerate(self._index_combinations):
                if self.num_dimensions > 1:
                    basis_matrix[d :: self.num_dimensions, term] = numpy.prod(
                        [
                            (
                                lookup_table_derivative[dim][ic[dim], :]
                                if dim == d
                                else lookup_table[dim][ic[dim], :]
                            )
                            for dim in range(len(ic))
                        ],
                        axis=0,
                    )
                else:
                    basis_matrix[d :: self.num_dimensions, term] = (
                        lookup_table_derivative[d][ic[d], :]
                    )

        data = numpy.reshape(y, (self.num_dimensions * len(X),))

        # solve for coefficients
        if self.regularization is None or numpy.iscomplexobj(basis_matrix):
            self._coefficients = numpy.linalg.lstsq(basis_matrix, data, rcond=None)[0]
        elif isinstance(self.regularization, L2Regularization):
            regressor = sklearn.linear_model.Ridge(alpha=self.regularization.alpha, fit_intercept=False)
            self._coefficients = numpy.squeeze(regressor.fit(basis_matrix, data).coef_)
        elif isinstance(self.regularization, L1Regularization):
            regressor = sklearn.linear_model.Lasso(alpha=self.regularization.alpha, fit_intercept=False)
            self._coefficients = numpy.squeeze(regressor.fit(basis_matrix, data).coef_)
        else:
            self._coefficients = numpy.linalg.lstsq(basis_matrix, data, rcond=None)[0]
        self._needs_fit = True
        self._fit_gradient_flag = True

        # determine integration constant if possible
        if not y0 is None:
            # validate data inputs
            if X0 is None:
                X0 = self.domain[:, 0].reshape((1, -1))
            else:
                X0 = numpy.array(X0, ndmin=2)
            if X0.shape != (1, self.num_dimensions):
                raise IndexError(
                    "Must be 2D array with shape (1, num_dimensions) if not None."
                )
            y0 = numpy.array(y0).item(0)
            predicted_y = self.predict(X0)
            integration_constant = y0 - predicted_y
            self._integration_constant = integration_constant
        return self

    @abc.abstractmethod
    def _create_terms(self):
        """create terms for the surrogate"""
        pass


class TensorProductSurrogate(SetProductSurrogate):
    """A surrogate that uses a combination of tensor product as terms

    Parameters
    ----------
    domain: numpy.ndarray
        the domain of the surrgoate

    basis_set: BasisFunctionSet or list of BasisFunctionSet
        the set of basis functions used to combine terms

    regularization : [:class:L1Regularization, :class:L2Regularization, None], default None
        the regularization method for determining the coefficients
    """

    def _create_terms(self):
        num_points = numpy.prod([len(p) for p in self._basis_sets])

        self._index_combinations = numpy.zeros(
            (num_points, len(self._basis_sets)), dtype=int
        )
        for i, point in enumerate(
            itertools.product(*[range(len(bs)) for bs in self._basis_sets])
        ):
            self._index_combinations[i] = point


class SmolyakSparseProductSurrogate(SetProductSurrogate):
    """A surrogate from sparse combinations of terms

    Parameters
    ----------
    domain: numpy.ndarray
        the domain of the surrgoate

    basis_set: BasisFunctionSet or list of BasisFunctionSet
        the set of basis functions used to combine terms

    regularization : [:class:L1Regularization, :class:L2Regularization, None], default None
        the regularization method for determining the coefficients
    """

    def _create_terms(self):
        # find number of levels per dimension
        num_levels_per_dim = [ob.num_levels for ob in self._basis_sets]
        max_num_levels = max(num_levels_per_dim)

        # get the combinations of levels based on maximum possible number of levels
        level_combinations = []
        for sum_of_levels in range(max_num_levels):
            level_combinations.extend(
                list(
                    samples._generate_compositions(
                        sum_of_levels, self.num_dimensions, include_zero=True
                    )
                )
            )
        level_combinations = numpy.array(level_combinations, dtype=int)

        # remove combinations where a dimension exceeds its number of levels
        # only check if basis sets have different numbers of levels
        if min(num_levels_per_dim) != max_num_levels:
            valid_comb = numpy.all(
                numpy.less(level_combinations, num_levels_per_dim), axis=1
            )
            level_combinations = level_combinations[valid_comb]

        # generate sets of indexes based on combinations of levels
        for level_comb in level_combinations:
            level_point_combinations = [
                numpy.arange(
                    self._basis_sets[d].start_level[level],
                    self._basis_sets[d].end_level[level],
                )
                for d, level in enumerate(level_comb)
            ]
            num_terms = numpy.prod([len(p) for p in level_point_combinations])
            index_combinations_ = numpy.zeros(
                (num_terms, self.num_dimensions), dtype=int
            )
            for i, point in enumerate(itertools.product(*level_point_combinations)):
                index_combinations_[i] = point
            # add newly generated term indexes to set
            if self._index_combinations is None:
                self._index_combinations = index_combinations_
            else:
                self._index_combinations = numpy.concatenate(
                    (self._index_combinations, index_combinations_), axis=0
                )


class RegularizationMethod:
    pass


class L2Regularization(RegularizationMethod):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha


class L1Regularization(RegularizationMethod):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
