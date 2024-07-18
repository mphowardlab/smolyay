import abc
import itertools
import warnings

import numpy
import sklearn
import sklearn.base
import sklearn.linear_model

import smolyay


class BaseSurrogate(
    sklearn.base.BaseEstimator,
    sklearn.base.MultiOutputMixin,
    sklearn.base.RegressorMixin,
):
    r"""Create a surrogate to approximate a complex function.

    Depending on the dimensionality (number of independent variables),
    sampling method, and basis functions, a surrogate model can be generated
    that approximates a set of data.
    ``domain`` is the domain of the function to be approximated.

    :attr:`points` stores the points that are used for sampling.
    :attr:`data` stores the output of the true function at the sampled points.
    :attr:`num_dimensions` is the number of dimensionns/independent variables.
    :meth:`train`, generates a trained surrogate model given a function and
    the points to sample at.
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
        self._data = None
        self._points = None
        self._valid_cache = False

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
            self._valid_cache = False

    @property
    def num_dimensions(self):
        """int: number of independent variables."""
        return self.domain.shape[0]

    @property
    def data(self):
        """list: data at sampling grid points."""
        if self._data is not None:
            return self._data.tolist()
        else:
            return None

    @property
    def points(self):
        """numpy.ndarray: points that are sampled"""
        return self._points

    def train(self, function, X):
        """Fit surrogate's components (basis functions) to the function.

        Parameters
        ----------
        function: callable
            Function to be approximated.
        """
        data = [function(x) for x in X]
        self.fit(X, data)

    @abc.abstractmethod
    def fit(self, X, y=None):
        """Fit surrogate's components (basis functions) to data.

        Parameters
        ----------
        X : list
            points that are sampled

        y : list
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
        ndarray of shape (n_samples,) or (n_samples, n_targets)
            Surrogate output at x.

        Raises
        ------
        RuntimeError
            For surrogate to be evaluated, function needs to be trained.
        ValueError
            Input must lie in domain of surrogate.
        """


class ProductSetSurrogate(BaseSurrogate):
    r"""Create a surrogate to approximate a complex function.

    Depending on the dimensionality (number of independent variables),
    sampling method, and combination of basis functions, a surrogate
    model can be generated that approximates a set of data.
    ``domain`` is the domain of the function to be approximated.
    ``basis_sets`` is a list of :class:BasisFunctionSets that describe
    the basis functions used to construct terms in the surrogate function,
    with the :class:BasisFunctionSet at index i in the list being the set of
    basis functions used for the ith dimensional variable.
    ``alpha`` is the lambda parameter used by Lasso and Ridge Regression
    ``regression`` is the method of regression used in the event that the
    number of terms does not match the number of sample points. Ridge
    regression and lasso regression are available, as well as finding
    the solution of least squares. If the basis functions return complex
    values, then solving for least squares will be used instead of ridge
    or lasso.

    :attr:`points` stores the points that are used for sampling.
    :attr:`data` stores the output of the true function at the sampled points.
    :attr:`index_combinations` describes the combination of basis function
    used to construct the terms of the surrogate, where each row is a term
    represented by a list of size ``num_dimensions`` that give the index of
    the basis functions from the :class:BasisFunctionSets in `basis_sets`
    that make up a given term.
    :attr:`coefficients` is the list of coefficients for each term in the
    surrogate equation. These cefficients are determined in :meth:`fit`.
    :meth:`train`, generates a trained surrogate model given a function and
    the points to sample at.
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

    alpha: float, default 1e-10
        the regression parameter used in Ridge and Lasso regression

    regression : ["ridge", "lasso", "lstsq"], default "ridge"
        the regression method if the number of points and terms don't match
    """

    def __init__(self, domain, basis_sets, alpha=1e-10, regression="ridge"):
        super().__init__(domain)
        self._basis_sets = None
        self._alpha = None
        self._regression = None
        self._index_combinations = None
        self._coefficients = None
        self._fit_gradient_flag = False
        self._integration_constant_flag = False

        self._basis_sets = basis_sets
        self.alpha = alpha
        self.regression = regression

    @property
    def alpha(self):
        """float: constant of the L2 term"""
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value < 0:
            raise ValueError("Alpha term must be positive")
        if self.alpha != value:
            self._alpha = value
            self._valid_cache = False

    @property
    def regression(self):
        """float: constant of the L2 and L1 term"""
        return self._regression

    @regression.setter
    def regression(self, value):
        regression = str(value).casefold()
        if not regression in ["ridge", "lasso", "lstsq"]:
            raise ValueError("Regression must be ridge or lasso")
        if self.regression != regression:
            self._regression = regression
            self._valid_cache = False

    @property
    def basis_sets(self):
        """list of BasisFunctionSet: the set of basis functions for the terms."""
        return self._basis_sets

    @property
    def index_combinations(self):
        """list of BasisFunctionSet: the set of basis functions for the terms."""
        if self._index_combinations is None:
            self._create_terms()
        return self._index_combinations
    
    @property
    def number_terms(self):
        """int: the number of terms in the surrogate model equation."""
        return self.index_combinations.shape[0]
    
    @property
    def coefficients(self):
        """numpy.ndarray: the coefficients of the terms"""
        return self._coefficients

    def predict(self, X, ignore_integration_warning=False):
        """Evaluate surrogate at a given input.

        Parameters
        ----------
        X: array-like with shape (n_samples, num_dimensions) or list of object
            Points at which the model is evaluated

        Returns
        -------
        ndarray of shape (n_samples,) or (n_samples, n_targets)
            Surrogate output at x.

        Raises
        ------
        RuntimeError
            For surrogate to be evaluated, function needs to be trained.
        ValueError
            Input must lie in domain of surrogate.
        NotImplementedError
            Predict after fitting to gradient not supported.
        """
        # validate inputs
        X = self._validate_data(X, ensure_2d=True, dtype="numeric", reset=False)
        if not self._valid_cache:
            raise RuntimeError("Model must be trained!")
        if self._fit_gradient_flag and not self._integration_constant_flag and not ignore_integration_warning:
           warnings.warn("Integration constant unavailable.")
        oob = any(
            numpy.any(X[:, i] < self.domain[i][0])
            or numpy.any(X[:, i] > self.domain[i][1])
            for i in range(self.num_dimensions)
        )
        if oob:
            raise ValueError("X must lie in domain of surrogate")

        # create lookup table
        num_basis_max = numpy.max([len(p) for p in self._basis_sets])
        if any(bs.is_complex for bs in self.basis_sets):
            lookup_table = numpy.zeros(
                (self.num_dimensions, num_basis_max, len(X)), dtype="complex_"
            )
        else:
            lookup_table = numpy.zeros(
                (self.num_dimensions, num_basis_max, len(X)), dtype="complex_"
            )
        # solve for the inputs at all the basis functions
        for dim in range(self.num_dimensions):
            for i, basis_fun in enumerate(self.basis_sets[dim]):
                new_X = basis_fun.domain[0] + (
                    basis_fun.domain[1] - basis_fun.domain[0]
                ) * (
                    (X[:, dim] - self.domain[dim, 0])
                    / (self.domain[dim, 1] - self.domain[dim, 0])
                )
                numpy.clip(new_X, basis_fun.domain[0], basis_fun.domain[1], out=new_X)
                lookup_table[dim, i, :] = basis_fun(new_X)

        # use lookup table to combine terms
        answer = numpy.zeros(len(X))
        for ic, coeff in zip(self.index_combinations, self.coefficients):
            answer = answer + numpy.real(
                coeff
                * numpy.prod(
                    [lookup_table[dim, ic[dim], :] for dim in range(len(ic))], axis=0
                )
            )

        # return results
        if len(X) == 1:
            return answer[0]
        else:
            return answer

    def predict_gradient(self, X):
        """Evaluate gradient of the surrogate at a given input.

        Parameters
        ----------
        X: array-like with shape (n_samples, num_dimensions) or list of object
            Points at which the model is evaluated

        Returns
        -------
        ndarray of shape (n_samples,) or (n_samples, n_targets)
            Surrogate output at x.

        Raises
        ------
        RuntimeError
            For surrogate to be evaluated, function needs to be trained.
        ValueError
            Input must lie in domain of surrogate.
        """
        # validate inputs
        X = self._validate_data(X, ensure_2d=True, dtype="numeric", reset=False)
        if not self._valid_cache:
            raise RuntimeError("Model must be trained!")
        oob = any(
            numpy.any(X[:, i] < self.domain[i][0])
            or numpy.any(X[:, i] > self.domain[i][1])
            for i in range(self.num_dimensions)
        )
        if oob:
            raise ValueError("X must lie in domain of surrogate")
        # create lookup table
        num_basis_max = numpy.max([len(p) for p in self._basis_sets])
        if any(bs.is_complex for bs in self.basis_sets):
            lookup_table = numpy.zeros(
                (self.num_dimensions, num_basis_max, len(X)), dtype="complex_"
            )
            lookup_table_derivative = numpy.zeros(
                (self.num_dimensions, num_basis_max, len(X)), dtype="complex_"
            )
        else:
            lookup_table = numpy.zeros((self.num_dimensions, num_basis_max, len(X)))
            lookup_table_derivative = numpy.zeros(
                (self.num_dimensions, num_basis_max, len(X))
            )
        # solve for the inputs at all the basis functions
        for dim in range(self.num_dimensions):
            for i, basis_fun in enumerate(self.basis_sets[dim]):
                new_X = basis_fun.domain[0] + (
                    basis_fun.domain[1] - basis_fun.domain[0]
                ) * (
                    (X[:, dim] - self.domain[dim, 0])
                    / (self.domain[dim, 1] - self.domain[dim, 0])
                )
                numpy.clip(new_X, basis_fun.domain[0], basis_fun.domain[1], out=new_X)
                lookup_table[dim, i, :] = basis_fun(new_X)
                lookup_table_derivative[dim, i, :] = (
                    basis_fun.derivative(new_X)
                    * (basis_fun.domain[1] - basis_fun.domain[0])
                    / (self.domain[dim, 1] - self.domain[dim, 0])
                )
        # use lookup table to combine terms
        answer = numpy.zeros((len(X), self.num_dimensions))
        for d in range(self.num_dimensions):
            for ic, coeff in zip(self.index_combinations, self.coefficients):
                answer[:, d] = answer[:, d] + numpy.real(
                    coeff
                    * numpy.prod(
                        [
                            (
                                lookup_table_derivative[dim, ic[dim], :]
                                if dim == d
                                else lookup_table[dim, ic[dim], :]
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

    def fit(self, X, y):
        """Fit surrogate's components (basis functions) to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
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
        if isinstance(
            X,
            (
                smolyay.samples.UnidimensionalPointSet,
                smolyay.samples.MultidimensionalPointSet,
            ),
        ):
            X = X.points
        X, y = self._validate_data(
            X,
            y,
            multi_output=True,
            y_numeric=True,
            ensure_2d=True,
            dtype="numeric",
        )
        self._points = X
        self._data = y
        oob = any(
            numpy.any(X[:, i] < self.domain[i][0])
            or numpy.any(X[:, i] > self.domain[i][1])
            for i in range(self.num_dimensions)
        )
        if oob:
            raise ValueError("X must lie in domain of surrogate")
        # create basis matrix
        num_basis_max = numpy.max([len(p) for p in self._basis_sets])
        if any(bs.is_complex for bs in self.basis_sets):
            lookup_table = numpy.zeros(
                (self.num_dimensions, num_basis_max, len(X)), dtype="complex_"
            )
            basis_matrix = numpy.zeros(
                (len(X), self.number_terms), dtype="complex_"
            )
        else:
            lookup_table = numpy.zeros((self.num_dimensions, num_basis_max, len(X)))
            basis_matrix = numpy.zeros((len(X), self.number_terms))
        # solve for the inputs at all the basis functions
        for dim in range(self.num_dimensions):
            for i, basis_fun in enumerate(self.basis_sets[dim]):
                new_X = basis_fun.domain[0] + (
                    basis_fun.domain[1] - basis_fun.domain[0]
                ) * (
                    (X[:, dim] - self.domain[dim, 0])
                    / (self.domain[dim, 1] - self.domain[dim, 0])
                )
                numpy.clip(new_X, basis_fun.domain[0], basis_fun.domain[1], out=new_X)
                lookup_table[dim, i, :] = basis_fun(new_X)

        # use lookup table to solve for each term
        for term, ic in enumerate(self.index_combinations):
            basis_matrix[:, term] = numpy.prod(
                [lookup_table[dim, ic[dim], :] for dim in range(len(ic))], axis=0
            )
        # solve for coefficients
        if basis_matrix.shape[0] == basis_matrix.shape[1]:
            self._coefficients = numpy.linalg.solve(basis_matrix, self._data)
        else:
            if numpy.any(numpy.iscomplex(basis_matrix)) or self.regression == "lstsq":

                self._coefficients = numpy.linalg.lstsq(
                    basis_matrix, self._data, rcond=None
                )[0]
            elif self.regression == "ridge":
                basis_matrix = numpy.real(basis_matrix)
                regressor = sklearn.linear_model.Ridge(
                    alpha=self.alpha, fit_intercept=False
                )
                self._coefficients = numpy.squeeze(regressor.fit(basis_matrix, y).coef_)
            else:
                basis_matrix = numpy.real(basis_matrix)
                regressor = sklearn.linear_model.Lasso(
                    alpha=self.alpha, fit_intercept=False
                )
                self._coefficients = numpy.squeeze(regressor.fit(basis_matrix, y).coef_)
        self._valid_cache = True
        self._fit_gradient_flag = False
        return self

    def fit_gradient(self, X, y, constant_x=None, constant_y=None):
        """Fit surrogate's components (basis functions) to gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            points that are sampled

        y : array-like of shape (n_samples, n_features)
            gradient function at grid points.

        constant_x : array-like of shape (n_features)
            a point where the function has a specified value to solve
            the integration constant

        constant_y : numeric
            the value at constant_x

        Returns
        -------
        self : object
            BaseSurrogate class instance

        Raises
        ------
        IndexError
            y must be 2D array with shape (n_samples, n_features).
        ValueError
            Input must lie in domain of surrogate.
        """
        # validate inputs
        if isinstance(
            X,
            (
                smolyay.samples.UnidimensionalPointSet,
                smolyay.samples.MultidimensionalPointSet,
            ),
        ):
            X = X.points
        X, y = self._validate_data(
            X,
            y,
            multi_output=True,
            y_numeric=True,
            ensure_2d=True,
            dtype="numeric",
        )

        if y.shape != X.shape:
            raise IndexError("y must be 2D array with shape (n_samples, n_features).")
        oob = any(
            numpy.any(X[:, i] < self.domain[i][0])
            or numpy.any(X[:, i] > self.domain[i][1])
            for i in range(self.num_dimensions)
        )
        if oob:
            raise ValueError("X must lie in domain of surrogate")
        # add inputs to training attributes for safekeeping
        self._points = X
        self._data = y

        ## Create basis matrix
        num_basis_max = numpy.max([len(p) for p in self._basis_sets])
        if any(bs.is_complex for bs in self.basis_sets):
            lookup_table = numpy.zeros(
                (self.num_dimensions, num_basis_max, len(X)), dtype="complex_"
            )
            lookup_table_derivative = numpy.zeros(
                (self.num_dimensions, num_basis_max, len(X)), dtype="complex_"
            )
            basis_matrix = numpy.zeros(
                (len(X) * self.num_dimensions, self.number_terms),
                dtype="complex_",
            )
        else:
            lookup_table = numpy.zeros((self.num_dimensions, num_basis_max, len(X)))
            lookup_table_derivative = numpy.zeros(
                (self.num_dimensions, num_basis_max, len(X))
            )
            basis_matrix = numpy.zeros(
                (len(X) * self.num_dimensions, self.number_terms),
            )
        # solve for the inputs at all the basis functions
        for dim in range(self.num_dimensions):
            for i, basis_fun in enumerate(self.basis_sets[dim]):
                new_X = basis_fun.domain[0] + (
                    basis_fun.domain[1] - basis_fun.domain[0]
                ) * (
                    (X[:, dim] - self.domain[dim, 0])
                    / (self.domain[dim, 1] - self.domain[dim, 0])
                )
                numpy.clip(new_X, basis_fun.domain[0], basis_fun.domain[1], out=new_X)
                lookup_table[dim, i, :] = basis_fun(new_X)
                lookup_table_derivative[dim, i, :] = (
                    basis_fun.derivative(new_X)
                    * (basis_fun.domain[1] - basis_fun.domain[0])
                    / (self.domain[dim, 1] - self.domain[dim, 0])
                )

        # use lookup table to solve for each term
        for d in range(self.num_dimensions):
            for term, ic in enumerate(self.index_combinations):
                if self.num_dimensions > 1:
                    basis_matrix[d :: self.num_dimensions, term] = numpy.prod(
                        [
                            (
                                lookup_table_derivative[dim, ic[dim], :]
                                if dim == d
                                else lookup_table[dim, ic[dim], :]
                            )
                            for dim in range(len(ic))
                        ],
                        axis=0,
                    )
                else:
                    basis_matrix[d :: self.num_dimensions, term] = (
                        lookup_table_derivative[d, ic[d], :]
                    )

        data = numpy.reshape(self._data, (self.num_dimensions * len(X),))

        # solve for coefficients
        if (
            basis_matrix.shape[0] / self.num_dimensions == basis_matrix.shape[1]
            or numpy.any(numpy.iscomplex(basis_matrix))
            or self.regression == "lstsq"
        ):
            self._coefficients = numpy.linalg.lstsq(basis_matrix, data, rcond=None)[0]
        else:
            if self.regression == "ridge":
                basis_matrix = numpy.real(basis_matrix)
                regressor = sklearn.linear_model.Ridge(
                    alpha=self.alpha, fit_intercept=False
                )
            else:
                basis_matrix = numpy.real(basis_matrix)
                regressor = sklearn.linear_model.Lasso(
                    alpha=self.alpha, fit_intercept=False
                )
            self._coefficients = numpy.squeeze(regressor.fit(basis_matrix, data).coef_)
        self._valid_cache = True
        self._fit_gradient_flag = True
        if not constant_x is None and not constant_y is None:
            constant_x, constant_y = self._validate_data(
                constant_x,
                constant_y,
                multi_output=False,
                y_numeric=True,
            )
            predicted_y = self.predict(constant_x, ignore_integration_warning=True)
            integration_constant = constant_y - predicted_y
            self._coefficients[0] = integration_constant
            self._integration_constant_flag = True
        else:
            self._integration_constant_flag = False
        return self

    @abc.abstractmethod
    def _create_terms(self):
        """create terms for the surrogate"""
        pass


class TensorProductSurrogate(ProductSetSurrogate):
    """A surrogate that uses a combination of tensor product as terms

    Parameters
    ----------
    domain: numpy.ndarray
        the domain of the surrgoate

    basis_set: BasisFunctionSet or list of BasisFunctionSet
        the set of basis functions used to combine terms

    alpha: float, default 1e-10
        the regression parameter used in Ridge regression

    regression : ["ridge","lasso"], default "ridge"
        the regression method if the number of points and terms don't match
    """

    def _create_terms(self):
        num_points = numpy.prod([len(p) for p in self._basis_sets])

        self._index_combinations = numpy.zeros(
            (num_points, len(self._basis_sets)), dtype=int
        )
        for i, point in enumerate(
            itertools.product(*[numpy.arange(len(bs)) for bs in self._basis_sets])
        ):
            self._index_combinations[i] = point


class SmolyakSparseProductSurrogate(ProductSetSurrogate):
    """A surrogate from sparse combinations of terms

    Parameters
    ----------
    domain: numpy.ndarray
        the domain of the surrgoate

    basis_set: BasisFunctionSet or list of BasisFunctionSet
        the set of basis functions used to combine terms

    alpha: float, default 1e-10
        the regression parameter used in Ridge regression

    regression : ["ridge","lasso"], default "ridge"
        the regression method if the number of points and terms don't match
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
                    smolyay.samples._generate_compositions(
                        sum_of_levels, self.num_dimensions, include_zero=True
                    )
                )
            )
        level_combinations = numpy.array(level_combinations)

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
