import abc
import math

import numpy


class Normalizer(abc.ABC):
    r"""A transformation on the training data of a surrogate

    Prior to training a surrogate model, a transformation can be applied
    to the data to put it on a different scale or otherwise normalize it.
    Transformed data can potentially result in a better behaved surrogate,
    depending on the normalization and the surrogate. The output of the
    surrogate trained on normalized data must be unnormalized to match the
    real function.

    This class specifies a type of normalization, defined by a
    transform function :meth:`transform` and an inverse transform
    :meth:`inverse_transform`. The names of the methods in this class are
    chosen to maintain compatibility with scripts that use scalars in the
    sklearn.preprocessing package to transform and normalize data.

    :meth:`transform` is an abstract method to be defined by child class that
    normalizes the input.
    :meth:`inverse_transform` is an abstract method to be defined by child
    class that is expected to perform an inverse operation to :meth:`transform`
    that unnormalizes the input.

    :meth:`check_normalize` checks if :meth:`inverse_transform` is the inverse
    of :meth:`transform`. If defined correctly, applying :meth:`transform`
    and :meth:`inverse_transform` sequentially should return the initial input,
    and the method will return True will be returned if the final output of the
    sequential operation is sufficiently close to the initial input.

    :meth:`fit` is expected to be overridden by child classes that use the
    training data to calculate parameters that need fitting.

    :meth:`fit_tranform` fits the Normalizer using the data, and then performs
    the transform on the data.

    """

    def __init__(self):
        self._valid_cache = False

    def fit(self, x):
        """Fit the Normalizer

        To be overridden should a child class require the training data
        for calculations.

        Parameters
        ----------
        x : numerical data
            the training data

        Returns
        -------
        Normalizer
            the normalizer
        """
        self._valid_cache = True
        return self

    @abc.abstractmethod
    def transform(self, x):
        """Normalization function

        Parameters
        ----------
        x : numerical data
            data to be transformed

        Return
        ------
        normalized data
        """
        pass

    def fit_transform(self, x):
        """Fit the data and then normalize it

        Parameters
        ----------
        x : numerical data
            data to be transformed

        Return
        ------
        normalized data
        """
        return self.fit(x).transform(x)

    @abc.abstractmethod
    def inverse_transform(self, x):
        """Inverse normalization function

        Parameters
        ----------
        x : numerical data
            normalized data to be transformed

        Return
        ------
        unnormalized data
        """
        pass

    @abc.abstractmethod
    def derivative(self, x, n=1):
        """The derivative of the transformation.

        Parameters
        ----------
        x : array-like
            the input data

        n : int, optional
            order of derivative. Default is 1.

        Returns
        -------
        array-like
            derivative at x
        """
        pass

    @abc.abstractmethod
    def inverse_derivative(self, x, n=1):
        """The derivative of the inverse transformation.

        Parameters
        ----------
        x : array-like
            the input data

        n : int, optional
            order of derivative. Default is 1.

        Returns
        -------
        array-like
            derivative at x
        """
        pass

    def check_normalize(self, x):
        """Check error from normalizing process
        If defined correctly, performing :meth:`inverse_transform` on
        the output of :meth:`transform` should return the input of
        :meth:`transform`. As theory does not always align with practice,
        this method checks if the data changes from its initial
        value after the transform and inverse transform are done in
        sequence. If the data doesn't change, it returns True. If it
        does change, it returns False.

        Parameters
        ----------
        x : numerical data
            data to compare before and after transformation

        Returns
        --------
        error : float
            statistic to represent how data changes from initial value
        """
        x = numpy.array(x)
        new_x = self.inverse_transform(self.transform(x))
        return numpy.allclose(x, new_x)


class IntervalNormalizer(Normalizer):
    """Scales data onto the interval [0,1]

    Using the min and max of the original training data, the training
    data is normalized to the range [0,1].

    The tranformation equation is
    ..math::
        y = (x - min_val/(max_val - min_val)
        where :math:min_val is the minimum of the training data and
        :math:max_data is the maximum of the training data

    Should any subsequent data fall outside of the range established
    by the original training data, that data will outside the range [0,1]

    The inverse transformation equation is
    ..math::
        y = x * (max_val - min_val) + min_val

    In the case where the original training data contains one variable,
    and thus the min and the max are the same, no normalization is
    applied.

    The properties ``min_val`` and ``max_val`` store the min and max of
    the original training data.
    """

    def __init__(self):
        super().__init__()
        self._min_val = None
        self._max_val = None

    @property
    def min_val(self):
        """float: min of original training data"""
        return self._min_val

    @property
    def max_val(self):
        """float: max of original training data"""
        return self._max_val

    def fit(self, x):
        """Calculates the min and max

        Parameters
        ----------
        x : numerical data
            the training data

        Returns
        -------
        Normalizer
            the normalizer
        """
        self._max_val = numpy.max(x)
        self._min_val = numpy.min(x)
        self._valid_cache = True
        return self

    def transform(self, x):
        """Normalize the data using the min and max

        Using the min and max of the training data, the input is
        scaled such that the max of the original training data is 1 and
        the min of the original traning data is 0.

        Parameters
        ----------
        x : numerical data
            data to be transformed

        Return
        ------
        numpy:ndarray
            the transformed data

        Raises
        ------
        ValueError
            normalizer was never fit.
        """
        if not self._valid_cache:
            raise ValueError("Normalizer needs fitting!")
        x = numpy.array(x)
        if self.min_val >= self.max_val:
            return x
        else:
            return (x - self.min_val) / (self.max_val - self.min_val)

    def inverse_transform(self, x):
        """Inverse normalization function

        Using the min and max of the training data, the input is
        scaled such that 1 becomes the max of the original training data
        and 0 becomes the min of the original training data

        Parameters
        ----------
        x : numerical data
            normalized data to be transformed

        Return
        ------
        numpy:ndarray
            the untransformed data

        Raises
        ------
        ValueError
            normalizer was never fit.
        """
        if not self._valid_cache:
            raise ValueError("Normalizer needs fitting!")
        x = numpy.array(x)
        if self.min_val >= self.max_val:
            return x
        else:
            return x * (self.max_val - self.min_val) + self.min_val

    def derivative(self, x, n=1):
        """The derivative of the transformation.

        Parameters
        ----------
        x : array-like
            the input data

        n : int, optional
            order of derivative. Default is 1.

        Returns
        -------
        array-like
            derivative at x

        Raises
        ------
        ValueError
            normalizer was never fit.
        """
        if not self._valid_cache:
            raise ValueError("Normalizer needs fitting!")
        if n == 1:
            return 1 / (self.max_val - self.min_val) * numpy.ones(numpy.shape(x))
        elif n > 1:
            return numpy.zeros(numpy.shape(x))
        else:
            raise NotImplementedError("Derivative order " + str(n) + " not supported.")

    def inverse_derivative(self, x, n=1):
        """The derivative of the inverse transformation.

        Parameters
        ----------
        x : array-like
            the input data

        n : int, optional
            order of derivative. Default is 1.

        Returns
        -------
        array-like
            derivative at x

        Raises
        ------
        ValueError
            normalizer was never fit.
        """
        if not self._valid_cache:
            raise ValueError("Normalizer needs fitting!")
        if n == 1:
            return (self.max_val - self.min_val) * numpy.ones(numpy.shape(x))
        elif n > 1:
            return numpy.zeros(numpy.shape(x))
        else:
            raise NotImplementedError("Derivative order " + str(n) + " not supported.")


class ZScoreNormalizer(Normalizer):
    """Normalizes data by setting the mean to 0 and std to 1

    Using the mean and standard deviation of the original training data,
    the training data is normalized so the mean becomes 0 and the standard
    deviation becomes 1.

    The tranformation equation is
    ..math::
        y = (x - mean_data) / std_data
        where :math:mean_data is the mean of the training data and
        :math:std_data is the standard deviation of the training data

    The inverse transformation equation is
    ..math::
        y = x * std_data + mean_data

    In the case where the original training data contains one variable,
    and thus the min and the max are the same, no normalization is
    applied.

    The properties ``mean_val`` and ``std_val`` store the mean and
    standard deviation of the original training data.
    """

    def __init__(self):
        super().__init__()
        self._mean_val = None
        self._std_val = None

    @property
    def mean_val(self):
        """float: mean of original training data"""
        return self._mean_val

    @property
    def std_val(self):
        """float: std of original training data"""
        return self._std_val

    def fit(self, x):
        """Obtain the original training data and reset mean and std

        Parameters
        ----------
        x : numerical data
            the training data

        Returns
        -------
        Normalizer
            the normalizer
        """
        x = numpy.array(x)
        self._mean_val = numpy.mean(x)
        try:
            self._std_val = float(numpy.std(numpy.array(x, dtype=numpy.float128)))
        except AttributeError:
            self._std_val = numpy.std(x)
        self._valid_cache = True
        return self

    def transform(self, x):
        """Normalize the data using the mean and std

        Using the mean and std of the training data, the input is
        scaled such that the mean of the original training data is 0 and
        the std of the original training data is 0.

        Parameters
        ----------
        x : numerical data
            data to be transformed

        Return
        ------
        numpy:ndarray
            the transformed data

        Raises
        ------
        ValueError
            normalizer was never fit.
        """
        if not self._valid_cache:
            raise ValueError("Normalizer needs fitting!")
        x = numpy.array(x)
        return (x - self.mean_val) / (self.std_val)

    def inverse_transform(self, x):
        """Inverse normalization function

        Using the mean and std of the training data, the input is
        scaled such that 1 becomes the mean of the original training data and
        the an std of 1 becomes the std of the original training data.

        Parameters
        ----------
        x : numerical data
            normalized data to be transformed

        Return
        ------
        numpy:ndarray
            the untransformed data

        Raises
        ------
        ValueError
            normalizer was never fit.
        """
        if not self._valid_cache:
            raise ValueError("Normalizer needs fitting!")
        x = numpy.array(x)
        return x * self.std_val + self.mean_val

    def derivative(self, x, n=1):
        """The derivative of the transformation.

        Parameters
        ----------
        x : array-like
            the input data

        n : int, optional
            order of derivative. Default is 1.

        Returns
        -------
        array-like
            derivative at x

        Raises
        ------
        ValueError
            normalizer was never fit.
        """
        if not self._valid_cache:
            raise ValueError("Normalizer needs fitting!")
        if n == 1:
            return 1 / (self.std_val) * numpy.ones(numpy.shape(x))
        elif n > 1:
            return numpy.zeros(numpy.shape(x))
        else:
            raise NotImplementedError("Derivative order " + str(n) + " not supported.")

    def inverse_derivative(self, x, n=1):
        """The derivative of the inverse transformation.

        Parameters
        ----------
        x : array-like
            the input data

        n : int, optional
            order of derivative. Default is 1.

        Returns
        -------
        array-like
            derivative at x

        Raises
        ------
        ValueError
            normalizer was never fit.
        """
        if not self._valid_cache:
            raise ValueError("Normalizer needs fitting!")
        if n == 1:
            return (self.std_val) * numpy.ones(numpy.shape(x))
        elif n > 1:
            return numpy.zeros(numpy.shape(x))
        else:
            raise NotImplementedError("Derivative order " + str(n) + " not supported.")


class SymmetricalLogNormalizer(Normalizer):
    r"""Transforms data onto the symmetrical logarithm scale
    
    A logarithmic scale is a nonlinear scale that is commonly used to 
    display data that grows exponentially or has a range with many 
    orders of magnitude. A logarithmic transform can only accept values 
    greater than zero. To get around this, a more flexible version of the 
    logarithmic scale can be used, known as the symmetrical log scale, for
    representing data with large magnitudes that is positive and negative. 
    This scale gets around the issue of log(0) being undefined by keeping
    the interval that contains 0 linear. In this transformation, negative
    outputs are the result of negative inputs, while in a log transformation
    negative outputs are the result of inputs less than 1. 

    The tranformation equation is 
    ..math::
        y = \begin{cases}
                \log_{10}(x/c+1) & \text{if } x > 0 \\
                -\log_{10}(-x/c+1)  & \text{if } x < 0 \\
                0 & \text{if } x = 0 \\
            \end{cases}
    
    in piece-wise form, and  
    ..math::
        y = \sgn(x) \log_{10}(1+\abs(x/c))
    
    using the sign function, where c is a constant that can be adjusted to 
    refine the linear interval near 0.

    The inverse transformation equation is
    ..math::
        y = \begin{cases}
                c(-1 + 10^{x}) & \text{if } x > 0 \\
                c(1 - 10^{-x})  & \text{if } x < 0 \\
                0 & \text{if } x = 0 \\
            \end{cases}
    
    in piece-wise form, and  
    ..math::
        y = \sgn(x) c (-1 + 10^{abs(x)})
    """

    def __init__(self, linthresh=1):
        super().__init__()
        self._valid_cache = True
        self._linthresh = linthresh

    @property
    def linthresh(self):
        """float: constant to determine size of linear interval around 0"""
        return self._linthresh

    @linthresh.setter
    def linthresh(self, value):
        if value <= 0:
            raise ValueError("linthresh must be greater than 0.")
        self._linthresh = value

    def transform(self, x):
        """Normalization function

        Parameters
        ----------
        x : numerical data
            data to be transformed

        Return
        ------
        normalized data
        """
        return numpy.sign(x) * numpy.log10(1 + numpy.abs(x) / self.linthresh)

    def inverse_transform(self, x):
        """Inverse normalization function

        Parameters
        ----------
        x : numerical data
            normalized data to be transformed

        Return
        ------
        unnormalized data
        """
        return numpy.sign(x) * self.linthresh * (-1 + numpy.power(10, numpy.abs(x)))

    def derivative(self, x, n=1):
        """The derivative of the transformation.

        Parameters
        ----------
        x : array-like
            the input data

        n : int, optional
            order of derivative. Default is 1.

        Returns
        -------
        array-like
            derivative at x
        """
        return (
            (-numpy.sign(x)) ** (n + 1)
            * math.factorial(n - 1)
            / (numpy.log(10) * (numpy.abs(x) + self.linthresh) ** n)
        )

    def inverse_derivative(self, x, n=1):
        """The derivative of the inverse transformation.

        Parameters
        ----------
        x : array-like
            the input data

        n : int, optional
            order of derivative. Default is 1.

        Returns
        -------
        array-like
            derivative at x
        """
        return (
            (numpy.sign(x)) ** (n + 1)
            * self.linthresh
            * (numpy.log(10) ** n)
            * numpy.power(10, numpy.abs(x))
        )

class AsinhNormalizer(Normalizer):
    r"""Transforms data using the inverse hyperbolic sine
    
    A nonlinear transformation that on small values close to
    zero causes little change but is asymptotically logarithmic
    on large absolute magnitudes, thus having a similar effect
    of a logarithmic scale on large values while also able to
    support negative values.
    
    The inverse hyperbolic sine (asinh or sinh^-1) is 
    
    :math::
        \sinh^{-1}(x) = \ln(x + \sqrt{x^{2} + 1})

    Its inverse, the hyperbolic sine (sinh) is

    :math::
        \sinh(x) = \frac{e^{x} - e^{-x}}{2}

    Its derivative is 

    :math::
        \frac{\mathrm{d} }{\mathrm{d} x} \sinh^{-1}(x) = \frac{1}{\sqrt{x^{2} + 1}}
    
    and contains no discontinuities.

    The parameter `linthresh` designates the range about 0 that
    will remain approximately linear after the transformation.
    The transformation is therefore :math:c*\sinh^{-1}(x/c) where
    c is `linthresh`.

    Parameters
    ----------
    linthresh : float, optional
        range about 0 that will be quasi-linear. Default is 1.
    """

    def __init__(self, linthresh=1):
        super().__init__()
        self._valid_cache = True
        self._linthresh = linthresh

    @property
    def linthresh(self):
        """float: constant to determine size of quasi-linear interval around 0"""
        return self._linthresh

    @linthresh.setter
    def linthresh(self, value):
        if value <= 0:
            raise ValueError("linthresh must be greater than 0.")
        self._linthresh = value

    def transform(self, x):
        """Normalization function

        Parameters
        ----------
        x : numerical data
            data to be transformed

        Return
        ------
        normalized data
        """
        x = numpy.array(x)
        return self.linthresh*numpy.arcsinh(x/self.linthresh)

    def inverse_transform(self, x):
        """Inverse normalization function

        Parameters
        ----------
        x : numerical data
            normalized data to be transformed

        Return
        ------
        unnormalized data
        """
        x = numpy.array(x)
        return self.linthresh*numpy.sinh(x/self.linthresh)

    def derivative(self, x, n=1):
        """The derivative of the transformation.

        Evaluates the 1st and 2nd derivative of the inverse hypobolic sin.

        Parameters
        ----------
        x : array-like
            the input data

        n : int, optional
            order of derivative. Default is 1.

        Returns
        -------
        array-like
            derivative at x

        Raises
        ------
        NotImplementedError
            Only 1st and 2nd derivative are supported.
        """
        x = numpy.array(x)
        if n == 1:
            return 1/numpy.sqrt((x**2)/(self.linthresh**2) + 1)
        elif n == 2:
            return - x/((self.linthresh**2)*numpy.power((x**2)/(self.linthresh**2) + 1,3/2))
        elif n == 3:
            return (2 * x**2 - self.linthresh**2) / (self.linthresh**4 * (x**2 / self.linthresh**2 + 1)**(5 / 2))
        else:
            raise NotImplementedError("Derivative order " + str(n) + " is not supported.")

    def inverse_derivative(self, x, n=1):
        """The derivative of the inverse transformation.

        Evaluates the 1st and 2nd derivative of the hypobolic sin.

        Parameters
        ----------
        x : array-like
            the input data

        n : int, optional
            order of derivative. Default is 1.

        Returns
        -------
        array-like
            derivative at x

        Raises
        ------
        NotImplementedError
            Only 1st and 2nd derivative are supported.
        """
        x = numpy.array(x)
        if n % 2 == 1:
            return numpy.cosh(x/self.linthresh)/(self.linthresh**(n - 1))
        elif n % 2 == 0:
            return numpy.sinh(x/self.linthresh)/(self.linthresh**(n - 1))
        else:
            raise NotImplementedError("Derivative order " + str(n) + " is not supported.")