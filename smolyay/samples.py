import abc
import collections.abc
import itertools

import numpy
import scipy.stats.qmc

from smolyay import _growth


class UnidimensionalPointSet(collections.abc.Sequence):
    """Set of unidimensional points

    A set of unique unidimensional points within a domain.

    Parameters
    ----------
    domain: list
        Domain of the sample points.
    """

    def __init__(self, domain):
        self._points = None
        self._domain = None
        self._valid_cache = False

        self.domain = domain

    @property
    def domain(self):
        """numpy.ndarray: Domain of the `points`."""
        return self._domain

    @domain.setter
    def domain(self, value):
        domain = numpy.sort(numpy.array(value, dtype=float))
        if domain.shape != (2,):
            raise TypeError("Domain must be array with two variables")
        if domain[0] >= domain[1]:
            raise ValueError("Lower bound must be less than upper bound")
        if not numpy.array_equal(self._domain, domain):
            self._domain = domain
            self._valid_cache = False

    @property
    def points(self):
        """numpy.ndarray: Points in the set."""
        if not self._valid_cache:
            self._create()
            self._valid_cache = True
        return self._points

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, key):
        return self.points[key]

    @abc.abstractmethod
    def _create(self):
        """Create the points in the set."""
        pass

    def _scale_to_domain(self, points, old_domain):
        points = self.domain[0] + (self.domain[1] - self.domain[0]) * (
            (points - old_domain[0]) / (old_domain[1] - old_domain[0])
        )
        numpy.clip(points, self.domain[0], self.domain[1], out=points)
        return points


class NestedUnidimensionalPointSet(UnidimensionalPointSet):
    """Set of unidimensional points assigned to levels.


    Parameters
    ----------
    domain: list
        Domain of the sample points.

    num_levels: int
       The number of levels.
    """

    def __init__(self, domain, num_levels):
        super().__init__(domain)
        self._num_levels = None
        self._num_per_level = None
        self._start_level = None
        self._end_level = None

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
            self._num_levels = num_levels
            self._valid_cache = False

    @property
    def num_per_level(self):
        """numpy.ndarray: number of points per level."""
        if not self._valid_cache:
            self._create()
            self._valid_cache = True
        return self._num_per_level

    @property
    def start_level(self):
        """numpy.ndarray: the starting index of each level."""
        if not self._valid_cache:
            self._create()
            self._valid_cache = True
        return self._start_level

    @property
    def end_level(self):
        """numpy.ndarray: the ending index of each level."""
        if not self._valid_cache:
            self._create()
            self._valid_cache = True
        return self._end_level

    def level(self, index):
        """numpy.ndarray: Points in a level"""
        return self.points[self.start_level[index] : self.end_level[index]]


class ClenshawCurtisPointSet(UnidimensionalPointSet):
    r"""Set of unidimensional points for Clenshaw Curtis sampling

    The :attr:`points` for this interpolation scheme are the extrema of the
    Chebyshev polynomials of the first kind on the domain :math:`[-1, 1]`:

    .. math::

        x_i^* = -\cos(\pi i/n), i = 0,...,n

    For the special case :math:`n = 0`, there is only one point :math:`x_0^* = 0`.

    The points are then scaled from the domain :math:`[-1, 1]` to the domain
    specified by the parameter `domain`.

    Parameters
    ----------
    domain: list
        Domain of the sample points.

    degree: int
        Degree of the Chebyshev polynomial of the first kind to get extrema from.
    """

    def __init__(self, domain, degree):
        super().__init__(domain)
        self._degree = None

        self.degree = degree

    @property
    def degree(self):
        """int: degree of polynomial to create points for."""
        return self._degree

    @degree.setter
    def degree(self, value):
        degree = int(value)
        if degree < 0:
            raise ValueError("Degree must be 0 or greater")
        if degree != self._degree:
            self._degree = degree
            self._valid_cache = False

    def _create(self):
        r"""Create the points in the set.

        Generating the extrema of a Cheybshev polynomial of the first kind at
        a given degree.
        """
        if self.degree > 0:
            points = -numpy.cos(
                numpy.pi * numpy.linspace(0, self.degree, self.degree + 1) / self.degree
            )
        else:
            # special case where degree == 0
            points = numpy.zeros(1)
        self._points = self._scale_to_domain(points, [-1, 1])


class NestedClenshawCurtisPointSet(NestedUnidimensionalPointSet):
    r"""Generate nested Clenshaw Curtis points

    The :attr:`points` for this interpolation scheme are the nested Clenshaw
    Curtis points, which are the extrema of the Chebyshev polynomials of the
    first kind with degree :math:`n` where n is part of a sequence
     
    .. math::

        n = \begin{cases}
            0 & \text{ if } L = 0\\ 
            2^{L} & \text{ if } L > 0 
        \end{cases}

    where L is a whole number.

    As the extrema of the Chebyshev polynomials of degree :math:`n` are nested,
    meaning any :math:`n` will have extrema at the same points as every n that
    precedes it in the sequence. The extrema are organized into levels such that
    a level L will contain the extrema of Chebyshev polynomial n(L) that are
    not the extrema of any Chebyshev polynomial n(k) where k is a whole number
    and k < L.

    The order :math:`o(L) of the nested Clenshaw Curtis set, which describes the
    number of cummulative points at each level, is used to determine the 
    number of points at each individual level. The order of the Clenshaw Curtis
    set is described by the following equation:

    .. math::

        o(L) = \begin{cases}
                1 & \text{ if } L = 0\\ 
                2^{L} + 1 & \text{ if } L > 0 
        \end{cases}

    which leads to a sequence :math:`{1, 3, 5, 9, 17, ...}`.

    Determining the number of points each level is then

    .. math::
        num_per_level(L) = o(L) - o(L - 1)

    The points are then scaled from the domain :math:`[-1, 1]` to the domain
    specified by the :attr:`domain`.

    Parameters
    ----------
    domain: list
        Domain of the sample points.

    num_levels: int
        The number of levels.
    """

    def _create(self):
        r"""Create the points in the set.

        Generating nested extrema of chebyshev polynomials of the first kind.
        """
        # create properties for levels, level 0 is a special case with 1 point
        self._num_per_level = _growth.clenshaw_curtis_exponential_growth(self.num_levels)
        self._start_level,self._end_level =_growth.get_level_start_and_end(self._num_per_level)

        # points, level 0 is a special case only 0 as a point
        num_points = self._end_level[-1]
        points = numpy.zeros(num_points, dtype=float)
        for i in range(1, self.num_levels):
            # find indexes of extrema not already found. Fraction index/degree
            # cannot be further simplified
            degree = 2**i
            if i == 1:
                # special case for level == 1
                indexes = numpy.arange(0, degree + 1, 2)
            else:
                indexes = numpy.arange(1, degree, 2, dtype=int)
                divisible_indexes = numpy.gcd(indexes, degree) > 1
                indexes = indexes[~divisible_indexes]
            points[self._start_level[i] : self._end_level[i]] = -numpy.cos(
                numpy.pi * indexes / degree
            )
        self._points = self._scale_to_domain(points, [-1, 1])


class SlowNestedClenshawCurtisPointSet(NestedUnidimensionalPointSet):
    r"""Set for Clenshaw Curtis slow exponential growth.

    The :attr:`points` for this interpolation scheme are the nested Clenshaw
    Curtis points.

    As the extrema of the Chebyshev polynomials are organized into levels such
    that a level L that is nonempty will contain extrema of some degree n that
    are not found in a preceding n. The total number of unique points increases 
    exponentially with increasing n.
    
    While the total number of unique points increases exponentially with
    increasing n, the rate of increasing points with respect to level L can be
    limited at or below a linear :math:`2*L + 1`. To do this, if adding the
    next set of points at a level L causes the rate of new points to go above
    :math:`2*L + 1`, then the points will not be added and the level will be
    empty.

    The order :math:`o(L) of the slow nested Clenshaw Curtis set describes
    the number of cummulative points at each level to account for the limiting
    rate :math:`2*L + 1`. Then is used to determine the number of points at
    each individual level. The order of the slow Clenshaw Curtis set is
    described by the following equation:

    .. math::

        o(L) = \begin{cases}
                1 & \text{ if } L = 0\\ 
                2^{k} + 1 & \text{ if } L > 0 
        \end{cases}

        where k = \left \lceil \log_{2}(L) \right \rceil + 1

    The sequence of o(L) is then :math:`{1, 3, 5, 9, 9, 17, ...}`.
    
    Since the points are nested, the number of unique points at
    a level L is 
    
    .. math::
        num_per_level(L) = o(L) - o(L - 1)    

    The points are then scaled from the domain :math:`[-1, 1]` to the domain
    specified by the :attr:`domain`.

    Parameters
    ----------
    domain: list
        Domain of the sample points.

    num_levels: int
        The number of levels.
    """

    def _create(self):
        r"""Create the points in the set.

        Generating nested extrema of chebyshev polynomials of the first kind.
        """
        # create properties for levels, level 0 is a special case with 1 point
        self._num_per_level = _growth.clenshaw_curtis_slow_exponential_growth(self.num_levels)
        self._start_level,self._end_level =_growth.get_level_start_and_end(self._num_per_level)

        # points, level 0 is a special case only 0 as a point
        num_points = self._end_level[-1]
        points = numpy.zeros(num_points, dtype=float)
        for i in range(1, self.num_levels):
            if self._num_per_level[i] == 0:
                continue
            # find indexes of extrema not already found. Fraction index/degree
            # cannot be further simplified
            degree = int(2 ** (numpy.ceil(numpy.log2(i)) + 1))
            if i == 1:
                # special case for level == 1
                indexes = numpy.arange(0, degree + 1, 2)
            else:
                indexes = indexes = numpy.arange(1, degree, 2, dtype=int)
                divisible_indexes = numpy.gcd(indexes, degree) > 1
                indexes = indexes[~divisible_indexes]
            points[self._start_level[i] : self._end_level[i]] = -numpy.cos(
                numpy.pi * indexes / degree
            )
        self._points = self._scale_to_domain(points, [-1, 1])


class TrigonometricPointSet(UnidimensionalPointSet):
    r"""Set of unidimensional points for Trigonometric sampling

    The :attr:`points` for this interpolation scheme are the
    equidistant trigonometric interpolation points for m points

    .. math::

        x^f_j = \frac{2\pi j}{m(f)},  1 \leq j \leq m(l), f \geq 0

    where f represents the positive frequency of the complex
    trigonometric polynomial of the form

    .. math::

        p(x) = \sum_{n=-f}^{f} a_{n}\exp(xi * n)

    where the sequence :math:`a_{n}` are coefficients. The
    relationship between the frequency and the number of points
    is set at

    .. math::
        `m = 2*\left | f \right | + 1`

    to ensure that m is equal to the number of coefficients in
    the trigonometric polynomial.

    The points are then scaled from the domain :math:`[0, 2\pi]`
    to the domain specified by the parameter `domain`.

    Parameters
    ----------
    domain: list
        Domain of the sample points.

    frequency: int
        The frequency to take points from.
    """

    def __init__(self, domain, frequency):
        super().__init__(domain)
        self._frequency = None

        self.frequency = frequency

    @property
    def frequency(self):
        """int: frequency to create points for."""
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        frequency = int(value)
        if frequency < 0:
            raise ValueError("Frequency must be 0 or greater")
        if frequency != self._frequency:
            self._frequency = frequency
            self._valid_cache = False

    def _create(self):
        r"""Create the points in the set.

        Generating trigonometic points at a given frequency.
        """
        num_points = 2 * self.frequency + 1
        indexes = numpy.arange(num_points)
        points = 2 * numpy.pi * indexes / num_points
        self._points = self._scale_to_domain(points, [0, 2 * numpy.pi])


class NestedTrigonometricPointSet(NestedUnidimensionalPointSet):
    r"""Set of unidimensional points for Trigonometric sampling

    The :attr:`points` for this interpolation scheme are nested
    trigonometric points

    .. math::

        x^l_j = \frac{j-1}{m(l)},  1 \leq j \leq m(l), l \geq 0

    These points are nested, such that the order of elements in
    `points` corresponds to the indices in `levels`.

    To determine the number of points per level, an order(L) is
    used to describe the number of points at each level L. For
    these Trigonometric points, ensuring the total number of
    points is an odd number, this order equation is

    .. math::

        o(L) = 3^{L}

    Since the points are nested, the number of unique points at
    a level L is

    .. math::
        num_per_level(L) = o(L) - o(L - 1)

    The points are then scaled from the domain :math:`[0, 2\pi]`
    to the domain specified by the :attr:`domain`.

    Parameters
    ----------
    domain: list
        Domain of the sample points.

    num_levels: int
        The number of levels.
    """

    def _create(self):
        r"""Create the points in the set.

        Generating the trignometric points using the frequencies
        :math:1, 3, 9, ..., 3^{i} where i is an integer.
        """
        # create properties for levels, level 0 is a special case with 1 point
        self._num_per_level = _growth.trigonometric_exponential_growth(self.num_levels)
        self._start_level,self._end_level =_growth.get_level_start_and_end(self._num_per_level)

        # points, level 0 is a special case only 0 as a point
        num_points = self._end_level[-1]
        points = numpy.zeros(num_points, dtype=float)
        for i in range(self.num_levels):
            # find fraction index/num_points that cannot be further simplified
            num_points = 3**i
            if i == 0:
                # special case where level == 0
                indexes = numpy.arange(0, num_points, dtype=int)
            else:
                indexes = numpy.arange(1, num_points + 1, 1, dtype=int)
                divisible_indexes = numpy.gcd(indexes, num_points) > 1
                indexes = indexes[~divisible_indexes]
            points[self._start_level[i] : self._end_level[i]] = (
                2 * numpy.pi * indexes / num_points
            )
        self._points = self._scale_to_domain(points, [0, 2 * numpy.pi])


class MultidimensionalPointSet(abc.ABC):
    """Multidimensional set of points."""

    def __init__(self):
        self._domain = None
        self._points = None
        self._valid_cache = False

    @property
    def num_dimensions(self):
        """int: number of independent variables."""
        return self.domain.shape[0]

    @property
    def domain(self):
        """numpy.ndarray: domain of the point set."""
        return self._domain

    @property
    def points(self):
        """numpy.ndarray: Points to be sampled."""
        if not self._valid_cache:
            self._create()
            self._valid_cache = True
        return self._points

    def __len__(self):
        return self.points.shape[0]

    @abc.abstractmethod
    def _create(self):
        """Abstract method for generating the mulitdimensional grid set"""
        pass


class RandomPointSet(MultidimensionalPointSet):
    """Point Set for random points

    Abstract class for multidimensional point sets that generate
    random points using a seed.

    Parameters
    ----------
    domain: list
        domain of the random points

    num_points: int
        number of points to generate

    seed: int
        seed for generating the random points
    """

    def __init__(self, domain, num_points, seed):
        super().__init__()
        self._num_points = None
        self._seed = None

        self.domain = domain
        self.num_points = num_points
        self.seed = seed

    @MultidimensionalPointSet.domain.setter
    def domain(self, value):
        domain = numpy.sort(numpy.array(value, ndmin=2), axis=1)
        if domain.ndim != 2 or domain.shape[1] != 2:
            raise TypeError("Domain must have size (num_dimensions, 2)")
        if numpy.any(domain[:, 0] >= domain[:, 1]):
            raise ValueError("Lower bound must be less than upper bound")
        if not numpy.array_equal(self._domain, domain):
            self._domain = domain
            self._valid_cache = False

    @property
    def num_points(self):
        """int: Number of points"""
        return self._num_points

    @num_points.setter
    def num_points(self, value):
        num_points = int(value)
        if self._num_points != num_points:
            self._num_points = num_points
            self._valid_cache = False

    @property
    def seed(self):
        """int: random seed for generating points"""
        return self._seed

    @seed.setter
    def seed(self, value):
        value = int(value)
        if self._seed != value:
            self._seed = value
            self._valid_cache = False


class UniformRandomPointSet(RandomPointSet):
    """Points drawn from a uniform random distribution."""

    def _create(self):
        """Generate a set of random points

        Generates uniformly distributed random points
        """
        lower_bounds = self.domain[:, 0]
        upper_bounds = self.domain[:, 1]
        num_dimensions = len(lower_bounds)
        p_gen = numpy.random.default_rng(seed=self.seed).uniform(
            size=(self.num_points, num_dimensions)
        )
        self._points = scipy.stats.qmc.scale(p_gen, lower_bounds, upper_bounds)


class _QMCRandomPointSet(RandomPointSet):
    """Points drawn randomly using a QMCEngine

    This random point set relies on the QMCEngine objects in
    the scipy.stats.qmc module to generate points.

    In addition to the domain, number of points, and the seed
    to generate the points, all QMCEngines have 2 additional
    optional parameters, ``scramble`` and ``optimization``.

    Parameters
    ----------
    domain: list
        domain of the random points

    num_points: int
        number of points to generate

    seed: int
        seed for generating the random points

    scramble : bool
        Default True. Applies centering to Latin Hypercube points, Owen
        scrambling to Halton points, and LMS+shift scrambling to Sobol points.

    optimization : {None, "random-cd", "lloyd"}
        Default None. If "random-cd" the coordinates of points are adjusted to
        lower the centered discrepancy. If "lloyd", adjust points using a
        Lloyd-Max algorithm to encourage even spacing.
    """

    def __init__(self, domain, num_points, seed, scramble=True, optimization=None):
        super().__init__(domain, num_points, seed)
        self._scramble = None
        self._optimization = None

        self.scramble = scramble
        self.optimization = optimization

    @property
    def scramble(self):
        """bool: If True, applies centering to Latin Hypercube points, Owen
        scrambling to Halton points, and LMS+shift scrambling to Sobol points."""
        return self._scramble

    @scramble.setter
    def scramble(self, value):
        scramble = bool(value)
        if self._scramble != scramble:
            self._scramble = scramble
            self._valid_cache = False

    @property
    def optimization(self):
        """{None, “random-cd”, “lloyd”}: perform post-processing on points"""
        return self._optimization

    @optimization.setter
    def optimization(self, value):
        if not value in [None, "random-cd", "lloyd"]:
            raise TypeError("optimization must be None, random-cd, or lloyd")
        if self._optimization != value:
            self._optimization = value
            self._valid_cache = False


class LatinHypercubeRandomPointSet(_QMCRandomPointSet):
    """Points drawn randomly using a Latin hypercube.


    Parameters
    ----------
    domain: list
        domain of the random points

    num_points: int
        number of points to generate

    seed: int
        seed for generating the random points

    scramble : bool
        Default True. Applies centering to Latin Hypercube points.

    optimization : {None, "random-cd", "lloyd"}
        Default None. If "random-cd" the coordinates of points are adjusted to
        lower the centered discrepancy. If "lloyd", adjust points using a
        Lloyd-Max algorithm to encourage even spacing.

    strength : {1, 2}, optional
        Default 1. If 1, produces a normal LHS. If 2, produces an orthogonal
        array based LHS.
    """

    def __init__(
        self, domain, num_points, seed, scramble=True, optimization=None, strength=1
    ):
        super().__init__(domain, num_points, seed, scramble, optimization)
        self._strength = None

        self.strength = strength

    @property
    def strength(self):
        """{1, 2}: Whether to create an orthogonal array of points"""
        return self._strength

    @strength.setter
    def strength(self, value):
        strength = int(value)
        if not strength in [1, 2]:
            raise TypeError("Strength must be 1 or 2")
        if self._strength != strength:
            self._strength = strength
            self._valid_cache = False

    def _create(self):
        """Generate a set of random points

        Generates a set of monte carlo LatinHypercube points.
        """
        lower_bounds = self.domain[:, 0]
        upper_bounds = self.domain[:, 1]
        p_gen = scipy.stats.qmc.LatinHypercube(
            self.num_dimensions,
            scramble=self.scramble,
            strength=self.strength,
            optimization=self.optimization,
            seed=self.seed,
        ).random(n=self.num_points)
        self._points = scipy.stats.qmc.scale(p_gen, lower_bounds, upper_bounds)


class HaltonRandomPointSet(_QMCRandomPointSet):
    """Points drawn randomly using a Halton sequence.

    Parameters
    ----------
    domain: list
        domain of the random points

    num_points: int
        number of points to generate

    seed: int
        seed for generating the random points

    scramble : bool
        Default True. Applies Owen scrambling to Halton points.

    optimization : {None, "random-cd", "lloyd"}
        Default None. If "random-cd" the coordinates of points are adjusted to
        lower the centered discrepancy. If "lloyd", adjust points using a
        Lloyd-Max algorithm to encourage even spacing.

    """

    def _create(self):
        """Generate a set of random points

        Generates a set of quasi-monte carlo Halton Sequence points.
        """
        lower_bounds = self.domain[:, 0]
        upper_bounds = self.domain[:, 1]
        p_gen = scipy.stats.qmc.Halton(
            self.num_dimensions,
            scramble=self.scramble,
            optimization=self.optimization,
            seed=self.seed,
        ).random(n=self.num_points)
        self._points = scipy.stats.qmc.scale(p_gen, lower_bounds, upper_bounds)


class SobolRandomPointSet(_QMCRandomPointSet):
    """Points drawn randomly using a Sobol sequence.

    Parameters
    ----------
    domain: list
        domain of the random points

    num_points: int
        number of points to generate. Must be a power of 2.

    seed: int
        seed for generating the random points

    scramble : bool
        Default True. Applies LMS+shift scrambling to Sobol points.

    optimization : {None, "random-cd", "lloyd"}
        Default None. If "random-cd" the coordinates of points are adjusted to
        lower the centered discrepancy. If "lloyd", adjust points using a
        Lloyd-Max algorithm to encourage even spacing.

    bits : int
        Default 30. Sets the max number of points that can be generated,
        which is 2**bits.

    Raises
    ------
    ValueError
        number of points must be a power of two
    ValueError
        number of points must be less than 2**bits
    """

    def __init__(
        self, domain, num_points, seed, scramble=True, optimization=None, bits=30
    ):
        self._bits = 64  # bits must have a value before num_points can be set
        super().__init__(domain, num_points, seed, scramble, optimization)
        self.bits = bits

    @property
    def bits(self):
        """int: Sets max number of points to generate at 2**bits."""
        return self._bits

    @bits.setter
    def bits(self, value):
        bits = int(value)
        if bits > 64:
            raise ValueError("bits max value is 64")
        elif bits < 1:
            raise ValueError("bits must be at least 1")
        elif 2**bits < self.num_points:
            raise ValueError("2**bits must be greater than number of points")

        if self._bits != bits:
            self._bits = bits
            self._valid_cache = False

    @property
    def num_points(self):
        """int: random seed for generating points"""
        return self._num_points

    @num_points.setter
    def num_points(self, value):
        num_points = int(value)
        if num_points & (num_points - 1) != 0:
            raise ValueError("Number of points must be power of 2")
        elif num_points > 2**self.bits:
            raise ValueError("Number of points must be less than 2**bits")

        if self._num_points != num_points:
            self._num_points = num_points
            self._valid_cache = False

    def _create(self):
        """Generate a set of random points

        Generates a set of quasi-monte carlo Sobol Sequence points.
        """
        lower_bounds = self.domain[:, 0]
        upper_bounds = self.domain[:, 1]
        p_gen = scipy.stats.qmc.Sobol(
            self.num_dimensions,
            scramble=self.scramble,
            bits=self.bits,
            optimization=self.optimization,
            seed=self.seed,
        ).random(n=self.num_points)
        self._points = scipy.stats.qmc.scale(p_gen, lower_bounds, upper_bounds)


class PointSetProduct(MultidimensionalPointSet):
    """Points generated from combinations of unidimensional points.

    Parameters
    ----------
    point_sets : list of :class:UnidimensionalPointSet
        unique set of 1D points
    """

    def __init__(self, point_sets):
        super().__init__()
        self._point_sets = point_sets

    @property
    def domain(self):
        """numpy.ndarray: domain of the point set."""
        return numpy.array([up.domain for up in self.point_sets])

    @domain.setter
    def domain(self, value):
        domain = numpy.sort(numpy.array(value, ndmin=2), axis=1)
        if domain.ndim != 2 or domain.shape[1] != 2:
            raise TypeError("Domain must have size (num_dimensions, 2)")
        if numpy.any(domain[:, 0] >= domain[:, 1]):
            raise ValueError("Lower bound must be less than upper bound")
        if len(self.point_sets) != domain.shape[0]:
            raise IndexError("Domain does not match number of point sets")
        if not numpy.array_equal(self._domain, domain):
            self._domain = domain
            for ps, d in zip(self.point_sets, domain):
                ps.domain = d
            self._valid_cache = False

    @property
    def point_sets(self):
        """list of :class:UnidimensionalPointSet set of unique points"""
        return self._point_sets


class TensorProductPointSet(PointSetProduct):
    """Points generated using all combinations of unidimensional points.

    Depending on the dimensionality, and points provided by
    :class:`UnidimensionalPointSet`, make full tensor grids.
    :meth:`generates_points` generates a full tensor grid

    Parameters
    ----------
    point_sets : list of :class:UnidimensionalPointSet
        unique set of 1D points
    """

    def _create(self):
        """Generating the tensor grid

        Using the parameters of the class, generate a grid set using all
        combinations of unidimensional points
        """
        num_points = numpy.prod([len(p) for p in self._point_sets])
        self._points = numpy.zeros((num_points, len(self._point_sets)), dtype=float)
        for i, point in enumerate(itertools.product(*self._point_sets)):
            self._points[i] = point


class SmolyakSparseProductPointSet(PointSetProduct):
    """Points generated from sparse combinations of unidimensional points.

    Depending on the dimensionality, and levels of each point set in
    param:`point_sets`, makes sparse smolyak grids.

    Parameters
    ----------
    point_sets : list of :class:UnidimensionalPointSet
        unique set of 1D points
    """

    def _create(self):
        """Generating the sparse smolyak grid

        Using the parameters of the class, generate a grid set using all
        valid smolyak combinations of unidimensional points
        """
        # reset points
        self._points = None

        # find number of levels per dimension
        num_levels_per_dim = [ob.num_levels for ob in self._point_sets]
        max_num_levels = max(num_levels_per_dim)

        # get the combinations of levels based on maximum possible number of levels
        level_combinations = []
        for sum_of_levels in range(max_num_levels):
            level_combinations.extend(
                list(
                    _generate_compositions(
                        sum_of_levels, self.num_dimensions, include_zero=True
                    )
                )
            )
        level_combinations = numpy.array(level_combinations)

        # remove combinations where a dimension exceeds its number of levels
        # only check if point sets have different numbers of levels
        if min(num_levels_per_dim) != max_num_levels:
            valid_comb = numpy.all(
                numpy.less(level_combinations, num_levels_per_dim), axis=1
            )
            level_combinations = level_combinations[valid_comb]

        # generate sets of points based on combinations of levels
        for level_comb in level_combinations:
            level_point_combinations = [
                self._point_sets[d].level(level) for d, level in enumerate(level_comb)
            ]
            num_points = numpy.prod([len(p) for p in level_point_combinations])
            points_ = numpy.zeros((num_points, self.num_dimensions), dtype=float)
            for i, point in enumerate(itertools.product(*level_point_combinations)):
                points_[i] = point

            # add newly generated points to set
            if self._points is None:
                self._points = points_
            else:
                self._points = numpy.concatenate((self._points, points_), axis=0)


def _generate_compositions(value, num_parts, include_zero):
    """Generate compositions of a value into num_parts parts.

    The algorithm that is being used is NEXCOM and can be found in
    "Combinatorial Algorithms For Computers and Calculators",
    Second Edition, 1978.
    Authors: ALBERT NIJENHUIS and HERBERT S. WILF.
    https://doi.org/10.1016/C2013-0-11243-3
    ``include_zero`` parameter determines whether the compositions
    that contain zeros are parts of the output or not.
    The first composition will be (``value``, 0, ..., 0)
    (or (``value-num_parts``, 0, ..., 0) if ``include_zero`` is ``False``).
    Next, the first component (index = 0) is dropped by 1, and next component
    (index = 1) is incremented by 1. This goes on until index 0 reaches ``0``.
    Once, generated all compositions, the  next component is incremented
    by 1 and is fixed until all compositions are generated with the same
    method. This goes on until the last component reaches the ``value``
    (or ``value - num_parts`` if ``include_zero`` is ``False``).
    It is important to note that if ``include_zero`` is ``False``,
    all components will be incremented by 1.

    Parameters
    ----------
    value: int
        Value.
    num_parts: int
        Number of parts.
    include_zero : bool
        True if compositions contain zero, False otherwise.

    Yields
    ------
    list
        All possible compositions of the value into num_parts.

    Raises
    ------
    ValueError
        Number of parts cannot be greater than value if the desired output
        does not include compositions containing zeroes.
    """
    if value < num_parts and include_zero is False:
        raise ValueError(
            "When include_zero is {}, num_parts cannot be greater"
            " than the value".format(False)
        )
    value = value if include_zero else value - num_parts

    # (A) first entry
    r = [0] * num_parts
    r[0] = value
    t = value
    h = 0
    yield list(r) if include_zero else (numpy.array(r) + 1).tolist()

    # (D)
    while r[num_parts - 1] != value:
        # (B)
        if t != 1:
            h = 0

        # (C)
        h += 1
        t = r[h - 1]
        r[h - 1] = 0
        r[0] = t - 1
        r[h] += 1
        yield list(r) if include_zero else (numpy.array(r) + 1).tolist()
