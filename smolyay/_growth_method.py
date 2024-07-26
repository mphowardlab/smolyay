import numpy


class ClenshawCurtisExponentialGrowthMixin:
    r"""Nested levels for Clenshaw Curtis Exponential Growth
    
    Describes a nested set of levels in which the cumulative
    number of elements grows with each new level added via 
    the following equation:

    .. math::

        o(L) = \begin{cases}
                1 & \text{ if } L = 0\\ 
                2^{L} + 1 & \text{ if } L > 0 
        \end{cases}

    which leads to a sequence :math:`{1, 3, 5, 9, 17, ...}`.

    Determining the number of points each level is then

    .. math::
        num_per_level(L) = o(L) - o(L - 1)
    """

    def _create_levels(self, num_levels):
        rule = lambda x: 1 if x == 0 else 2**x + 1
        self._num_per_level = numpy.ones(num_levels, dtype=int)
        self._num_per_level[1:] = [
            rule(i) - rule(i - 1) for i in range(1, num_levels)
        ]
        self._end_level = numpy.cumsum(self._num_per_level)
        self._start_level = self._end_level - self._num_per_level
    

class ClenshawCurtisSlowExponentialGrowthMixin:
    r"""Nested levels for Clenshaw Curtis Slow Exponential Growth
    
    Describes a nested set of levels in which the cumulative
    number of elements grows with each new level added via 
    the following equation:

    .. math::

        o(L) = \begin{cases}
                1 & \text{ if } L = 0\\ 
                2^{k} + 1 & \text{ if } L > 0 
        \end{cases}

        where k = \left \lceil \log_{2}(L) \right \rceil + 1

    The sequence of o(L) is then :math:`{1, 3, 5, 9, 9, 17, ...}`.

    This rate is capped such that the cumulative number of elements
    at any L is capped by a limiting rate :math:`2*L + 1`.

    Determining the number of points each level is then

    .. math::
        num_per_level(L) = o(L) - o(L - 1)
    """

    def _create_levels(self, num_levels):
        rule = lambda x: 1 if x == 0 else int(2 ** (numpy.ceil(numpy.log2(x)) + 1) + 1)
        self._num_per_level = numpy.ones(num_levels, dtype=int)
        self._num_per_level[1:] = [
            rule(i) - rule(i - 1) for i in range(1, num_levels)
        ]
        self._end_level = numpy.cumsum(self._num_per_level)
        self._start_level = self._end_level - self._num_per_level
    
class TrigonometricExponentialGrowthMixin:
    r"""Nested levels for Trigonometric Exponential Growth
    
    Describes a nested set of levels in which the cumulative
    number of elements grows with each new level added via 
    the following equation:

    .. math::

        o(L) = 3^{L}

    The sequence of o(L) is then :math:`{1, 3, 9, ...}`.

    Determining the number of points each level is then

    .. math::
        num_per_level(L) = o(L) - o(L - 1)
    """

    def _create_levels(self, num_levels):
        rule = lambda x: 3**x
        self._num_per_level = numpy.ones(num_levels, dtype=int)
        self._num_per_level[1:] = [
            rule(i) - rule(i - 1) for i in range(1, num_levels)
        ]
        self._end_level = numpy.cumsum(self._num_per_level)
        self._start_level = self._end_level - self._num_per_level