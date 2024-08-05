import numpy


def make_clenshaw_curtis_exponential_level_sizes(num_levels):
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

    Parmeters
    ---------
    num_levels : int
        the number of levels to generate
    
    Returns
    -------
    num_per_level : list of ints
        number of elements per level
    """
    rule = lambda x: 1 if x == 0 else 2**x + 1
    num_per_level = numpy.ones(num_levels, dtype=int)
    num_per_level[1:] = [
        rule(i) - rule(i - 1) for i in range(1, num_levels)
    ]
    return num_per_level
    

def make_clenshaw_curtis_slow_exponential_level_sizes(num_levels):
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

    Parmeters
    ---------
    num_levels : int
        the number of levels to generate
    
    Returns
    -------
    num_per_level : list of ints
        number of elements per level
    """
    rule = lambda x: 1 if x == 0 else int(2 ** (numpy.ceil(numpy.log2(x)) + 1) + 1)
    num_per_level = numpy.ones(num_levels, dtype=int)
    num_per_level[1:] = [
        rule(i) - rule(i - 1) for i in range(1, num_levels)
    ]
    return num_per_level
    
def make_trigonometric_exponential_level_sizes(num_levels):
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
    
    Parmeters
    ---------
    num_levels : int
        the number of levels to generate
    
    Returns
    -------
    num_per_level : list of ints
        number of elements per level
    """

    rule = lambda x: 3**x
    num_per_level = numpy.ones(num_levels, dtype=int)
    num_per_level[1:] = [
        rule(i) - rule(i - 1) for i in range(1, num_levels)
    ]
    return num_per_level

def get_level_start_and_end(num_per_level):
    """Computes the start and end indexes for each level.
    
    For a list of elements that are divided into levels,
    each section that belongs to a specific level will have 
    start and ending indexes.
    
    Parameters
    ----------
    num_per_level : list of int
        the number of elements per level
    
    Returns
    -------
    start_level : list of int
        the starting index of each level
    
    end_level : list of int
        the ending index of each level
    """
    end_level = numpy.cumsum(num_per_level)
    start_level = end_level - num_per_level
    return start_level, end_level