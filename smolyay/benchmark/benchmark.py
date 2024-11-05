import abc

import numpy
from smolyay import samples

class BenchmarkFunction(abc.ABC):
    """Benchmark Function

    These functions operate on a defined `domain` they can be evaluated on,
    and the upper and lower bounds of this domain can be the domain in which
    solutions exist or can be arbitrary.

    """
    def __call__(self,x):
        """Evaluate the function.

        Parameters
        ----------
        x : list
            Function input.

        Raises
        ------
        IndexError
            If the shape of the input does not match dimension of domain
        ValueError
            If the input is outside the function domain.
        """
        if isinstance(x, samples.UnidimensionalPointSet):
            # input has a domain property, so just compare domain properties
            if self.dimension > 1:
                raise IndexError("Input must match dimension of domain")
            if x.domain[0]  < self.domain[0][0] or x.domain[1]  > self.domain[0][1]:
                raise ValueError("Input outside domain of function.")
            x = x.points
        elif isinstance(x, samples.MultidimensionalPointSet):
            # input has a domain property, so just compare domain properties
            if self.dimension != x.num_dimensions:
                raise IndexError("Input must match dimension of domain")
            if any(x.domain[i][0] < self.domain[i][0] or x.domain[i][1] > self.domain[i][1]
                for i in range(self.dimension)):
                    raise ValueError("Input outside domain of function.")
            x = x.points
        else:
            # does not have a domain property, need to check all values are in domain
            x = numpy.array(x, copy=None, ndmin=2)
            if self.dimension == 1:
                if x.ndim == 2 and x.shape[0] == 1 and x.shape[1] > 1:
                    # the cast to 2d puts these in wrong order, so transpose
                    x = x.T
                else:
                    x = x[..., numpy.newaxis]
            if x.shape[-1] != self.dimension:
                raise IndexError("Input must match dimension of domain")
            if any(
                numpy.any(x[..., i] < self.domain[i][0])
                or numpy.any(x[..., i] > self.domain[i][1])
                for i in range(self.dimension)
            ):
                raise ValueError("Input outside domain of function.")
        return numpy.squeeze(self._function(x))
    
    @property
    def name(self):
        """str: Name of the function"""
        return type(self).__name__

    @property
    def dimension(self):
        """int: Number of variables."""
        return len(self.domain)

    @property
    def lower_bounds(self):
        """list: the lower bounds of the domain of each variable."""
        return [bound[0] for bound in self.domain]
    
    @property
    def upper_bounds(self):
        """list: the upper bounds of the domain of each variable."""
        return [bound[1] for bound in self.domain]

    @property
    @abc.abstractmethod
    def domain(self):
        """list: Domain of the function.
        
        The domain must be specified as lower and upper bounds for each variable as a list of lists.
        """
        pass
 
    @abc.abstractmethod
    def _function(self,x):
        pass



