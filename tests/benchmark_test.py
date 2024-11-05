import pytest
import warnings

import numpy
import importlib
import inspect

import smolyay
from smolyay.benchmark import BenchmarkFunction, branin


class TestClass1D(BenchmarkFunction):
    """A test class for testing BenchmarkFunction objects with 1 dimension"""

    @property
    def domain(self):
        return [[0.4, 10]]

    def _function(self, x):
        return numpy.squeeze(4 * 0.5 * ((2 / x) ** 12 - (2 / x) ** 6))


class TestClass3D(BenchmarkFunction):
    """A test class for testing BenchmarkFunction objects with >1 dimensions"""

    @property
    def domain(self):
        return [[0, 10], [-5, -1], [11, 13]]

    def _function(self, x):
        return x[..., 0] ** 0.5 + x[..., 1] * 3 - x[..., 2] / 3 + x[..., 0] / x[..., 1]


"""List of non-abstract class objects used as benchmark functions"""
functions = []
for name, cls in inspect.getmembers(
    importlib.import_module("smolyay.benchmark"), inspect.isclass
):
    if not inspect.isabstract(cls):
        f = cls()
        functions.append(f)


@pytest.fixture
def test_class_1():
    return TestClass1D()


@pytest.fixture
def test_class_3():
    return TestClass3D()


def test_name(test_class_1, test_class_3):
    """Test name is the name of the function"""
    assert test_class_1.name == "TestClass1D"
    assert test_class_3.name == "TestClass3D"


def test_domain(test_class_1, test_class_3):
    """Test return domain as list of lists"""
    assert test_class_1.domain == [[0.4, 10]]
    assert test_class_3.domain == [[0, 10], [-5, -1], [11, 13]]


def test_lower_bounds(test_class_1, test_class_3):
    """Test return lower bounds"""
    assert test_class_1.lower_bounds == [0.4]
    assert test_class_3.lower_bounds == [0, -5, 11]


def test_upper_bounds(test_class_1, test_class_3):
    """Test return upper bounds"""
    assert test_class_1.upper_bounds == [10]
    assert test_class_3.upper_bounds == [10, -1, 13]


def test_dimension(test_class_1, test_class_3):
    """Test return dimension"""
    assert test_class_1.dimension == 1
    assert test_class_3.dimension == 3


@pytest.mark.filterwarnings("error")
def test_call_no_error_1(test_class_1):
    """Test valid inputs of call give no errors for 1D functions"""
    x_1d = numpy.linspace(test_class_1.lower_bounds, test_class_1.upper_bounds)
    for x in x_1d:
        test_class_1(x)


@pytest.mark.filterwarnings("error")
def test_call_no_error_3D(test_class_3):
    """Test valid inputs of call give no errors for >1D functions"""
    x_3d = numpy.linspace(test_class_3.lower_bounds, test_class_3.upper_bounds)
    for x in x_3d:
        test_class_3(x)


def test_call_error_1D(test_class_1):
    """Test if there is an error for inputs out of bounds for 1D functions"""
    with pytest.raises(ValueError):
        test_class_1(0)
    with pytest.raises(ValueError):
        test_class_1(20)


def test_call_error_3D(test_class_3):
    """Test if there is an error for inputs out of bounds for >1D functions"""
    with pytest.raises(ValueError):
        test_class_3([-2, -2, 12])
    with pytest.raises(ValueError):
        test_class_3([11, -2, 12])
    with pytest.raises(ValueError):
        test_class_3([5, -10, 12])
    with pytest.raises(ValueError):
        test_class_3([5, 4, 12])
    with pytest.raises(ValueError):
        test_class_3([5, -2, 6])
    with pytest.raises(ValueError):
        test_class_3([5, -2, 24])


def test_call_shape_1D(test_class_1):
    """Test that different sizes of inputs give excepted shape"""
    x = 0.5
    x_2array = numpy.array([0.5, 0.6, 0.7, 0.8], ndmin=2)
    x_3array = numpy.array([[0.5, 0.6, 0.7], [0.44, 0.55, 0.66]])
    assert numpy.ndim(test_class_1(x)) == 0
    assert len(test_class_1(x_2array)) == 4
    assert x_3array.shape == test_class_1(x_3array).shape


def test_call_shape_3D(test_class_3):
    """Test that different sizes of inputs give excepted shape"""
    x = [1, -2, 11.1]
    x_2array = numpy.array(
        [[1, -2, 11.1], [2, -2.5, 11.2], [3, -2.6, 11.3], [4, -2.7, 11.4]], ndmin=2
    )
    x_3array = numpy.array(
        [
            [[1, -2, 11.1], [2, -2.5, 11.2], [3, -2.6, 11.3], [4, -2.7, 11.4]],
            [[5, -3, 12.1], [6, -3.5, 12.2], [7, -3.6, 12.3], [8, -3.7, 12.4]],
        ]
    )
    assert numpy.ndim(test_class_3(x)) == 0
    assert len(test_class_3(x_2array)) == 4
    assert x_3array.shape[:-1] == test_class_3(x_3array).shape


def test_class_dimension_error(test_class_3):
    """Test that invalid input size gives error"""
    x = [1, 11.1]
    x_2array = numpy.array(
        [[1, -2, 11.1, 4], [2, -2.5, 11.2, 4], [3, -2.6, 11.3, 4], [4, -2.7, 11.4, 4]],
        ndmin=2,
    )
    x_3array = numpy.array(
        [
            [[1, 11.1], [2, -2.5], [-2.6, 11.3], [4, -2.7]],
            [[5, 12.1], [6, 12.2], [-3.6, 12.3], [8, -3.7]],
        ]
    )
    with pytest.raises(IndexError):
        test_class_3(x)
    with pytest.raises(IndexError):
        test_class_3(x_2array)
    with pytest.raises(IndexError):
        test_class_3(x_3array)


def test_class_pointset_dimension_error(test_class_1, test_class_3):
    """Test error wrong dimension of UnidimensionalPointSet and MultidimensionalPointSet"""
    x_1D = smolyay.samples.ClenshawCurtisPointSet([1, 5], 6)
    x_2D = smolyay.samples.HaltonRandomPointSet([[1, 5], [1, 5]], 15, 1)
    with pytest.raises(IndexError):
        test_class_3(x_1D)
    with pytest.raises(IndexError):
        test_class_3(x_2D)
    with pytest.raises(IndexError):
        test_class_1(x_2D)


def test_class_pointset_domain_error(test_class_1, test_class_3):
    """Test error wrong domain of UnidimensionalPointSet and MultidimensionalPointSet"""
    x_1D = smolyay.samples.ClenshawCurtisPointSet([0, 5], 6)
    x_multi_1D = smolyay.samples.HaltonRandomPointSet([0, 5], 15, 1)
    x_3D = smolyay.samples.HaltonRandomPointSet([[1, 5], [1, 5], [12, 13]], 15, 1)
    with pytest.raises(ValueError):
        test_class_1(x_1D)
    with pytest.raises(ValueError):
        test_class_1(x_multi_1D)
    with pytest.raises(ValueError):
        test_class_3(x_3D)


@pytest.mark.filterwarnings("error")
def test_call_no_error_multi_input_1D(test_class_1):
    """Test multiple inputs of call give no errors for 1D functions"""
    x_1d = numpy.linspace(test_class_1.lower_bounds, test_class_1.upper_bounds)
    y_1d_multi = test_class_1(x_1d)
    y_1d = [test_class_1(x) for x in x_1d]
    assert len(y_1d_multi) == len(x_1d)
    assert numpy.array_equiv(y_1d, y_1d_multi)


@pytest.mark.filterwarnings("error")
def test_call_no_error_multi_input_3D(test_class_3):
    """Test multiple inputs of call give no errors for >1D functions"""
    x_3d = numpy.linspace(test_class_3.lower_bounds, test_class_3.upper_bounds)
    y_3d_multi = test_class_3(x_3d)
    y_3d = [test_class_3(x) for x in x_3d]
    assert len(y_3d_multi) == len(x_3d)
    assert numpy.array_equiv(y_3d, y_3d_multi)


def test_call_error_1D_multi_input(test_class_1):
    """Test if there is an error for inputs out of bounds for 1D functions"""
    with pytest.raises(ValueError):
        test_class_1([0.5, 0.6, 1.7, 0, 9.2])
    with pytest.raises(ValueError):
        test_class_1([20, 5, 6, 2])


def test_call_error_3D_multi_input(test_class_3):
    """Test if there is an error for inputs out of bounds for >1D functions"""
    with pytest.raises(ValueError):
        test_class_3([[-2, -2, 12], [3, -1, 11.5], [4, -4, 11.1], [1, -4.4, 12.7]])
    with pytest.raises(ValueError):
        test_class_3([[1, -1.1, 11.3], [9, -1.2, 11.4], [11, -2, 12]])
    with pytest.raises(ValueError):
        test_class_3([[1, -1, 11.5], [5, -10, 12], [2, -2, 13], [3, -3, 12]])
    with pytest.raises(ValueError):
        test_class_3([[5, 4, 12], [3, 6, 12]])
    with pytest.raises(ValueError):
        test_class_3([[0, -1, 11], [5, -2, 6]])
    with pytest.raises(ValueError):
        test_class_3([[5, -2, 24], [1, -2, 11]])


@pytest.mark.parametrize("fun", functions)
def test_functions_domain_match_dimension(fun):
    """Test that all domains are the correct shape"""
    assert len(fun.domain) == fun.dimension
    for bound in fun.domain:
        assert len(bound) == 2


@pytest.mark.parametrize("fun", functions)
def test_functions_good_bounds(fun):
    """Test if all lower bounds are lower than upper bounds"""
    for bound in fun.domain:
        assert bound[0] < bound[1]


@pytest.mark.parametrize("fun", functions)
@pytest.mark.filterwarnings("error")
def test_functions_call(fun):
    """Test all functions do not raise error or warning for inputs in bounds"""
    x_list = numpy.linspace(fun.lower_bounds, fun.upper_bounds)
    y_call = [fun(i) for i in x_list]
    y_call_multi = fun(x_list)
    assert numpy.array_equiv(y_call, y_call_multi)


@pytest.mark.parametrize("fun", functions)
def test_functions_call_error_lower(fun):
    """Test all functions error for inputs below bounds"""
    below_bounds = numpy.add(fun.lower_bounds, -5)
    with pytest.raises(ValueError):
        fun(below_bounds)


@pytest.mark.parametrize("fun", functions)
def test_functions_call_error_above(fun):
    """Test all functions error for inputs above bounds"""
    above_bounds = numpy.add(fun.upper_bounds, 5)
    with pytest.raises(ValueError):
        fun(above_bounds)


@pytest.mark.parametrize("fun", functions)
def test_functions_call_shape(fun):
    """Test all functions give the correct shape"""
    x_list = numpy.zeros((2, 50, fun.dimension))
    mid_point = numpy.divide(numpy.add(fun.lower_bounds, fun.upper_bounds), 2)
    x_list[0] = numpy.linspace(fun.lower_bounds, mid_point, 50)
    x_list[1] = numpy.linspace(mid_point, fun.upper_bounds, 50)
    assert fun(x_list).shape == (2, 50)
