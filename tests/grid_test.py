import numpy
import pytest
import scipy.stats.qmc

import smolyay


@pytest.mark.parametrize(
    "random_point_set",
    [
        smolyay.samples.UniformRandomPointSet,
        smolyay.samples.LatinHypercubeRandomPointSet,
        smolyay.samples.HaltonRandomPointSet,
        smolyay.samples.SobolRandomPointSet,
    ],
    ids=["Uniform", "Latin", "Halton", "Sobol"],
)
def test_random_initalize(random_point_set):
    """Test that the random point set initializes correctly"""
    f = random_point_set([[-10, 10], [0, 2]], 64, 1234)
    assert numpy.array_equal(f.domain, [[-10, 10], [0, 2]])
    assert f.num_dimensions == 2
    assert f.num_points == 64
    assert len(f) == 64
    assert isinstance(f.num_points, int)
    assert f.seed == 1234
    assert isinstance(f.seed, int)

    f.domain = [-10, 10]
    assert numpy.array_equal(f.domain, [[-10, 10]])
    f.num_points = 128.0
    assert f.num_points == 128
    assert isinstance(f.num_points, int)
    f.seed = 40.0
    assert f.seed == 40
    assert isinstance(f.seed, int)


@pytest.mark.parametrize(
    "qmc_point_set",
    [
        smolyay.samples.LatinHypercubeRandomPointSet,
        smolyay.samples.HaltonRandomPointSet,
        smolyay.samples.SobolRandomPointSet,
    ],
    ids=["Latin", "Halton", "Sobol"],
)
def test_random_qmc_initalize(qmc_point_set):
    """Test that the random point set initializes correctly"""
    f = qmc_point_set([[-10, 20]], 64, 5678, True, "random-cd")
    assert numpy.array_equal(f.domain, [[-10, 20]])
    assert f.num_dimensions == 1
    assert f.num_points == 64
    assert f.seed == 5678
    assert isinstance(f.scramble, bool)
    assert f.scramble == True
    assert f.optimization == "random-cd"

    f.scramble = 0
    assert isinstance(f.scramble, bool)
    assert f.scramble is False
    f.optimization = "lloyd"
    assert f.optimization == "lloyd"
    f.optimization = None
    assert f.optimization is None


def test_random_latin_initialize():
    """That the LatinHypercubeRandomPointSet initializes correctly"""
    f = smolyay.samples.LatinHypercubeRandomPointSet(
        [[-10, 20]], 64, 5678, True, "random-cd", 1
    )
    assert numpy.array_equal(f.domain, [[-10, 20]])
    assert f.num_dimensions == 1
    assert f.num_points == 64
    assert f.seed == 5678
    assert isinstance(f.scramble, bool)
    assert f.scramble is True
    assert f.optimization == "random-cd"
    assert f.strength == 1
    assert isinstance(f.strength, int)
    f.strength = 2
    assert f.strength == 2
    assert isinstance(f.strength, int)


def test_random_sobol_initialize():
    """That the SobolRandomPointSet initializes correctly"""
    f = smolyay.samples.SobolRandomPointSet(
        [[-10, 20]], 64, 5678, True, "random-cd", 30
    )
    assert numpy.array_equal(f.domain, [[-10, 20]])
    assert f.num_dimensions == 1
    assert f.num_points == 64
    assert f.seed == 5678
    assert isinstance(f.scramble, bool)
    assert f.scramble == True
    assert f.optimization == "random-cd"
    assert f.bits == 30
    assert isinstance(f.bits, int)
    f.bits = 42
    assert f.bits == 42
    assert isinstance(f.bits, int)


@pytest.mark.parametrize(
    "product_point_set",
    [
        smolyay.samples.TensorProductPointSet,
        smolyay.samples.SmolyakSparseProductPointSet,
    ],
    ids=["Tensor", "Smolyak"],
)
def test_product_initialize(product_point_set):
    point_sets = [
        smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3),
        smolyay.samples.NestedClenshawCurtisPointSet([-2, 2], 3),
    ]
    f = product_point_set(point_sets)
    assert f.point_sets == point_sets
    assert numpy.array_equal(f.domain, [[-1, 1], [-2, 2]])
    f.domain = [[-10, 10], [-5, 3]]
    assert numpy.array_equal(f.domain, [[-10, 10], [-5, 3]])
    assert numpy.array_equal(f.point_sets[0].domain, [-10, 10])
    assert numpy.array_equal(f.point_sets[1].domain, [-5, 3])


@pytest.mark.parametrize(
    "random_point_set",
    [
        smolyay.samples.UniformRandomPointSet,
        smolyay.samples.LatinHypercubeRandomPointSet,
        smolyay.samples.HaltonRandomPointSet,
        smolyay.samples.SobolRandomPointSet,
    ],
    ids=["Uniform", "Latin", "Halton", "Sobol"],
)
def test_random_domain_error(random_point_set):
    """Test that an exception is given if the domain is invalid"""
    # reverse domain
    f = random_point_set([[10, -10]], 64, 1234)
    assert numpy.array_equal(f.domain, [[-10, 10]])
    f = random_point_set([[-10, 10]], 64, 1234)
    f.domain = [[5, -10], [9, -9]]
    assert numpy.array_equal(f.domain, [[-10, 5], [-9, 9]])
    # invalid domain
    with pytest.raises(TypeError):
        random_point_set([[-10, 10, 11], [0, 2, 11]], 64, 1234)
    with pytest.raises(TypeError):
        random_point_set([[[-10, 10]]], 64, 1234)
    with pytest.raises(ValueError):
        random_point_set([[10, 10], [5, 10], [9, 12]], 64, 1234)
    with pytest.raises(TypeError):
        f = random_point_set([[-10, 10], [-10, 10]], 64, 1234)
        f.domain = [[-10, 10, 11], [0, 2, 11]]
    with pytest.raises(TypeError):
        f = random_point_set([[-10, 10], [-10, 10]], 64, 1234)
        f.domain = [[[-10, 10]]]
    with pytest.raises(ValueError):
        f = random_point_set([[-10, 10], [-10, 10]], 64, 1234)
        f.domain = [[10, 10], [5, 10], [9, 12]]


def test_random_sobol_error():
    """Test classmethod error using sobol if number of points not a power of 2"""
    # power of 2 error
    with pytest.raises(ValueError):
        smolyay.samples.SobolRandomPointSet([[0, 2]], 70, 1234)
    with pytest.raises(ValueError):
        f = smolyay.samples.SobolRandomPointSet([[0, 2]], 64, 1234)
        f.num_points = 70
    # bits limits error
    with pytest.raises(ValueError):
        smolyay.samples.SobolRandomPointSet([[0, 2]], 70, 1234, bits=72)
    with pytest.raises(ValueError):
        f = smolyay.samples.SobolRandomPointSet([[0, 2]], 64, 1234)
        f.bits = 72
    with pytest.raises(ValueError):
        smolyay.samples.SobolRandomPointSet([[0, 2]], 70, 1234, bits=-4)
    with pytest.raises(ValueError):
        f = smolyay.samples.SobolRandomPointSet([[0, 2]], 64, 1234)
        f.bits = -4
    # 2**bits < num_points
    with pytest.raises(ValueError):
        smolyay.samples.SobolRandomPointSet([[0, 2]], 2048, 1234, bits=5)
    with pytest.raises(ValueError):
        f = smolyay.samples.SobolRandomPointSet([[0, 2]], 16, 1234, bits=5)
        f.num_points = 2048
    with pytest.raises(ValueError):
        f = smolyay.samples.SobolRandomPointSet([[0, 2]], 16, 1234, bits=5)
        f.bits = 2


@pytest.mark.parametrize(
    "product_point_set",
    [
        smolyay.samples.TensorProductPointSet,
        smolyay.samples.SmolyakSparseProductPointSet,
    ],
    ids=["Tensor", "Smolyak"],
)
def test_product_domain_error(product_point_set):
    point_sets = [
        smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3),
        smolyay.samples.NestedClenshawCurtisPointSet([-2, 2], 3),
    ]
    f = product_point_set(point_sets)
    with pytest.raises(TypeError):
        f.domain = [[-10, 10, 11], [0, 2, 11]]
    with pytest.raises(TypeError):
        f.domain = [[[-10, 10], [-10, 10]]]
    with pytest.raises(ValueError):
        f.domain = [[10, 10], [5, 10]]
    with pytest.raises(IndexError):
        f.domain = [[-10, 10]]
    with pytest.raises(IndexError):
        f.domain = [[-10, 10], [-10, 10], [-10, 10]]


@pytest.mark.parametrize(
    "random_point_set,domain,num_points,seed,answer",
    [
        (
            smolyay.samples.HaltonRandomPointSet,
            [[-3, 5], [6, 9]],
            5,
            4,
            scipy.stats.qmc.scale(
                scipy.stats.qmc.Halton(2, seed=4).random(n=5), [-3, 6], [5, 9]
            ),
        ),
        (
            smolyay.samples.LatinHypercubeRandomPointSet,
            [[-10, 10], [0, 2], [0, 9]],
            70,
            1234,
            scipy.stats.qmc.scale(
                scipy.stats.qmc.LatinHypercube(3, seed=1234).random(n=70),
                [-10, 0, 0],
                [10, 2, 9],
            ),
        ),
        (
            smolyay.samples.SobolRandomPointSet,
            [[-10, 10], [0, 9]],
            32,
            1234,
            scipy.stats.qmc.scale(
                scipy.stats.qmc.Sobol(2, seed=1234).random(n=32), [-10, 0], [10, 9]
            ),
        ),
        (
            smolyay.samples.UniformRandomPointSet,
            [[-10, 10], [0, 9], [0, 1], [0, 1]],
            100,
            1234,
            scipy.stats.qmc.scale(
                numpy.random.default_rng(seed=1234).uniform(size=(100, 4)),
                [-10, 0, 0, 0],
                [10, 9, 1, 1],
            ),
        ),
    ],
    ids=["Halton", "LatinHypercube", "Sobol", "Uniform"],
)
def test_random_points(random_point_set, domain, num_points, seed, answer):
    """Test each method for generating random points"""
    points = random_point_set(domain, num_points, seed).points
    assert numpy.array_equal(points, answer)


def test_generate_tensor_points():
    """Test the generate_tensor_combinations for a series with multiple sets."""
    point_sets = [
        smolyay.samples.TrigonometricPointSet([-1, 1], 1),
        smolyay.samples.ClenshawCurtisPointSet([-1, 1], 1),
    ]
    answer = [[-1, -1], [-1, 1], [-1 / 3, -1], [-1 / 3, 1], [1 / 3, -1], [1 / 3, 1]]
    f = smolyay.samples.TensorProductPointSet(point_sets)
    assert numpy.allclose(f.points, answer)


def test_generate_tensor_points_from_arrays():
    """Test the generate_tensor_combinations for using numpy arrays."""
    point_sets = [numpy.array([9, 8, 7]), numpy.array([1, 2])]
    answer = [[9, 1], [9, 2], [8, 1], [8, 2], [7, 1], [7, 2]]
    f = smolyay.samples.TensorProductPointSet(point_sets)
    assert numpy.array_equal(f.points, answer)


def test_generate_smolyak_points():
    """Test the generate_tensor_combinations for a series with one set."""
    point_sets = [
        smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3),
        smolyay.samples.NestedClenshawCurtisPointSet([-2, 2], 3),
    ]
    answer = [
        [0.0, 0.0],
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, -2.0],
        [0.0, 2.0],
        [-0.70710678, 0.0],
        [0.70710678, 0.0],
        [-1.0, -2.0],
        [-1.0, 2.0],
        [1.0, -2.0],
        [1.0, 2.0],
        [0.0, -1.41421356],
        [0.0, 1.41421356],
    ]
    f = smolyay.samples.SmolyakSparseProductPointSet(point_sets)
    assert numpy.allclose(f.points, answer)


def test_generate_smolyak_points_different_levels():
    """Test the generate_tensor_combinations for a series with one set."""
    point_sets = [
        smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3),
        smolyay.samples.NestedClenshawCurtisPointSet([-2, 2], 2),
    ]
    answer = [
        [0.0, 0.0],
        [-1.0, 0.0],
        [1.0, 0.0],
        [0.0, -2.0],
        [0.0, 2.0],
        [-0.70710678, 0.0],
        [0.70710678, 0.0],
        [-1.0, -2.0],
        [-1.0, 2.0],
        [1.0, -2.0],
        [1.0, 2.0],
    ]
    f = smolyay.samples.SmolyakSparseProductPointSet(point_sets)
    assert numpy.allclose(f.points, answer)


@pytest.mark.parametrize(
    "point_set",
    [
        smolyay.samples.TensorProductPointSet(
            [
                smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3),
                smolyay.samples.NestedClenshawCurtisPointSet([-2, 2], 2),
            ]
        ),
        smolyay.samples.SmolyakSparseProductPointSet(
            [
                smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3),
                smolyay.samples.NestedClenshawCurtisPointSet([-2, 2], 2),
            ]
        ),
        smolyay.samples.HaltonRandomPointSet(
            [[-3, 5], [6, 9]],
            5,
            4,
        ),
        smolyay.samples.LatinHypercubeRandomPointSet(
            [[-10, 10], [0, 2], [0, 9]],
            70,
            1234,
        ),
        smolyay.samples.SobolRandomPointSet(
            [[-10, 10], [0, 9]],
            32,
            1234,
        ),
        smolyay.samples.UniformRandomPointSet(
            [[-10, 10], [0, 9], [0, 1], [0, 1]], 100, 1234
        ),
    ],
    ids=["Tensor", "Smolyak", "Halton", "LatinHypercube", "Sobol", "Uniform"],
)
def test_array_like_qualities(point_set):
    assert numpy.array_equal(point_set + 1, point_set.points + 1)
    assert numpy.array_equal(point_set - 1, point_set.points - 1)
    assert numpy.array_equal(point_set * 2, point_set.points * 2)
    assert numpy.array_equal(point_set / 2, point_set.points / 2)
    assert numpy.array_equal(point_set == 1, point_set.points == 1)
    assert numpy.array_equal(point_set != 1, point_set.points != 1)
    assert numpy.array_equal(point_set >= 1, point_set.points >= 1)
    assert numpy.array_equal(point_set <= 1, point_set.points <= 1)
    assert numpy.array_equal(point_set > 1, point_set.points > 1)
    assert numpy.array_equal(point_set < 1, point_set.points < 1)


def test_generate_compositions_include_zero_true():
    """Test the generate compositions function if include_zero is true."""
    composition_expected = [[6, 0], [5, 1], [4, 2], [3, 3], [2, 4], [1, 5], [0, 6]]
    composition_obtained = []
    composition_obtained = list(
        smolyay.samples._generate_compositions(6, 2, include_zero=True)
    )
    assert composition_obtained == composition_expected


def test_generate_compositions_include_zero_false():
    """Test the generate compositions function if include_zero is false."""
    composition_expected = [[5, 1], [4, 2], [3, 3], [2, 4], [1, 5]]
    composition_obtained = list(
        smolyay.samples._generate_compositions(6, 2, include_zero=False)
    )
    assert composition_obtained == composition_expected


def test_generate_compositions_zero_false_error():
    """Test that generate compositions raises an error for invalid input."""
    with pytest.raises(ValueError):
        list(smolyay.samples._generate_compositions(6, 7, include_zero=False))
