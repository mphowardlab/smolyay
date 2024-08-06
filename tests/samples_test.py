import numpy
import pytest
import scipy.stats.qmc

import smolyay

sample_points_answers = [
    (smolyay.samples.ClenshawCurtisPointSet([-1, 1], 0), [0]),
    (smolyay.samples.ClenshawCurtisPointSet([-1, 1], 1), [-1, 1]),
    (smolyay.samples.ClenshawCurtisPointSet([-1, 1], 2), [-1, 0, 1]),
    (smolyay.samples.ClenshawCurtisPointSet([-1, 1], 3), [-1, -0.5, 0.5, 1]),
    (
        smolyay.samples.ClenshawCurtisPointSet([-1, 1], 8),
        [
            -1,
            -numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
            -1 / numpy.sqrt(2),
            -numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            0,
            numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            1 / numpy.sqrt(2),
            numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
            1,
        ],
    ),
    (smolyay.samples.TrigonometricPointSet([0, 2 * numpy.pi], 0), [0]),
    (
        smolyay.samples.TrigonometricPointSet([0, 2 * numpy.pi], 1),
        [0, 2 * numpy.pi / 3, 4 * numpy.pi / 3],
    ),
    (
        smolyay.samples.TrigonometricPointSet([0, 2 * numpy.pi], 4),
        2 * numpy.pi * numpy.linspace(0, 8 / 9, 9),
    ),
    (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 1), [0]),
    (smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 2), [0, -1, 1]),
    (
        smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3),
        [0, -1.0, 1.0, -1 / numpy.sqrt(2), 1 / numpy.sqrt(2)],
    ),
    (
        smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 4),
        [
            0,
            -1.0,
            1.0,
            -1 / numpy.sqrt(2),
            1 / numpy.sqrt(2),
            -numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
            -numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
        ],
    ),
    (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 1), [0]),
    (smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 2), [0, -1, 1]),
    (
        smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 3),
        [0, -1.0, 1.0, -1 / numpy.sqrt(2), 1 / numpy.sqrt(2)],
    ),
    (
        smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 4),
        [
            0,
            -1.0,
            1.0,
            -1 / numpy.sqrt(2),
            1 / numpy.sqrt(2),
            -numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
            -numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
        ],
    ),
    (
        smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 5),
        [
            0,
            -1.0,
            1.0,
            -1 / numpy.sqrt(2),
            1 / numpy.sqrt(2),
            -numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
            -numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
        ],
    ),
    (
        smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 6),
        [
            0,
            -1.0,
            1.0,
            -1 / numpy.sqrt(2),
            1 / numpy.sqrt(2),
            -numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
            -numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
            numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
            -numpy.sqrt(2 + numpy.sqrt(2 + numpy.sqrt(2))) / 2,
            -numpy.sqrt(2 + numpy.sqrt(2 - numpy.sqrt(2))) / 2,
            -numpy.sqrt(2 - numpy.sqrt(2 - numpy.sqrt(2))) / 2,
            -numpy.sqrt(2 - numpy.sqrt(2 + numpy.sqrt(2))) / 2,
            numpy.sqrt(2 - numpy.sqrt(2 + numpy.sqrt(2))) / 2,
            numpy.sqrt(2 - numpy.sqrt(2 - numpy.sqrt(2))) / 2,
            numpy.sqrt(2 + numpy.sqrt(2 - numpy.sqrt(2))) / 2,
            numpy.sqrt(2 + numpy.sqrt(2 + numpy.sqrt(2))) / 2,
        ],
    ),
    (smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 1), [0]),
    (
        smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 2),
        [0, 2 * numpy.pi / 3, 4 * numpy.pi / 3],
    ),
    (
        smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 3),
        2
        * numpy.pi
        * numpy.array(
            [
                0,
                1 / 3,
                2 / 3,
                1 / 9,
                2 / 9,
                4 / 9,
                5 / 9,
                7 / 9,
                8 / 9,
            ]
        ),
    ),
]

sample_points_ids = [
    "ClenshawCurtis [0]",
    "ClenshawCurtis [1]",
    "ClenshawCurtis [2]",
    "ClenshawCurtis [3]",
    "ClenshawCurtis [8]",
    "Trigonometric [0]",
    "Trigonometric [1]",
    "Trigonometric [4]",
    "NestedClenshawCurtis [0]",
    "NestedClenshawCurtis [1]",
    "NestedClenshawCurtis [2]",
    "NestedClenshawCurtis [3]",
    "SlowNestedClenshawCurtis [0]",
    "SlowNestedClenshawCurtis [1]",
    "SlowNestedClenshawCurtis [2]",
    "SlowNestedClenshawCurtis [3]",
    "SlowNestedClenshawCurtis [4]",
    "SlowNestedClenshawCurtis [5]",
    "NestedTrigonometric [0]",
    "NestedTrigonometric [1]",
    "NestedTrigonometric [2]",
]

nested_sample_ids = [
    "NestedClenshawCurtis [0]",
    "NestedClenshawCurtis [1]",
    "NestedClenshawCurtis [2]",
    "NestedClenshawCurtis [3]",
    "SlowNestedClenshawCurtis [0]",
    "SlowNestedClenshawCurtis [1]",
    "SlowNestedClenshawCurtis [2]",
    "SlowNestedClenshawCurtis [3]",
    "SlowNestedClenshawCurtis [4]",
    "SlowNestedClenshawCurtis [5]",
    "NestedTrigonometric [0]",
    "NestedTrigonometric [1]",
    "NestedTrigonometric [2]",
]


def test_initialize_clenshaw():
    """Test initialization and setters"""
    f = smolyay.samples.ClenshawCurtisPointSet([-2, 1], 3)
    assert numpy.array_equal(f.domain, [-2, 1])
    assert f.degree == 3
    assert isinstance(f.degree, int)
    f.degree = float(5)
    assert f.degree == 5
    assert isinstance(f.degree, int)


def test_degree_error():
    """Test degree error given invalid degree"""
    with pytest.raises(ValueError):
        smolyay.samples.ClenshawCurtisPointSet([-2, 1], -7)
    f = smolyay.samples.ClenshawCurtisPointSet([-2, 1], 3)
    with pytest.raises(ValueError):
        f.degree = -5


def test_initialize_trig():
    """Test initialization and setters"""
    f = smolyay.samples.TrigonometricPointSet([0, 4 * numpy.pi], 3)
    assert numpy.array_equal(f.domain, [0, 4 * numpy.pi])
    assert f.frequency == 3
    assert isinstance(f.frequency, int)


def test_frequency_error():
    """Test frequency error given invalid frequency"""
    with pytest.raises(ValueError):
        smolyay.samples.TrigonometricPointSet([-2, 1], -4)
    f = smolyay.samples.TrigonometricPointSet([-2, 1], 3)
    with pytest.raises(ValueError):
        f.frequency = -5


@pytest.mark.parametrize(
    "nested_samples",
    [
        smolyay.samples.NestedClenshawCurtisPointSet,
        smolyay.samples.SlowNestedClenshawCurtisPointSet,
        smolyay.samples.NestedTrigonometricPointSet,
    ],
    ids=[
        "NestedClenshawCurtis",
        "SlowNestedClenshawCurtis",
        "NestedTrigonometric",
    ],
)
def test_initialize_nested(nested_samples):
    """Test initialization and setters"""
    f = nested_samples([-10, 10], 4)
    assert numpy.array_equal(f.domain, [-10, 10])
    assert f.num_levels == 4
    assert isinstance(f.num_levels, int)
    f.num_levels = float(5)
    assert f.num_levels == 5
    assert isinstance(f.num_levels, int)


@pytest.mark.parametrize(
    "samples,set_args",
    [
        (smolyay.samples.ClenshawCurtisPointSet, {"degree": 4}),
        (smolyay.samples.TrigonometricPointSet, {"frequency": 4}),
        (smolyay.samples.NestedClenshawCurtisPointSet, {"num_levels": 4}),
        (smolyay.samples.SlowNestedClenshawCurtisPointSet, {"num_levels": 4}),
        (smolyay.samples.NestedTrigonometricPointSet, {"num_levels": 4}),
    ],
    ids=[
        "ClenshawCurtis",
        "Trigonometric",
        "NestedClenshawCurtis",
        "SlowNestedClenshawCurtis",
        "NestedTrigonometric",
    ],
)
def test_domain_error(samples, set_args):
    """Test error given invalid domain and that reversed domains swap"""
    # reverse domain
    f = samples([10, -10], **set_args)
    assert numpy.array_equal(f.domain, [-10, 10])
    f = samples([-10, 10], **set_args)
    f.domain = [5, -10]
    assert numpy.array_equal(f.domain, [-10, 5])
    # invalid domain
    with pytest.raises(TypeError):
        samples([-10, 10, 20], **set_args)
    with pytest.raises(TypeError):
        samples([[-10, 10], [-10, 10]], **set_args)
    with pytest.raises(ValueError):
        samples([10, 10], **set_args)
    with pytest.raises(TypeError):
        f = samples([-10, 10], **set_args)
        f.domain = [[-10, 10], [-10, 10]]
    with pytest.raises(TypeError):
        f = samples([-10, 10], **set_args)
        f.domain = [-10, 10, 20]
    with pytest.raises(ValueError):
        f = samples([-10, 10], **set_args)
        f.domain = [10, 10]


@pytest.mark.parametrize(
    "nested_samples",
    [
        smolyay.samples.NestedClenshawCurtisPointSet,
        smolyay.samples.SlowNestedClenshawCurtisPointSet,
        smolyay.samples.NestedTrigonometricPointSet,
    ],
    ids=[
        "NestedClenshawCurtis",
        "SlowNestedClenshawCurtis",
        "NestedTrigonometric",
    ],
)
def test_num_levels_error(nested_samples):
    """Test error given invalid num_levels"""
    with pytest.raises(ValueError):
        f = nested_samples([-10, 10], 0)
    f = nested_samples([-10, 10], 2)
    with pytest.raises(ValueError):
        f.num_levels = 0


@pytest.mark.parametrize("samples,points", sample_points_answers, ids=sample_points_ids)
def test_generate_points(samples, points):
    """Test the points of initialized UnidimensionalPointSet"""
    assert len(samples) == len(points)
    assert numpy.allclose(samples.points, points, atol=1e-10)

    # test for different domain and that points update after changing domain
    new_points = numpy.array(points) * 5 + 10
    new_domain = samples.domain * 5 + 10
    samples.domain = new_domain
    assert numpy.array_equal(samples.domain, new_domain)
    assert numpy.allclose(samples.points, new_points, atol=1e-10)
    assert samples[0] == pytest.approx(new_points[0])
    assert numpy.allclose(samples[:], new_points, atol=1e-10)
    assert numpy.allclose(list(samples), new_points, atol=1e-10)


@pytest.mark.parametrize(
    "nested_samples,num_per_level,start_level,end_level",
    [
        (
            smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 1),
            [1],
            [0],
            [1],
        ),
        (
            smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 2),
            [1, 2],
            [0, 1],
            [1, 3],
        ),
        (
            smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 3),
            [1, 2, 2],
            [0, 1, 3],
            [1, 3, 5],
        ),
        (
            smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], 4),
            [1, 2, 2, 4],
            [0, 1, 3, 5],
            [1, 3, 5, 9],
        ),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 1),
            [1],
            [0],
            [1],
        ),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 2),
            [1, 2],
            [0, 1],
            [1, 3],
        ),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 3),
            [1, 2, 2],
            [0, 1, 3],
            [1, 3, 5],
        ),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 4),
            [1, 2, 2, 4],
            [0, 1, 3, 5],
            [1, 3, 5, 9],
        ),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 5),
            [1, 2, 2, 4, 0],
            [0, 1, 3, 5, 9],
            [1, 3, 5, 9, 9],
        ),
        (
            smolyay.samples.SlowNestedClenshawCurtisPointSet([-1, 1], 6),
            [1, 2, 2, 4, 0, 8],
            [0, 1, 3, 5, 9, 9],
            [1, 3, 5, 9, 9, 17],
        ),
        (
            smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 1),
            [1],
            [0],
            [1],
        ),
        (
            smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 2),
            [1, 2],
            [0, 1],
            [1, 3],
        ),
        (
            smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], 3),
            [1, 2, 6],
            [0, 1, 3],
            [1, 3, 9],
        ),
    ],
    ids=nested_sample_ids,
)
def test_nested_levels(nested_samples, num_per_level, start_level, end_level):
    """Test number of points per level, start level indexes, and end level indexes"""
    assert numpy.array_equal(nested_samples.num_per_level, num_per_level)
    assert numpy.array_equal(nested_samples.start_level, start_level)
    assert numpy.array_equal(nested_samples.end_level, end_level)
    for level, (start, end) in enumerate(
        zip(nested_samples.start_level, nested_samples.end_level)
    ):
        assert numpy.allclose(nested_samples.level(level), nested_samples[start:end])


# Test MultidimensionalPointSets
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
    """Test random point sets initialization and shared setters"""
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
    """Test Monte Carlo point sets initialization and shared setters"""
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
    """Test that an exception is given if the domain is invalid"""
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
    """Test TensorProductPointSet using a list of sets"""
    point_sets = [
        smolyay.samples.TrigonometricPointSet([-1, 1], 1),
        smolyay.samples.ClenshawCurtisPointSet([-1, 1], 1),
    ]
    answer = [[-1, -1], [-1, 1], [-1 / 3, -1], [-1 / 3, 1], [1 / 3, -1], [1 / 3, 1]]
    f = smolyay.samples.TensorProductPointSet(point_sets)
    assert numpy.allclose(f.points, answer)


def test_generate_tensor_points_from_arrays():
    """Test TensorProductPointSet using different sized numpy arrays"""
    point_sets = [numpy.array([9, 8, 7]), numpy.array([1, 2])]
    answer = [[9, 1], [9, 2], [8, 1], [8, 2], [7, 1], [7, 2]]
    f = smolyay.samples.TensorProductPointSet(point_sets)
    assert len(f) == 6
    assert numpy.array_equal(f.points, answer)


def test_generate_smolyak_points():
    """Test the SmolyakSparseProductPointSet using a list of sets"""
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
    assert len(f) == 13
    assert numpy.allclose(f.points, answer)


def test_generate_smolyak_points_different_levels():
    """Test SmolyakSparseProductPointSet using sets with different levels"""
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
    assert len(f) == 11
    assert numpy.allclose(f.points, answer)


def test_generate_compositions_include_zero_true():
    """Test the generate compositions function if include_zero is true"""
    composition_expected = [[6, 0], [5, 1], [4, 2], [3, 3], [2, 4], [1, 5], [0, 6]]
    composition_obtained = []
    composition_obtained = list(
        smolyay.samples._generate_compositions(6, 2, include_zero=True)
    )
    assert composition_obtained == composition_expected


def test_generate_compositions_include_zero_false():
    """Test the generate compositions function if include_zero is false"""
    composition_expected = [[5, 1], [4, 2], [3, 3], [2, 4], [1, 5]]
    composition_obtained = list(
        smolyay.samples._generate_compositions(6, 2, include_zero=False)
    )
    assert composition_obtained == composition_expected


def test_generate_compositions_zero_false_error():
    """Test that generate compositions raises an error for invalid input"""
    with pytest.raises(ValueError):
        list(smolyay.samples._generate_compositions(6, 7, include_zero=False))
