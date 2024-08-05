import pytest

import numpy

import smolyay


basis_call_answer_key = [
    (
        smolyay.basis.ChebyshevFirstKind(0),
        {0.5: 1, 1: 1, -1: 1, -0.25: 1, -0.5: 1},
    ),
    (
        smolyay.basis.ChebyshevFirstKind(1),
        {0.5: 0.5, 1: 1, -1: -1, -0.25: -0.25, -0.5: -0.5},
    ),
    (
        smolyay.basis.ChebyshevFirstKind(2),
        {0.5: -0.5, 1: 1, -1: 1, -0.25: -0.875, -0.5: -0.5},
    ),
    (
        smolyay.basis.ChebyshevSecondKind(0),
        {0.5: 1, 1: 1, -1: 1, -0.25: 1, -0.5: 1},
    ),
    (
        smolyay.basis.ChebyshevSecondKind(1),
        {0.5: 1, 1: 2, -1: -2, -0.25: -0.5, -0.5: -1},
    ),
    (
        smolyay.basis.ChebyshevSecondKind(2),
        {0.5: 0, 1: 3, -1: 3, -0.25: -0.75, -0.5: 0},
    ),
    (
        smolyay.basis.Trigonometric(0),
        {
            0: 1,
            numpy.pi / 3: 1,
            3 * numpy.pi / 2: 1,
            numpy.pi / 6: 1,
            2 * numpy.pi: 1,
        },
    ),
    (
        smolyay.basis.Trigonometric(1),
        {
            0: 1,
            numpy.pi / 3: numpy.exp(numpy.pi / 3 * 1j),
            3 * numpy.pi / 2: numpy.exp(3 * numpy.pi / 2 * 1j),
            numpy.pi / 6: numpy.exp(numpy.pi / 6 * 1j),
            2 * numpy.pi: numpy.exp(2 * numpy.pi * 1j),
        },
    ),
    (
        smolyay.basis.Trigonometric(-1),
        {
            0: 1,
            numpy.pi / 3: numpy.exp(numpy.pi / 3 * 1j * -1),
            3 * numpy.pi / 2: numpy.exp(3 * numpy.pi / 2 * 1j * -1),
            numpy.pi / 6: numpy.exp(numpy.pi / 6 * 1j * -1),
            2 * numpy.pi: numpy.exp(2 * numpy.pi * 1j * -1),
        },
    ),
]


basis_derivative_answer_key = [
    (
        smolyay.basis.ChebyshevFirstKind(0),
        {0.5: 0, 1: 0, -1: 0, -0.25: 0, -0.5: 0},
    ),
    (
        smolyay.basis.ChebyshevFirstKind(1),
        {0.5: 1, 1: 1, -1: 1, -0.25: 1, -0.5: 1},
    ),
    (
        smolyay.basis.ChebyshevFirstKind(2),
        {0.5: 2, 1: 4, -1: -4, -0.25: -1, -0.5: -2},
    ),
    (
        smolyay.basis.ChebyshevSecondKind(0),
        {0.5: 0, 1: 0, -1: 0, -0.25: 0, -0.5: 0},
    ),
    (
        smolyay.basis.ChebyshevSecondKind(1),
        {0.5: 2, 1: 2, -1: 2, -0.25: 2, -0.5: 2},
    ),
    (
        smolyay.basis.ChebyshevSecondKind(2),
        {0.5: 4, 1: 8, -1: -8, -0.25: -2, -0.5: -4},
    ),
    (
        smolyay.basis.Trigonometric(0),
        {
            0: 0,
            numpy.pi / 3: 0,
            3 * numpy.pi / 2: 0,
            numpy.pi / 6: 0,
            2 * numpy.pi: 0,
        },
    ),
    (
        smolyay.basis.Trigonometric(1),
        {
            0: 1j,
            numpy.pi / 3: 1j * numpy.exp(numpy.pi / 3 * 1j),
            3 * numpy.pi / 2: 1j * numpy.exp(3 * numpy.pi / 2 * 1j),
            numpy.pi / 6: 1j * numpy.exp(numpy.pi / 6 * 1j),
            2 * numpy.pi: 1j * numpy.exp(2 * numpy.pi * 1j),
        },
    ),
    (
        smolyay.basis.Trigonometric(-1),
        {
            0: -1j,
            numpy.pi / 3: -1j * numpy.exp(numpy.pi / 3 * 1j * -1),
            3 * numpy.pi / 2: -1j * numpy.exp(3 * numpy.pi / 2 * 1j * -1),
            numpy.pi / 6: -1j * numpy.exp(numpy.pi / 6 * 1j * -1),
            2 * numpy.pi: -1j * numpy.exp(2 * numpy.pi * 1j * -1),
        },
    ),
]

basis_nth_derivative_answer_key = [
    (
        2,
        smolyay.basis.ChebyshevFirstKind(0),
        {0.5: 0, 1: 0, -1: 0, -0.25: 0, -0.5: 0},
    ),
    (
        2,
        smolyay.basis.ChebyshevFirstKind(1),
        {0.5: 0, 1: 0, -1: 0, -0.25: 0, -0.5: 0},
    ),
    (
        2,
        smolyay.basis.ChebyshevFirstKind(2),
        {0.5: 4, 1: 4, -1: 4, -0.25: 4, -0.5: 4},
    ),
    (
        2,
        smolyay.basis.ChebyshevFirstKind(3),
        {0.5: 12, 1: 24, -1: -24, -0.25: -6, -0.5: -12},
    ),
    (
        2,
        smolyay.basis.Trigonometric(0),
        {
            0: 0,
            numpy.pi / 3: 0,
            3 * numpy.pi / 2: 0,
            numpy.pi / 6: 0,
            2 * numpy.pi: 0,
        },
    ),
    (
        2,
        smolyay.basis.Trigonometric(1),
        {
            0: -1,
            numpy.pi / 3: -1 * numpy.exp(numpy.pi / 3 * 1j),
            3 * numpy.pi / 2: -1 * numpy.exp(3 * numpy.pi / 2 * 1j),
            numpy.pi / 6: -1 * numpy.exp(numpy.pi / 6 * 1j),
            2 * numpy.pi: -1 * numpy.exp(2 * numpy.pi * 1j),
        },
    ),
    (
        2,
        smolyay.basis.Trigonometric(-1),
        {
            0: -1,
            numpy.pi / 3: -1 * numpy.exp(numpy.pi / 3 * 1j * -1),
            3 * numpy.pi / 2: -1 * numpy.exp(3 * numpy.pi / 2 * 1j * -1),
            numpy.pi / 6: -1 * numpy.exp(numpy.pi / 6 * 1j * -1),
            2 * numpy.pi: -1 * numpy.exp(2 * numpy.pi * 1j * -1),
        },
    ),
    (
        3,
        smolyay.basis.Trigonometric(0),
        {
            0: 0,
            numpy.pi / 3: 0,
            3 * numpy.pi / 2: 0,
            numpy.pi / 6: 0,
            2 * numpy.pi: 0,
        },
    ),
    (
        3,
        smolyay.basis.Trigonometric(1),
        {
            0: -1j,
            numpy.pi / 3: -1j * numpy.exp(numpy.pi / 3 * 1j),
            3 * numpy.pi / 2: -1j * numpy.exp(3 * numpy.pi / 2 * 1j),
            numpy.pi / 6: -1j * numpy.exp(numpy.pi / 6 * 1j),
            2 * numpy.pi: -1j * numpy.exp(2 * numpy.pi * 1j),
        },
    ),
    (
        3,
        smolyay.basis.Trigonometric(-1),
        {
            0: 1j,
            numpy.pi / 3: 1j * numpy.exp(numpy.pi / 3 * 1j * -1),
            3 * numpy.pi / 2: 1j * numpy.exp(3 * numpy.pi / 2 * 1j * -1),
            numpy.pi / 6: 1j * numpy.exp(numpy.pi / 6 * 1j * -1),
            2 * numpy.pi: 1j * numpy.exp(2 * numpy.pi * 1j * -1),
        },
    ),
]

basis_outside_domain = [
    (smolyay.basis.ChebyshevFirstKind(4), 1.01, -1.01, 0),
    (smolyay.basis.ChebyshevSecondKind(4), 1.01, -1.01, 0),
    (smolyay.basis.Trigonometric(4), 7, -0.01, numpy.pi),
]

basis_id = [
    "1st Cheb [0]",
    "1st Cheb [1]",
    "1st Cheb [2]",
    "2nd Cheb [0]",
    "2nd Cheb [1]",
    "2nd Cheb [2]",
    "Trig [0]",
    "Trig [1]",
    "Trig [-1]",
]
basis_id_nth_derivative = [
    "n2-1st Cheb [0]",
    "n2-1st Cheb [1]",
    "n2-1st Cheb [2]",
    "n2-1st Cheb [3]",
    "n2-Trig [0]",
    "n2-Trig [1]",
    "n2-Trig [-1]",
    "n3-Trig [0]",
    "n3-Trig [1]",
    "n3-Trig [-1]",
]
basis_set_call_answer_key = {
    "ChebyshevFirstKind": (
        [[1, 1, 1, 1, 1], [0.5, 1, -1, -0.25, -0.5], [-0.5, 1, 1, -0.875, -0.5]],
        [7, 12, -8, -0.5, -3],
    ),
    "ChebyshevSecondKind": (
        [[1, 1, 1, 1, 1], [1, 2, -2, -0.5, -1], [0, 3, 3, -0.75, 0]],
        [7, 12, -8, -0.5, -3],
    ),
    "Trigonometric": (
        [
            [1, 1, 1, 1, 1],
            [
                1,
                numpy.exp(numpy.pi / 3 * 1j),
                numpy.exp(3 * numpy.pi / 2 * 1j),
                numpy.exp(numpy.pi / 6 * 1j),
                numpy.exp(2 * numpy.pi * 1j),
            ],
            [
                1,
                numpy.exp(numpy.pi / 3 * 1j * -1),
                numpy.exp(3 * numpy.pi / 2 * 1j * -1),
                numpy.exp(numpy.pi / 6 * 1j * -1),
                numpy.exp(2 * numpy.pi * 1j * -1),
            ],
        ],
        [-8, -14 / 3, 7, -19 / 3, 12],
    ),
}

basis_set_derivative_answer_key = {
    "ChebyshevFirstKind": (
        [
            [0, 0, 0, 0, 0],
            [1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10],
            [2 / 10, 4 / 10, -4 / 10, -1 / 10, -2 / 10],
        ],
        [7, 12, -8, -0.5, -3],
    ),
    "ChebyshevSecondKind": (
        [
            [0, 0, 0, 0, 0],
            [2 / 10, 2 / 10, 2 / 10, 2 / 10, 2 / 10],
            [4 / 10, 8 / 10, -8 / 10, -2 / 10, -4 / 10],
        ],
        [7, 12, -8, -0.5, -3],
    ),
    "Trigonometric": (
        [
            [0, 0, 0, 0, 0],
            [
                1j * numpy.pi / 10,
                1j * numpy.exp(numpy.pi / 3 * 1j) * numpy.pi / 10,
                1j * numpy.exp(3 * numpy.pi / 2 * 1j) * numpy.pi / 10,
                1j * numpy.exp(numpy.pi / 6 * 1j) * numpy.pi / 10,
                1j * numpy.exp(2 * numpy.pi * 1j) * numpy.pi / 10,
            ],
            [
                -1j * numpy.pi / 10,
                -1j * numpy.exp(numpy.pi / 3 * 1j * -1) * numpy.pi / 10,
                -1j * numpy.exp(3 * numpy.pi / 2 * 1j * -1) * numpy.pi / 10,
                -1j * numpy.exp(numpy.pi / 6 * 1j * -1) * numpy.pi / 10,
                -1j * numpy.exp(2 * numpy.pi * 1j * -1) * numpy.pi / 10,
            ],
        ],
        [-8, -14 / 3, 7, -19 / 3, 12],
    ),
}

basis_set_2nd_derivative_answer_key = {
    "ChebyshevFirstKind": (
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [4 / 100, 4 / 100, 4 / 100, 4 / 100, 4 / 100],
            [12 / 100, 24 / 100, -24 / 100, -6 / 100, -12 / 100],
            [8 / 100, 80 / 100, 80 / 100, -10 / 100, 8 / 100],
        ],
        [7, 12, -8, -0.5, -3],
    ),
    "Trigonometric": (
        [
            [0, 0, 0, 0, 0],
            [
                -1 * (numpy.pi / 10) ** 2,
                -numpy.exp(numpy.pi / 3 * 1j) * (numpy.pi / 10) ** 2,
                -numpy.exp(3 * numpy.pi / 2 * 1j) * (numpy.pi / 10) ** 2,
                -numpy.exp(numpy.pi / 6 * 1j) * (numpy.pi / 10) ** 2,
                -numpy.exp(2 * numpy.pi * 1j) * (numpy.pi / 10) ** 2,
            ],
            [
                -1 * (numpy.pi / 10) ** 2,
                -numpy.exp(numpy.pi / 3 * 1j * -1) * (numpy.pi / 10) ** 2,
                -numpy.exp(3 * numpy.pi / 2 * 1j * -1) * (numpy.pi / 10) ** 2,
                -numpy.exp(numpy.pi / 6 * 1j * -1) * (numpy.pi / 10) ** 2,
                -numpy.exp(2 * numpy.pi * 1j * -1) * (numpy.pi / 10) ** 2,
            ],
        ],
        [-8, -14 / 3, 7, -19 / 3, 12],
    ),
}
basis_set_ids = [
    "1st Cheb",
    "2nd Cheb",
    "Trig",
    "1st Cheb-nested",
    "1st Cheb-slow nested",
    "Trig-nested",
]


# initialization tests
@pytest.mark.parametrize(
    "basis_fun",
    [
        smolyay.basis.ChebyshevFirstKind,
        smolyay.basis.ChebyshevSecondKind,
    ],
    ids=["1st Cheb", "2nd Cheb"],
)
def test_cheb_initial(basis_fun):
    """test degrees and domain return correctly"""
    f2 = basis_fun(2)
    assert f2.degree == 2
    assert isinstance(f2.degree, int)
    assert numpy.array_equal(f2.domain, [-1, 1])
    f2.degree = float(3)
    assert f2.degree == 3
    assert isinstance(f2.degree, int)


def test_trig_initial():
    """test frequency and domain return correctly"""
    f2 = smolyay.basis.Trigonometric(2)
    assert f2.frequency == 2
    assert isinstance(f2.frequency, int)
    assert numpy.array_equal(f2.domain, [0, 2 * numpy.pi])
    f2.frequency = float(3)
    assert f2.frequency == 3
    assert isinstance(f2.frequency, int)


# Test outside of valid domain
@pytest.mark.parametrize(
    "basis_fun,too_large,too_small,valid_input",
    basis_outside_domain,
    ids=["1st Cheb", "2nd Cheb", "Trig"],
)
def test_call_outside_domain_error(basis_fun, too_large, too_small, valid_input):
    """Test call raises error for input outside domain"""
    with pytest.raises(ValueError):
        basis_fun(too_large)
    with pytest.raises(ValueError):
        basis_fun(too_small)
    with pytest.raises(ValueError):
        basis_fun([valid_input, too_small, valid_input])
    with pytest.raises(ValueError):
        basis_fun(
            [
                [valid_input, too_large, valid_input],
                [valid_input, valid_input, valid_input],
            ]
        )


@pytest.mark.parametrize(
    "basis_fun,too_large,too_small,valid_input",
    basis_outside_domain,
    ids=["1st Cheb", "2nd Cheb", "Trig"],
)
def test_derivative_outside_domain_error(basis_fun, too_large, too_small, valid_input):
    """Test derivative raises error for input outside domain"""
    with pytest.raises(ValueError):
        basis_fun.derivative(too_large)
    with pytest.raises(ValueError):
        basis_fun.derivative(too_small)
    with pytest.raises(ValueError):
        basis_fun.derivative([valid_input, too_small, valid_input])
    with pytest.raises(ValueError):
        basis_fun.derivative(
            [
                [valid_input, too_large, valid_input],
                [valid_input, valid_input, valid_input],
            ]
        )


# Test scale to domain
@pytest.mark.parametrize(
    "basis_fun,answer_single,answer_multi",
    [
        (smolyay.basis.ChebyshevFirstKind(0), -0.2, [-0.2, -0.1, 0, 0.1]),
        (smolyay.basis.ChebyshevSecondKind(0), -0.2, [-0.2, -0.1, 0, 0.1]),
        (
            smolyay.basis.Trigonometric(0),
            4 * numpy.pi / 5,
            [0.8 * numpy.pi, 0.9 * numpy.pi, numpy.pi, 1.1 * numpy.pi],
        ),
    ],
    ids=["1st Cheb", "2nd Cheb", "Trig"],
)
def test_scale_domain(basis_fun, answer_single, answer_multi):
    """Test the set can scale points to basis function domain"""
    domain = (-8, 12)
    assert basis_fun.scale_to_domain(0, domain) == pytest.approx(answer_single)
    assert numpy.allclose(basis_fun.scale_to_domain([0, 1, 2, 3], domain), answer_multi)


# Test call function correctness
@pytest.mark.parametrize("basis_fun,answer_key", basis_call_answer_key, ids=basis_id)
@pytest.mark.incremental
class TestCall:
    def test_call(self, basis_fun, answer_key):
        """Test basis function call"""
        for x, y in answer_key.items():
            assert basis_fun(x) == pytest.approx(y)

    def test_call_1D(self, basis_fun, answer_key):
        """Test basis function call with a 1D array"""
        xs = list(answer_key.keys())
        answers = [answer_key[x] for x in xs]
        assert numpy.shape(basis_fun(xs)) == numpy.shape(xs)
        assert numpy.allclose(basis_fun(xs), answers)

        xs1 = numpy.ones((1, 1)) * xs[0]
        answer1 = numpy.ones((1, 1)) * answers[0]
        assert numpy.shape(basis_fun(xs1)) == numpy.shape(xs1)
        assert numpy.allclose(basis_fun(xs1), answer1)

    def test_call_2D(self, basis_fun, answer_key):
        """Test basis function call with a 2D array"""
        unique_inputs = list(answer_key.keys())
        xs = list(numpy.resize(unique_inputs, (8,)))
        answers = [answer_key[x] for x in xs]

        xs1 = numpy.reshape(xs, (2, 4))
        answer1 = numpy.reshape(answers, (2, 4))
        assert numpy.shape(basis_fun(xs1)) == numpy.shape(xs1)
        assert numpy.allclose(basis_fun(xs1), answer1)

        xs2 = numpy.reshape(xs, (1, 8))
        answer2 = numpy.reshape(answers, (1, 8))
        assert numpy.shape(basis_fun(xs2)) == numpy.shape(xs2)
        assert numpy.allclose(basis_fun(xs2), answer2)

        xs3 = numpy.reshape(xs, (8, 1))
        answer3 = numpy.reshape(answers, (8, 1))
        assert numpy.shape(basis_fun(xs3)) == numpy.shape(xs3)
        assert numpy.allclose(basis_fun(xs3), answer3)

        xs4 = numpy.ones((1, 1)) * xs[0]
        answer4 = numpy.ones((1, 1)) * answers[0]
        assert numpy.shape(basis_fun(xs4)) == numpy.shape(xs4)
        assert numpy.allclose(basis_fun(xs4), answer4)

    def test_call_3D(self, basis_fun, answer_key):
        """Test basis function call with a 3D array"""
        unique_inputs = list(answer_key.keys())
        xs = list(numpy.resize(unique_inputs, (24,)))
        answers = [answer_key[x] for x in xs]

        xs1 = numpy.reshape(xs, (2, 3, 4))
        answer1 = numpy.reshape(answers, (2, 3, 4))
        assert numpy.shape(basis_fun(xs1)) == numpy.shape(xs1)
        assert numpy.allclose(basis_fun(xs1), answer1)

        xs2 = numpy.reshape(xs, (1, 1, 24))
        answer2 = numpy.reshape(answers, (1, 1, 24))
        assert numpy.shape(basis_fun(xs2)) == numpy.shape(xs2)
        assert numpy.allclose(basis_fun(xs2), answer2)

        xs3 = numpy.reshape(xs, (1, 24, 1))
        answer3 = numpy.reshape(answers, (1, 24, 1))
        assert numpy.shape(basis_fun(xs3)) == numpy.shape(xs3)
        assert numpy.allclose(basis_fun(xs3), answer3)

        xs4 = numpy.reshape(xs, (1, 1, 24))
        answer4 = numpy.reshape(answers, (1, 1, 24))
        assert numpy.shape(basis_fun(xs4)) == numpy.shape(xs4)
        assert numpy.allclose(basis_fun(xs4), answer4)

        xs5 = numpy.reshape(xs, (1, 6, 4))
        answer5 = numpy.reshape(answers, (1, 6, 4))
        assert numpy.shape(basis_fun(xs5)) == numpy.shape(xs5)
        assert numpy.allclose(basis_fun(xs5), answer5)

        xs6 = numpy.reshape(xs, (6, 4, 1))
        answer6 = numpy.reshape(answers, (6, 4, 1))
        assert numpy.shape(basis_fun(xs6)) == numpy.shape(xs6)
        assert numpy.allclose(basis_fun(xs6), answer6)

        xs7 = numpy.reshape(xs, (6, 1, 4))
        answer7 = numpy.reshape(answers, (6, 1, 4))
        assert numpy.shape(basis_fun(xs7)) == numpy.shape(xs7)
        assert numpy.allclose(basis_fun(xs7), answer7)

        xs8 = numpy.ones((1, 1, 1)) * xs[0]
        answer8 = numpy.ones((1, 1, 1)) * answers[0]
        assert numpy.shape(basis_fun(xs8)) == numpy.shape(xs8)
        assert numpy.allclose(basis_fun(xs8), answer8)


# Test derivative function correctness
@pytest.mark.parametrize(
    "basis_fun,answer_key", basis_derivative_answer_key, ids=basis_id
)
@pytest.mark.incremental
class TestDeriviative:
    def test_derivative(self, basis_fun, answer_key):
        """Test basis function derivative"""
        for x, y in answer_key.items():
            assert basis_fun.derivative(x) == pytest.approx(y)

    def test_derivative_1D(self, basis_fun, answer_key):
        """Test basis function derivative with a 1D array"""
        xs = list(answer_key.keys())
        answers = [answer_key[x] for x in xs]
        assert numpy.shape(basis_fun.derivative(xs)) == numpy.shape(xs)
        assert numpy.allclose(basis_fun.derivative(xs), answers)

        xs1 = numpy.ones((1, 1)) * xs[0]
        answer1 = numpy.ones((1, 1)) * answers[0]
        assert numpy.shape(basis_fun.derivative(xs1)) == numpy.shape(xs1)
        assert numpy.allclose(basis_fun.derivative(xs1), answer1)

    def test_derivative_2D(self, basis_fun, answer_key):
        """Test basis function derivative with a 2D array"""
        unique_inputs = list(answer_key.keys())
        xs = list(numpy.resize(unique_inputs, (8,)))
        answers = [answer_key[x] for x in xs]

        xs1 = numpy.reshape(xs, (2, 4))
        answer1 = numpy.reshape(answers, (2, 4))
        assert numpy.shape(basis_fun.derivative(xs1)) == numpy.shape(xs1)
        assert numpy.allclose(basis_fun.derivative(xs1), answer1)

        xs2 = numpy.reshape(xs, (1, 8))
        answer2 = numpy.reshape(answers, (1, 8))
        assert numpy.shape(basis_fun.derivative(xs2)) == numpy.shape(xs2)
        assert numpy.allclose(basis_fun.derivative(xs2), answer2)

        xs3 = numpy.reshape(xs, (8, 1))
        answer3 = numpy.reshape(answers, (8, 1))
        assert numpy.shape(basis_fun.derivative(xs3)) == numpy.shape(xs3)
        assert numpy.allclose(basis_fun.derivative(xs3), answer3)

        xs4 = numpy.ones((1, 1)) * xs[0]
        answer4 = numpy.ones((1, 1)) * answers[0]
        assert numpy.shape(basis_fun.derivative(xs4)) == numpy.shape(xs4)
        assert numpy.allclose(basis_fun.derivative(xs4), answer4)

    def test_derivative_3D(self, basis_fun, answer_key):
        """Test basis function derivative with a 3D array"""
        unique_inputs = list(answer_key.keys())
        xs = list(numpy.resize(unique_inputs, (24,)))
        answers = [answer_key[x] for x in xs]

        xs1 = numpy.reshape(xs, (2, 3, 4))
        answer1 = numpy.reshape(answers, (2, 3, 4))
        assert numpy.shape(basis_fun.derivative(xs1)) == numpy.shape(xs1)
        assert numpy.allclose(basis_fun.derivative(xs1), answer1)

        xs2 = numpy.reshape(xs, (1, 1, 24))
        answer2 = numpy.reshape(answers, (1, 1, 24))
        assert numpy.shape(basis_fun.derivative(xs2)) == numpy.shape(xs2)
        assert numpy.allclose(basis_fun.derivative(xs2), answer2)

        xs3 = numpy.reshape(xs, (1, 24, 1))
        answer3 = numpy.reshape(answers, (1, 24, 1))
        assert numpy.shape(basis_fun.derivative(xs3)) == numpy.shape(xs3)
        assert numpy.allclose(basis_fun.derivative(xs3), answer3)

        xs4 = numpy.reshape(xs, (1, 1, 24))
        answer4 = numpy.reshape(answers, (1, 1, 24))
        assert numpy.shape(basis_fun.derivative(xs4)) == numpy.shape(xs4)
        assert numpy.allclose(basis_fun.derivative(xs4), answer4)

        xs5 = numpy.reshape(xs, (1, 6, 4))
        answer5 = numpy.reshape(answers, (1, 6, 4))
        assert numpy.shape(basis_fun.derivative(xs5)) == numpy.shape(xs5)
        assert numpy.allclose(basis_fun.derivative(xs5), answer5)

        xs6 = numpy.reshape(xs, (6, 4, 1))
        answer6 = numpy.reshape(answers, (6, 4, 1))
        assert numpy.shape(basis_fun.derivative(xs6)) == numpy.shape(xs6)
        assert numpy.allclose(basis_fun.derivative(xs6), answer6)

        xs7 = numpy.reshape(xs, (6, 1, 4))
        answer7 = numpy.reshape(answers, (6, 1, 4))
        assert numpy.shape(basis_fun.derivative(xs7)) == numpy.shape(xs7)
        assert numpy.allclose(basis_fun.derivative(xs7), answer7)

        xs8 = numpy.ones((1, 1, 1)) * xs[0]
        answer8 = numpy.ones((1, 1, 1)) * answers[0]
        assert numpy.shape(basis_fun.derivative(xs8)) == numpy.shape(xs8)
        assert numpy.allclose(basis_fun.derivative(xs8), answer8)


# Test higher order derivative function correctness
@pytest.mark.parametrize(
    "n,basis_fun,answer_key",
    basis_nth_derivative_answer_key,
    ids=basis_id_nth_derivative,
)
@pytest.mark.incremental
class TestNthDerivative:
    def test_nth_derivative(self, n, basis_fun, answer_key):
        """Test basis function derivative"""
        for x, y in answer_key.items():
            assert basis_fun.derivative(x, n) == pytest.approx(y)

    def test_nth_derivative_1D(self, n, basis_fun, answer_key):
        """Test basis function derivative with a 1D array"""
        xs = list(answer_key.keys())
        answers = [answer_key[x] for x in xs]
        assert numpy.shape(basis_fun.derivative(xs, n)) == numpy.shape(xs)
        assert numpy.allclose(basis_fun.derivative(xs, n), answers)

        xs1 = numpy.ones((1, 1)) * xs[0]
        answer1 = numpy.ones((1, 1)) * answers[0]
        assert numpy.shape(basis_fun.derivative(xs1, n)) == numpy.shape(xs1)
        assert numpy.allclose(basis_fun.derivative(xs1, n), answer1)

    def test_nth_derivative_2D(self, n, basis_fun, answer_key):
        """Test basis function derivative with a 2D array"""
        unique_inputs = list(answer_key.keys())
        xs = list(numpy.resize(unique_inputs, (8,)))
        answers = [answer_key[x] for x in xs]

        xs1 = numpy.reshape(xs, (2, 4))
        answer1 = numpy.reshape(answers, (2, 4))
        assert numpy.shape(basis_fun.derivative(xs1, n)) == numpy.shape(xs1)
        assert numpy.allclose(basis_fun.derivative(xs1, n), answer1)

        xs2 = numpy.reshape(xs, (1, 8))
        answer2 = numpy.reshape(answers, (1, 8))
        assert numpy.shape(basis_fun.derivative(xs2, n)) == numpy.shape(xs2)
        assert numpy.allclose(basis_fun.derivative(xs2, n), answer2)

        xs3 = numpy.reshape(xs, (8, 1))
        answer3 = numpy.reshape(answers, (8, 1))
        assert numpy.shape(basis_fun.derivative(xs3, n)) == numpy.shape(xs3)
        assert numpy.allclose(basis_fun.derivative(xs3, n), answer3)

        xs4 = numpy.ones((1, 1)) * xs[0]
        answer4 = numpy.ones((1, 1)) * answers[0]
        assert numpy.shape(basis_fun.derivative(xs4, n)) == numpy.shape(xs4)
        assert numpy.allclose(basis_fun.derivative(xs4, n), answer4)

    def test_nth_derivative_3D(self, n, basis_fun, answer_key):
        """Test basis function derivative with a 3D array"""
        unique_inputs = list(answer_key.keys())
        xs = list(numpy.resize(unique_inputs, (24,)))
        answers = [answer_key[x] for x in xs]

        xs1 = numpy.reshape(xs, (2, 3, 4))
        answer1 = numpy.reshape(answers, (2, 3, 4))
        assert numpy.shape(basis_fun.derivative(xs1, n)) == numpy.shape(xs1)
        assert numpy.allclose(basis_fun.derivative(xs1, n), answer1)

        xs2 = numpy.reshape(xs, (1, 1, 24))
        answer2 = numpy.reshape(answers, (1, 1, 24))
        assert numpy.shape(basis_fun.derivative(xs2, n)) == numpy.shape(xs2)
        assert numpy.allclose(basis_fun.derivative(xs2, n), answer2)

        xs3 = numpy.reshape(xs, (1, 24, 1))
        answer3 = numpy.reshape(answers, (1, 24, 1))
        assert numpy.shape(basis_fun.derivative(xs3, n)) == numpy.shape(xs3)
        assert numpy.allclose(basis_fun.derivative(xs3, n), answer3)

        xs4 = numpy.reshape(xs, (1, 1, 24))
        answer4 = numpy.reshape(answers, (1, 1, 24))
        assert numpy.shape(basis_fun.derivative(xs4, n)) == numpy.shape(xs4)
        assert numpy.allclose(basis_fun.derivative(xs4, n), answer4)

        xs5 = numpy.reshape(xs, (1, 6, 4))
        answer5 = numpy.reshape(answers, (1, 6, 4))
        assert numpy.shape(basis_fun.derivative(xs5, n)) == numpy.shape(xs5)
        assert numpy.allclose(basis_fun.derivative(xs5, n), answer5)

        xs6 = numpy.reshape(xs, (6, 4, 1))
        answer6 = numpy.reshape(answers, (6, 4, 1))
        assert numpy.shape(basis_fun.derivative(xs6, n)) == numpy.shape(xs6)
        assert numpy.allclose(basis_fun.derivative(xs6, n), answer6)

        xs7 = numpy.reshape(xs, (6, 1, 4))
        answer7 = numpy.reshape(answers, (6, 1, 4))
        assert numpy.shape(basis_fun.derivative(xs7, n)) == numpy.shape(xs7)
        assert numpy.allclose(basis_fun.derivative(xs7, n), answer7)

        xs8 = numpy.ones((1, 1, 1)) * xs[0]
        answer8 = numpy.ones((1, 1, 1)) * answers[0]
        assert numpy.shape(basis_fun.derivative(xs8, n)) == numpy.shape(xs8)
        assert numpy.allclose(basis_fun.derivative(xs8, n), answer8)


# Test call correctness at points that are special to a basis function
def test_cheb_call_extrema_points():
    """Test chebyshev polynomial at extrema"""
    f = smolyay.basis.ChebyshevFirstKind(4)
    extrema_points = [-1.0, -1 / numpy.sqrt(2), 0, 1 / numpy.sqrt(2), 1]
    extrema_output = [1, -1, 1, -1, 1]
    assert numpy.allclose(f(extrema_points), extrema_output)


def test_cheb_call_root_points():
    """Test chebyshev polynomial at roots"""
    f = smolyay.basis.ChebyshevFirstKind(4)
    root_points = [
        -numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
        -numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
        numpy.sqrt(numpy.sqrt(2) - 1) / (2**0.75),
        numpy.sqrt(numpy.sqrt(2) + 1) / (2**0.75),
    ]
    assert numpy.allclose(f(root_points), numpy.zeros(4))


def test_cheb_2nd_call_root_points():
    """Test chebyshev polynomial roots"""
    f = smolyay.basis.ChebyshevSecondKind(3)
    root_points = [-1 / numpy.sqrt(2), 0, 1 / numpy.sqrt(2)]
    assert numpy.allclose(f(root_points), numpy.zeros(3))


def test_is_complex():
    """Test that basis functions that output complex values indicate it"""
    assert smolyay.basis.ChebyshevFirstKind(0)._is_complex == False
    assert smolyay.basis.ChebyshevSecondKind(0)._is_complex == False
    assert smolyay.basis.Trigonometric(0)._is_complex == True


# Test a set of basis functions
def test_set_base_class_initialize():
    """Test BasisFunctionSet correctly initializes"""
    bf = smolyay.basis.ChebyshevFirstKind(0)
    f = smolyay.basis.BasisFunctionSet([bf])
    assert f.basis_functions[0] is bf
    assert f[0] is bf
    assert len(f) == 1
    assert numpy.array_equal(f.domain, [-1, 1])
    bf = smolyay.basis.Trigonometric(0)
    f = smolyay.basis.BasisFunctionSet([bf])
    assert f.basis_functions[0] is bf
    assert f[0] is bf
    assert len(f) == 1
    assert numpy.array_equal(f.domain, [0, 2 * numpy.pi])


def test_set_base_class_initialize_error():
    """Test BasisFunctionSet gives error if basis function have different domains"""
    with pytest.raises(AttributeError):
        f = smolyay.basis.BasisFunctionSet([])
        f.domain
    with pytest.raises(AttributeError):
        f = smolyay.basis.BasisFunctionSet([])
        f.scale_to_domain([1, 2, 3, 4], [0, 5])
    with pytest.raises(TypeError):
        smolyay.basis.BasisFunctionSet(
            [smolyay.basis.ChebyshevFirstKind(0), smolyay.basis.Trigonometric(0)]
        )
    with pytest.raises(TypeError):
        smolyay.basis.BasisFunctionSet(
            [smolyay.basis.ChebyshevSecondKind(0), smolyay.basis.Trigonometric(0)]
        )
    with pytest.raises(TypeError):
        smolyay.basis.BasisFunctionSet(
            [smolyay.basis.ChebyshevFirstKind(0), smolyay.basis.ChebyshevSecondKind(0)]
        )


@pytest.mark.parametrize(
    "basis_set",
    [
        smolyay.basis.ChebyshevFirstKindBasisFunctionSet,
        smolyay.basis.ChebyshevSecondKindBasisFunctionSet,
    ],
    ids=[
        "ChebyshevFirstKind",
        "ChebyshevSecondKind",
    ],
)
def test_set_cheb_initialize(basis_set):
    """Test Chebyshev function sets correctly initialize"""
    f = basis_set(3)
    assert f.basis_functions[0].degree == 0
    assert f[0].degree == 0
    assert f.basis_functions[1].degree == 1
    assert f[1].degree == 1
    assert f.basis_functions[2].degree == 2
    assert f[2].degree == 2
    assert len(f) == 3
    assert numpy.array_equal(f.domain, [-1, 1])


def test_set_trig_initialize():
    """Test TrigonometricBasisFunctionSet correctly initializes"""
    f = smolyay.basis.TrigonometricBasisFunctionSet(3)
    assert f.basis_functions[0].frequency == 0
    assert f[0].frequency == 0
    assert f.basis_functions[1].frequency == 1
    assert f[1].frequency == 1
    assert f.basis_functions[2].frequency == -1
    assert f[2].frequency == -1
    assert len(f) == 3
    assert numpy.array_equal(f.domain, [0, 2 * numpy.pi])


@pytest.mark.parametrize(
    "basis_set",
    [
        smolyay.basis.ChebyshevFirstKindBasisFunctionSet,
        smolyay.basis.ChebyshevSecondKindBasisFunctionSet,
        smolyay.basis.TrigonometricBasisFunctionSet,
    ],
    ids=[
        "ChebyshevFirstKind",
        "ChebyshevSecondKind",
        "Trigonometric",
    ],
)
def test_set_initialize_empty(basis_set):
    """Test basis function sets initialize when empty"""
    f = basis_set(0)
    assert len(f) == 0
    assert numpy.array_equal(f.basis_functions, [])
    with pytest.raises(AttributeError):
        f.domain
    with pytest.raises(AttributeError):
        f.scale_to_domain([1, 2, 3, 4], [0, 5])


def test_nested_set_base_class_initialize():
    """Test NestedBasisFunctionSet correctly initializes"""
    f = smolyay.basis.NestedBasisFunctionSet([], [])
    assert len(f) == 0
    assert f.num_levels == 0
    assert numpy.array_equal(f.num_per_level, [])
    assert numpy.array_equal(f.start_level, [])
    assert numpy.array_equal(f.end_level, [])
    bf = [smolyay.basis.ChebyshevFirstKind(n) for n in range(5)]
    f = smolyay.basis.NestedBasisFunctionSet(bf, [1, 1, 1, 2])
    assert f.basis_functions == bf
    assert len(f) == 5
    assert f.num_levels == 4
    assert numpy.array_equal(f.num_per_level, [1, 1, 1, 2])
    assert numpy.array_equal(f.start_level, [0, 1, 2, 3])
    assert numpy.array_equal(f.end_level, [1, 2, 3, 5])
    assert numpy.array_equal(f.level(3), bf[3:])


def test_nested_set_base_class_initialize_error():
    """Test NestedBasisFunctionSet error for invalid constructor inputs"""
    bf = [smolyay.basis.ChebyshevFirstKind(n) for n in range(5)]
    with pytest.raises(AttributeError):
        f = smolyay.basis.NestedBasisFunctionSet([], [0])
        f.domain
    with pytest.raises(AttributeError):
        f = smolyay.basis.NestedBasisFunctionSet([], [0])
        f.scale_to_domain([1, 2, 3, 4], [0, 5])
    with pytest.raises(ValueError):
        smolyay.basis.NestedBasisFunctionSet(bf, [1, 1, 1, 1])
    with pytest.raises(ValueError):
        smolyay.basis.NestedBasisFunctionSet(bf, [1, 1, 2, 3])
    with pytest.raises(TypeError):
        smolyay.basis.NestedBasisFunctionSet(
            [smolyay.basis.ChebyshevFirstKind(0), smolyay.basis.Trigonometric(0)],
            [1, 1],
        )
    with pytest.raises(TypeError):
        smolyay.basis.NestedBasisFunctionSet(
            [smolyay.basis.ChebyshevSecondKind(0), smolyay.basis.Trigonometric(0)],
            [1, 1],
        )


@pytest.mark.parametrize(
    "nested_sets,domain,length_2",
    [
        (smolyay.basis.NestedClenshawCurtisBasisFunctionSet, [-1, 1], 3),
        (smolyay.basis.SlowNestedClenshawCurtisBasisFunctionSet, [-1, 1], 3),
        (smolyay.basis.NestedTrigonometricBasisFunctionSet, [0, 2 * numpy.pi], 3),
    ],
    ids=[
        "NestedClenshawCurtis",
        "SlowNestedClenshawCurtis",
        "NestedTrigonometric",
    ],
)
def test_nested_sets_initialize(nested_sets, domain, length_2):
    """Test nested basis function sets initialization"""
    bf = nested_sets(2)
    assert numpy.array_equal(bf.domain, domain)
    assert bf.num_levels == 2
    assert len(bf) == length_2
    assert len(bf.num_per_level) == 2
    assert len(bf.start_level) == 2
    assert len(bf.end_level) == 2


@pytest.mark.parametrize(
    "nested_sets",
    [
        (smolyay.basis.NestedClenshawCurtisBasisFunctionSet),
        (smolyay.basis.SlowNestedClenshawCurtisBasisFunctionSet),
        (smolyay.basis.NestedTrigonometricBasisFunctionSet),
    ],
    ids=[
        "NestedClenshawCurtis",
        "SlowNestedClenshawCurtis",
        "NestedTrigonometric",
    ],
)
def test_num_levels_error(nested_sets):
    """test error given invalid num_levels"""
    with pytest.raises(ValueError):
        nested_sets(0)


@pytest.mark.parametrize(
    "basis_set,answer_single,answer_multi",
    [
        (
            smolyay.basis.ChebyshevFirstKindBasisFunctionSet(1),
            -0.2,
            [-0.2, -0.1, 0, 0.1],
        ),
        (
            smolyay.basis.ChebyshevSecondKindBasisFunctionSet(1),
            -0.2,
            [-0.2, -0.1, 0, 0.1],
        ),
        (
            smolyay.basis.TrigonometricBasisFunctionSet(1),
            4 * numpy.pi / 5,
            [0.8 * numpy.pi, 0.9 * numpy.pi, numpy.pi, 1.1 * numpy.pi],
        ),
        (
            smolyay.basis.NestedClenshawCurtisBasisFunctionSet(1),
            -0.2,
            [-0.2, -0.1, 0, 0.1],
        ),
        (
            smolyay.basis.SlowNestedClenshawCurtisBasisFunctionSet(1),
            -0.2,
            [-0.2, -0.1, 0, 0.1],
        ),
        (
            smolyay.basis.NestedTrigonometricBasisFunctionSet(1),
            4 * numpy.pi / 5,
            [0.8 * numpy.pi, 0.9 * numpy.pi, numpy.pi, 1.1 * numpy.pi],
        ),
    ],
    ids=basis_set_ids,
)
def test_set_scale_domain(basis_set, answer_single, answer_multi):
    """Test the set can scale points to basis function domain"""
    domain = (-8, 12)
    assert basis_set.scale_to_domain(0, domain) == pytest.approx(answer_single)
    assert numpy.allclose(
        basis_set.scale_to_domain(numpy.array([0, 1, 2, 3]), domain),
        answer_multi,
    )


@pytest.mark.parametrize(
    "basis_function_set,key_for_answer",
    [
        (
            smolyay.basis.ChebyshevFirstKindBasisFunctionSet(3),
            "ChebyshevFirstKind",
        ),
        (
            smolyay.basis.ChebyshevSecondKindBasisFunctionSet(3),
            "ChebyshevSecondKind",
        ),
        (
            smolyay.basis.TrigonometricBasisFunctionSet(3),
            "Trigonometric",
        ),
        (
            smolyay.basis.NestedClenshawCurtisBasisFunctionSet(2),
            "ChebyshevFirstKind",
        ),
        (
            smolyay.basis.SlowNestedClenshawCurtisBasisFunctionSet(2),
            "ChebyshevFirstKind",
        ),
        (
            smolyay.basis.NestedTrigonometricBasisFunctionSet(2),
            "Trigonometric",
        ),
    ],
    ids=basis_set_ids,
)
def test_set_call(basis_function_set, key_for_answer):
    """Test the set can scale points to basis function domain"""
    domain = (-8, 12)
    answer_key = basis_set_call_answer_key[key_for_answer][0]
    X = basis_set_call_answer_key[key_for_answer][1]
    assert numpy.allclose(basis_function_set(X, domain), answer_key)


@pytest.mark.parametrize(
    "basis_function_set,key_for_answer",
    [
        (
            smolyay.basis.ChebyshevFirstKindBasisFunctionSet(3),
            "ChebyshevFirstKind",
        ),
        (
            smolyay.basis.ChebyshevSecondKindBasisFunctionSet(3),
            "ChebyshevSecondKind",
        ),
        (
            smolyay.basis.TrigonometricBasisFunctionSet(3),
            "Trigonometric",
        ),
        (
            smolyay.basis.NestedClenshawCurtisBasisFunctionSet(2),
            "ChebyshevFirstKind",
        ),
        (
            smolyay.basis.SlowNestedClenshawCurtisBasisFunctionSet(2),
            "ChebyshevFirstKind",
        ),
        (
            smolyay.basis.NestedTrigonometricBasisFunctionSet(2),
            "Trigonometric",
        ),
    ],
    ids=basis_set_ids,
)
def test_set_derivative(basis_function_set, key_for_answer):
    """Test the set can compute derivative of all basis functions."""
    domain = (-8, 12)
    answer_key = basis_set_derivative_answer_key[key_for_answer][0]
    X = basis_set_derivative_answer_key[key_for_answer][1]
    assert numpy.allclose(basis_function_set.derivative(X, domain=domain), answer_key)


@pytest.mark.parametrize(
    "basis_function_set,key_for_answer",
    [
        (
            smolyay.basis.ChebyshevFirstKindBasisFunctionSet(5),
            "ChebyshevFirstKind",
        ),
        (
            smolyay.basis.TrigonometricBasisFunctionSet(3),
            "Trigonometric",
        ),
        (
            smolyay.basis.NestedClenshawCurtisBasisFunctionSet(3),
            "ChebyshevFirstKind",
        ),
        (
            smolyay.basis.SlowNestedClenshawCurtisBasisFunctionSet(3),
            "ChebyshevFirstKind",
        ),
        (
            smolyay.basis.NestedTrigonometricBasisFunctionSet(2),
            "Trigonometric",
        ),
    ],
    ids=[
        "1st Cheb",
        "Trig",
        "1st Cheb-nested",
        "1st Cheb-slow nested",
        "Trig-nested",
    ],
)
def test_set_2nd_derivative(basis_function_set, key_for_answer):
    """Test the set can compute 2nd derivative of all basis functions."""
    domain = (-8, 12)
    answer_key = basis_set_2nd_derivative_answer_key[key_for_answer][0]
    X = basis_set_2nd_derivative_answer_key[key_for_answer][1]
    assert numpy.allclose(
        basis_function_set.derivative(X, domain=domain, n=2), answer_key
    )


@pytest.mark.parametrize(
    "nested_sets,num_per_level,start_level,end_level",
    [
        (
            smolyay.basis.NestedClenshawCurtisBasisFunctionSet(1),
            [1],
            [0],
            [1],
        ),
        (
            smolyay.basis.NestedClenshawCurtisBasisFunctionSet(2),
            [1, 2],
            [0, 1],
            [1, 3],
        ),
        (
            smolyay.basis.NestedClenshawCurtisBasisFunctionSet(3),
            [1, 2, 2],
            [0, 1, 3],
            [1, 3, 5],
        ),
        (
            smolyay.basis.NestedClenshawCurtisBasisFunctionSet(4),
            [1, 2, 2, 4],
            [0, 1, 3, 5],
            [1, 3, 5, 9],
        ),
        (
            smolyay.basis.SlowNestedClenshawCurtisBasisFunctionSet(1),
            [1],
            [0],
            [1],
        ),
        (
            smolyay.basis.SlowNestedClenshawCurtisBasisFunctionSet(2),
            [1, 2],
            [0, 1],
            [1, 3],
        ),
        (
            smolyay.basis.SlowNestedClenshawCurtisBasisFunctionSet(3),
            [1, 2, 2],
            [0, 1, 3],
            [1, 3, 5],
        ),
        (
            smolyay.basis.SlowNestedClenshawCurtisBasisFunctionSet(4),
            [1, 2, 2, 4],
            [0, 1, 3, 5],
            [1, 3, 5, 9],
        ),
        (
            smolyay.basis.SlowNestedClenshawCurtisBasisFunctionSet(5),
            [1, 2, 2, 4, 0],
            [0, 1, 3, 5, 9],
            [1, 3, 5, 9, 9],
        ),
        (
            smolyay.basis.SlowNestedClenshawCurtisBasisFunctionSet(6),
            [1, 2, 2, 4, 0, 8],
            [0, 1, 3, 5, 9, 9],
            [1, 3, 5, 9, 9, 17],
        ),
        (
            smolyay.basis.NestedTrigonometricBasisFunctionSet(1),
            [1],
            [0],
            [1],
        ),
        (
            smolyay.basis.NestedTrigonometricBasisFunctionSet(2),
            [1, 2],
            [0, 1],
            [1, 3],
        ),
        (
            smolyay.basis.NestedTrigonometricBasisFunctionSet(3),
            [1, 2, 6],
            [0, 1, 3],
            [1, 3, 9],
        ),
    ],
    ids=[
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
    ],
)
def test_nested_levels(nested_sets, num_per_level, start_level, end_level):
    """test number of points per level, start level indexes, and end level indexes"""
    assert numpy.array_equal(nested_sets.num_per_level, num_per_level)
    assert numpy.array_equal(nested_sets.start_level, start_level)
    assert numpy.array_equal(nested_sets.end_level, end_level)
    for level, (start, end) in enumerate(
        zip(nested_sets.start_level, nested_sets.end_level)
    ):
        assert len(nested_sets.level(level)) == len(nested_sets[start:end])
