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


# Test a set of basis functions
def test_set_initialize():
    """Test BasisFunctionSet correctly initializes"""
    f = smolyay.basis.BasisFunctionSet([])
    f2 = smolyay.basis.BasisFunctionSet([smolyay.basis.ChebyshevFirstKind(0)])
    assert f.basis_functions == []
    assert len(f) == 0
    assert f2.basis_functions[0].degree == 0
    assert f2[0].degree == 0
    assert len(f2) == 1


def test_set_initialize_error():
    """Test BasisFunctionSet gives error if basis function have different domains"""
    # this should not error
    smolyay.basis.BasisFunctionSet(
        [smolyay.basis.ChebyshevFirstKind(0), smolyay.basis.ChebyshevSecondKind(0)]
    )
    # this should error
    with pytest.raises(ValueError):
        smolyay.basis.BasisFunctionSet(
            [smolyay.basis.ChebyshevFirstKind(0), smolyay.basis.Trigonometric(0)]
        )
    with pytest.raises(ValueError):
        smolyay.basis.BasisFunctionSet(
            [smolyay.basis.ChebyshevSecondKind(0), smolyay.basis.Trigonometric(0)]
        )


@pytest.mark.parametrize(
    "basis_fun,answer_single,answer_multi",
    [
        (smolyay.basis.ChebyshevFirstKind, -0.2, [-0.2, -0.1, 0, 0.1]),
        (smolyay.basis.ChebyshevSecondKind, -0.2, [-0.2, -0.1, 0, 0.1]),
        (
            smolyay.basis.Trigonometric,
            4 * numpy.pi / 5,
            [0.8 * numpy.pi, 0.9 * numpy.pi, numpy.pi, 1.1 * numpy.pi],
        ),
    ],
    ids=["1st Cheb", "2nd Cheb", "Trig"],
)
def test_set_scale_domain(basis_fun, answer_single, answer_multi):
    """Test the set can scale points to basis function domain"""
    domain = (-8, 12)
    f = smolyay.basis.BasisFunctionSet([basis_fun(0)])
    assert f._scale_to_domain(0, domain) == pytest.approx(answer_single)
    assert numpy.allclose(
        f._scale_to_domain(numpy.array([0, 1, 2, 3]), domain), answer_multi
    )


def test_nested_set_initialize():
    """Test NestedBasisFunctionSet correctly initializes"""
    bf = [smolyay.basis.ChebyshevFirstKind(n) for n in range(5)]
    f = smolyay.basis.NestedBasisFunctionSet(bf, [1, 1, 1, 2])
    assert f.basis_functions == bf
    assert len(f) == 5
    assert f.num_levels == 4
    assert numpy.array_equal(f.end_level, [1, 2, 3, 5])
    assert numpy.array_equal(f.start_level, [0, 1, 2, 3])
    assert numpy.array_equal(f.level(3), bf[3:])


def test_nested_set_initialize_error():
    """Test NestedBasisFunctionSet error for invalid constructor inputs"""
    bf = [smolyay.basis.ChebyshevFirstKind(n) for n in range(5)]
    with pytest.raises(IndexError):
        f = smolyay.basis.NestedBasisFunctionSet(bf, [1, 1, 1, 1])
    with pytest.raises(IndexError):
        f = smolyay.basis.NestedBasisFunctionSet(bf, [1, 1, 2, 3])


def test_is_complex():
    """Test that basis functions that output complex values indicate it"""
    assert smolyay.basis.ChebyshevFirstKind(0)._is_complex == False
    assert smolyay.basis.ChebyshevSecondKind(0)._is_complex == False
    assert smolyay.basis.Trigonometric(0)._is_complex == True
