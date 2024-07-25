import pytest
import warnings

import numpy
import sklearn.preprocessing

from smolyay.normalize import (
    Normalizer,
    SymmetricalLogNormalizer,
    IntervalNormalizer,
    ZScoreNormalizer,
    AsinhNormalizer,
)


@pytest.mark.parametrize(
    "normal",
    [
        IntervalNormalizer(),
        SymmetricalLogNormalizer(),
        ZScoreNormalizer(),
        AsinhNormalizer(),
    ],
    ids=[
        "IntervalNormalizer",
        "SymmetricalLogNormalizer",
        "ZScoreNormalizer",
        "AsinhNormalizer",
    ],
)
def test_fit(normal):
    """Test that fit returns Normalizer"""
    x = [1, 2, 3, 4, 5]
    n = normal.fit(x)
    assert isinstance(n, Normalizer)
    assert n == normal
    normal.transform(x)
    normal.inverse_transform(x)


@pytest.mark.parametrize(
    "normal",
    [
        IntervalNormalizer(),
        SymmetricalLogNormalizer(),
        ZScoreNormalizer(),
        AsinhNormalizer(),
    ],
    ids=[
        "IntervalNormalizer",
        "SymmetricalLogNormalizer",
        "ZScoreNormalizer",
        "AsinhNormalizer",
    ],
)
@pytest.mark.parametrize(
    "x",
    [
        [1, 2, 3, 4, 5],
        numpy.reshape(numpy.arange(6), (2, 3)),
        numpy.reshape(numpy.arange(24), (2, 3, 4)),
    ],
    ids=["1D array", "2D array", "3D array"],
)
def test_check(normal, x):
    """Test if the all the normalizers pass the check"""
    normal.fit(x)
    assert normal.check_normalize(x)


@pytest.mark.parametrize(
    "normal_class,set_args",
    [
        (SymmetricalLogNormalizer, [{"linthresh": 6}, {"linthresh": 0.5}]),
        (AsinhNormalizer, [{"linthresh": 6}, {"linthresh": 0.5}]),
    ],
    ids=["SymmetricalLogNormalizer", "AsinhNormalizer"],
)
@pytest.mark.parametrize(
    "x",
    [
        [1, 2, 3, 4, 5],
        numpy.reshape(numpy.arange(6), (2, 3)),
        numpy.reshape(numpy.arange(24), (2, 3, 4)),
    ],
    ids=["1D array", "2D array", "3D array"],
)
def test_check_optional_args(normal_class, set_args, x):
    """Test if the all the normalizers pass check if given optional arguments"""
    for a in set_args:
        normal = normal_class(**a)
        normal.fit(x)
        assert normal.check_normalize(x)


@pytest.mark.parametrize(
    "normal,answers",
    [
        (IntervalNormalizer(), [0, 0.25, 0.5, 0.75, 1]),
        (SymmetricalLogNormalizer(), numpy.log10([2, 3, 4, 5, 6])),
        (
            ZScoreNormalizer(),
            [
                -2 / numpy.sqrt(2),
                -1 / numpy.sqrt(2),
                0,
                1 / numpy.sqrt(2),
                2 / numpy.sqrt(2),
            ],
        ),
        (AsinhNormalizer(), [numpy.arcsinh([1, 2, 3, 4, 5])]),
    ],
    ids=[
        "IntervalNormalizer",
        "ZScoreNormalizer",
        "SymmetricalLogNormalizer",
        "AsinhNormalizer",
    ],
)
def test_transform(normal, answers):
    """Test that the transforms are correct"""
    x = [1, 2, 3, 4, 5]
    normal.fit(x)
    assert numpy.allclose(normal.transform(x), answers)


@pytest.mark.parametrize(
    "normal,answers",
    [
        (
            SymmetricalLogNormalizer(6),
            numpy.log10([1 + 1 / 6, 1 + 1 / 3, 1.5, 1 + 2 / 3, 1 + 5 / 6]),
        ),
        (SymmetricalLogNormalizer(0.5), numpy.log10([3, 5, 7, 9, 11])),
        (AsinhNormalizer(6), [numpy.arcsinh([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]) * 6]),
        (AsinhNormalizer(0.5), [numpy.arcsinh([2, 4, 6, 8, 10]) * 0.5]),
    ],
    ids=["SymmetricalLog-6", "SymmetricalLog-0.5", "Asinh-6", "Asinh-0.5"],
)
def test_transform_optional_args(normal, answers):
    """Test that the transforms are correct"""
    x = [1, 2, 3, 4, 5]
    normal.fit(x)
    assert numpy.allclose(normal.transform(x), answers)


@pytest.mark.parametrize(
    "normal,answers",
    [
        (IntervalNormalizer(), [5, 9, 13, 17, 21]),
        (SymmetricalLogNormalizer(), [9, 99, 999, 9999, 99999]),
        (
            ZScoreNormalizer(),
            [
                numpy.sqrt(2) + 3,
                2 * numpy.sqrt(2) + 3,
                3 * numpy.sqrt(2) + 3,
                4 * numpy.sqrt(2) + 3,
                5 * numpy.sqrt(2) + 3,
            ],
        ),
        (AsinhNormalizer(), numpy.sinh([1, 2, 3, 4, 5])),
    ],
    ids=[
        "IntervalNormalizer",
        "SymmetricalLogNormalizer",
        "ZScoreNormalizer",
        "AsinhNormalizer",
    ],
)
def test_inverse(normal, answers):
    """Test that the inverse transforms are correct"""
    x = [1, 2, 3, 4, 5]
    normal.fit(x)
    assert numpy.allclose(normal.inverse_transform(x), answers)


@pytest.mark.parametrize(
    "normal,answers",
    [
        (SymmetricalLogNormalizer(6), [9 * 6, 99 * 6, 999 * 6, 9999 * 6, 99999 * 6]),
        (
            SymmetricalLogNormalizer(0.5),
            [9 * 0.5, 99 * 0.5, 999 * 0.5, 9999 * 0.5, 99999 * 0.5],
        ),
        (AsinhNormalizer(6), [numpy.sinh([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]) * 6]),
        (AsinhNormalizer(0.5), [numpy.sinh([2, 4, 6, 8, 10]) * 0.5]),
    ],
    ids=["SymmetricalLog-6", "SymmetricalLog-0.5", "Asinh-6", "Asinh-0.5"],
)
def test_inverse_optional_args(normal, answers):
    """Test that the inverse transforms are correct"""
    x = [1, 2, 3, 4, 5]
    normal.fit(x)
    assert numpy.allclose(normal.inverse_transform(x), answers)


@pytest.mark.parametrize(
    "normal",
    [
        IntervalNormalizer(),
        ZScoreNormalizer(),
        SymmetricalLogNormalizer(),
        SymmetricalLogNormalizer(6),
        SymmetricalLogNormalizer(0.5),
        AsinhNormalizer(),
        AsinhNormalizer(6),
        AsinhNormalizer(0.5),
    ],
    ids=[
        "Interval",
        "ZScore",
        "SymmetricalLog-1",
        "SymmetricalLog-6",
        "SymmetricalLog-0.5",
        "Asinh-1",
        "Asinh-6",
        "Asinh-0.5",
    ],
)
def test_transform_derivative(normal):
    """Test that the derivative function returns the rate of change of the transform."""
    sample = numpy.arange(15)
    normal.fit(sample)
    X = numpy.linspace(1, 2, 200)
    y = normal.transform(X)
    answer_key = numpy.gradient(y, X[1] - X[0])
    deriv_1 = normal.derivative(X)
    assert numpy.allclose(deriv_1, answer_key, rtol=0.01)
    answer_key = numpy.gradient(deriv_1, X[1] - X[0])
    deriv_2 = normal.derivative(X, 2)
    assert numpy.allclose(deriv_2, answer_key, rtol=0.01)
    answer_key = numpy.gradient(deriv_2, X[1] - X[0])
    deriv_3 = normal.derivative(X, 3)
    assert numpy.allclose(deriv_3, answer_key, rtol=0.01)


@pytest.mark.parametrize(
    "normal",
    [
        IntervalNormalizer(),
        ZScoreNormalizer(),
        SymmetricalLogNormalizer(),
        SymmetricalLogNormalizer(6),
        SymmetricalLogNormalizer(0.5),
        AsinhNormalizer(),
        AsinhNormalizer(6),
        AsinhNormalizer(0.5),
    ],
    ids=[
        "Interval",
        "ZScore",
        "SymmetricalLog-1",
        "SymmetricalLog-6",
        "SymmetricalLog-0.5",
        "Asinh-1",
        "Asinh-6",
        "Asinh-0.5",
    ],
)
def test_inverse_transform_derivative(normal):
    """Test that the derivative function returns the rate of change of the inverse transform."""
    sample = numpy.arange(15)
    normal.fit(sample)
    X = numpy.linspace(1, 2, 200)
    y = normal.inverse_transform(X)
    answer_key = numpy.gradient(y, X[1] - X[0])
    deriv_1 = normal.inverse_derivative(X)
    assert numpy.allclose(deriv_1, answer_key, rtol=0.01)
    answer_key = numpy.gradient(deriv_1, X[1] - X[0])
    deriv_2 = normal.inverse_derivative(X, 2)
    assert numpy.allclose(deriv_2, answer_key, rtol=0.01)
    answer_key = numpy.gradient(deriv_2, X[1] - X[0])
    deriv_3 = normal.inverse_derivative(X, 3)
    assert numpy.allclose(deriv_3, answer_key, rtol=0.01)


def test_interval_attributes():
    """Test if the attributes of IntervalNormalizer are added"""
    x = [1, 2, 3, 4, 5]
    normal = IntervalNormalizer()
    normal.fit(x)
    assert normal.max_val == 5
    assert normal.min_val == 1


def test_interval_transform_error():
    """Test if IntervalNormalizer returns an error if not fit first"""
    normal = IntervalNormalizer()
    with pytest.raises(ValueError):
        normal.transform([1, 2])
    with pytest.raises(ValueError):
        normal.inverse_transform([1, 2])
    with pytest.raises(ValueError):
        normal.check_normalize([1, 3])


def test_interval_refit():
    """Test that min and max are recaluated"""
    x1 = [1, 2, 3, 4, 5]
    x2 = [1, 2, 3, 4, 5, 6]
    min2 = numpy.min(x2)
    max2 = numpy.max(x2)
    normal = IntervalNormalizer()
    normal.fit(x1)
    normal.fit(x2)
    assert normal.min_val == min2
    assert normal.max_val == max2


def test_symlog_attributes():
    """Test if the attributes of SymmetricalLogNormalizer are added"""
    normal = SymmetricalLogNormalizer(linthresh=10)
    assert normal.linthresh == 10
    normal.linthresh = 20
    assert normal.linthresh == 20
    with pytest.raises(ValueError):
        normal.linthresh = -5


def test_asinh_attributes():
    """Test if the attributes of AsinhNormalizer are added"""
    normal = AsinhNormalizer(linthresh=10)
    assert normal.linthresh == 10
    normal.linthresh = 20
    assert normal.linthresh == 20
    with pytest.raises(ValueError):
        normal.linthresh = -5


def test_zscore_attributes():
    """Test if the attributes of ZScoreNormalizer are added"""
    x = [1, 2, 3, 4, 5]
    normal = ZScoreNormalizer()
    normal.fit(x)
    assert normal.mean_val == 3
    assert normal.std_val == numpy.sqrt(2)


def test_zscore_transform_error():
    """Test if ZScoreNormalizer returns an error without original data"""
    normal = ZScoreNormalizer()
    with pytest.raises(ValueError):
        normal.transform([1, 2])
    with pytest.raises(ValueError):
        normal.inverse_transform([1, 2])
    with pytest.raises(ValueError):
        normal.check_normalize([1, 3])


def test_zscore_refit():
    """Test that mean and std are recaluated"""
    x1 = [1, 2, 3, 4, 5]
    x2 = [1, 2, 3, 4, 5, 6]
    mean2 = numpy.mean(x2)
    std2 = numpy.std(x2)
    normal = ZScoreNormalizer()
    normal.fit(x1)
    normal.fit(x2)
    assert normal.mean_val == mean2
    assert normal.std_val == std2
