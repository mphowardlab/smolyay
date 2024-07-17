import numpy
import pytest

import smolyay
from smolyay.surrogate import (
    TensorProductSurrogate,
    SmolyakSparseProductSurrogate,
)


def create_surrogate(
    surrogate_class,
    basis_function_class,
    nested_point_class,
    num_level,
    domain,
    **kwargs
):
    domain = numpy.array(domain, ndmin=2)
    point_sets = [nested_point_class(dom, num_level) for dom in domain]
    if basis_function_class == smolyay.basis.ChebyshevFirstKind:
        bs = [
            smolyay.basis.NestedBasisFunctionSet(
                [basis_function_class(n) for n in range(len(point_sets[0]))],
                point_sets[0].num_per_level,
            )
            for _ in range(len(point_sets))
        ]
    elif basis_function_class == smolyay.basis.Trigonometric:
        frequencies = []
        for i in range(len(point_sets[0])):
            if i % 2 == 1:
                frequencies.append((1 + i) / 2)
            else:
                frequencies.append(-i / 2)
        bs = [
            smolyay.basis.NestedBasisFunctionSet(
                [basis_function_class(f) for f in frequencies],
                point_sets[0].num_per_level,
            )
            for _ in range(len(point_sets))
        ]
    else:
        raise NotImplementedError
    return surrogate_class(domain, bs, **kwargs), point_sets


def function_1(x):
    """Test function 1."""
    x1, x2 = x
    return numpy.cos(x1) + (2 * x2**2 - 1)


def function_1_gradient(x):
    """Test function 1."""
    x1, x2 = x
    return -numpy.sin(x1), 4 * x2


def function_2(x):
    """Test function 2."""
    return x**2 - 3 * (2 + x) - x


def function_2_gradient(x):
    """Test function 2."""
    return 2 * x - 4


def function_2_gradient(x):
    """Test function 2."""
    return 2 * x - 4


def function_3(x):
    """Test function 3"""
    x1, x2 = x
    return x1 * x2 - 2 * x2


def function_3_gradient(x):
    """Test function 3"""
    x1, x2 = x
    return x1 * x2 - 2 * x2


def function_3_gradient(x):
    """Test function 3 (gradient)."""
    # function f = x1*x2 - 2*x2
    x1, x2 = x
    return x2, x1 - 2


def function_4(x):
    """Test function 4 (gradient)."""
    # function f = x**3 -2*x
    return x**3 - 2 * x


def function_4_gradient(x):
    """Test function 4 (gradient)."""
    # function f = x**3 -2*x
    return 3 * x**2 - 2


def function_5(x):
    """Test function 5 (gradient)."""
    x1, x2 = x
    return numpy.cos(x1) + numpy.sin(x2)


def function_5_gradient(x):
    """Test function 5 (gradient)."""
    x1, x2 = x
    return -numpy.sin(x1), numpy.cos(x2)


def function_6(x):
    """Test function 6 (gradient)."""
    return numpy.cos(x)


def function_6_gradient(x):
    """Test function 6 (gradient)."""
    return -numpy.sin(x)


def branin(x):
    """Branin function."""
    branin1 = (
        x[..., 1]
        - 5.1 * x[..., 0] ** (2) / (4 * numpy.pi**2)
        + 5 * x[..., 0] / (numpy.pi)
        - 6
    ) ** 2
    branin2 = 10 * (1 - 1 / (8 * numpy.pi)) * numpy.cos(x[..., 0])
    branin3 = 10
    branin_function = branin1 + branin2 + branin3
    return branin_function


def branin_gradient(x):
    """Gradient of the branin function."""
    answer = numpy.zeros(numpy.shape(x))
    answer[..., 0] = -10 * (1 - 1 / (8 * numpy.pi)) * numpy.sin(x[..., 0]) + 2 * (
        5 / numpy.pi - 51 * x[..., 0] / (20 * numpy.pi**2)
    ) * (
        -51 * (x[..., 0] ** 2) / (40 * (numpy.pi**2))
        + 5 * x[..., 0] / numpy.pi
        + x[..., 1]
        - 6
    )
    answer[..., 1] = 2 * (
        x[..., 1]
        - 51 * (x[..., 0] ** 2) / (40 * (numpy.pi**2))
        + 5 * x[..., 0] / numpy.pi
        - 6
    )
    return answer


@pytest.mark.parametrize(
    "product_set_surrogate,basis_sets,index_answer",
    [
        (
            TensorProductSurrogate,
            [
                smolyay.basis.BasisFunctionSet(
                    [smolyay.basis.ChebyshevFirstKind(n) for n in range(10)]
                )
                for _ in range(2)
            ],
            numpy.array(numpy.meshgrid(list(range(10)), list(range(10)))).T.reshape(
                -1, 2
            ),
        ),
        (
            TensorProductSurrogate,
            [
                smolyay.basis.BasisFunctionSet(
                    [smolyay.basis.ChebyshevFirstKind(n) for n in range(3)]
                ),
                smolyay.basis.BasisFunctionSet(
                    [smolyay.basis.Trigonometric(n) for n in [0, 1, -1, 2, -2]]
                ),
            ],
            numpy.array(numpy.meshgrid(list(range(3)), list(range(5)))).T.reshape(
                -1, 2
            ),
        ),
        (
            SmolyakSparseProductSurrogate,
            [
                smolyay.basis.NestedBasisFunctionSet(
                    [smolyay.basis.ChebyshevFirstKind(n) for n in range(5)], [1, 2, 2]
                )
                for _ in range(2)
            ],
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [0, 2],
                [3, 0],
                [4, 0],
                [1, 1],
                [1, 2],
                [2, 1],
                [2, 2],
                [0, 3],
                [0, 4],
            ],
        ),
        (
            SmolyakSparseProductSurrogate,
            [
                smolyay.basis.NestedBasisFunctionSet(
                    [smolyay.basis.ChebyshevFirstKind(n) for n in range(3)], [1, 2]
                ),
                smolyay.basis.NestedBasisFunctionSet(
                    [
                        smolyay.basis.Trigonometric(n)
                        for n in [0, 1, -1, 2, -2, 3, -3, 4, -4]
                    ],
                    [1, 2, 6],
                ),
            ],
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [0, 2],
                [1, 1],
                [1, 2],
                [2, 1],
                [2, 2],
                [0, 3],
                [0, 4],
                [0, 5],
                [0, 6],
                [0, 7],
                [0, 8],
            ],
        ),
    ],
    ids=["Tensor", "Tensor mixed basis", "Smolyak", "Smolyak mixed basis"],
)
def test_initialization_product_set(product_set_surrogate, basis_sets, index_answer):
    """Test if class is properly intiallized."""
    domain = [[-5, 10], [0, 15]]
    surrogate = product_set_surrogate(domain, basis_sets, 1e-5, "ridge")
    assert numpy.allclose(surrogate.domain, domain)
    assert basis_sets[0] is surrogate.basis_sets[0]
    assert surrogate.num_dimensions == 2
    assert surrogate.regression
    assert surrogate.alpha == 1e-5
    assert surrogate.regression == "ridge"
    print(surrogate.index_combinations)
    assert numpy.array_equal(surrogate.index_combinations, index_answer)

    surrogate.domain = [[-7, 15], [6, 14]]
    assert numpy.allclose(surrogate.domain, [[-7, 15], [6, 14]])
    surrogate.alpha = 1e-6
    assert surrogate.alpha == 1e-6
    surrogate.regression = "lasso"
    assert surrogate.regression == "lasso"


@pytest.mark.parametrize(
    "product_set_surrogate,basis_sets",
    [
        (
            TensorProductSurrogate,
            [
                smolyay.basis.BasisFunctionSet(
                    [smolyay.basis.ChebyshevFirstKind(n) for n in range(10)]
                )
                for _ in range(2)
            ],
        ),
        (
            SmolyakSparseProductSurrogate,
            [
                smolyay.basis.NestedBasisFunctionSet(
                    [smolyay.basis.ChebyshevFirstKind(n) for n in range(5)], [1, 2, 2]
                )
                for _ in range(2)
            ],
        ),
    ],
    ids=["Tensor", "Smolyak"],
)
def test_regression_error(product_set_surrogate, basis_sets):
    """test error at invalid regression method value"""
    surrogate = product_set_surrogate([[4, 5], [3, 5]], basis_sets)
    with pytest.raises(ValueError):
        surrogate.regression = "not a regression method"


@pytest.mark.parametrize(
    "product_set_surrogate,grid_obj",
    [
        (TensorProductSurrogate, smolyay.samples.TensorProductPointSet),
        (SmolyakSparseProductSurrogate, smolyay.samples.SmolyakSparseProductPointSet),
    ],
    ids=["Tensor", "Smolyak"],
)
@pytest.mark.parametrize(
    "domain",
    [
        [[-1, 1], [-1, 1]],
        [[-5, 5], [-5, 5]],
        [[-7, 7], [-5, 5]],
        [[-5, 10], [-5, 10]],
        [[-5, 10], [0, 15]],
    ],
    ids=[
        "basis",
        "identical and mirrored",
        "different and mirrored",
        "identical",
        "all different",
    ],
)
def test_fit_2D(product_set_surrogate, grid_obj, domain):
    """Test if class is fit."""
    num_level = 5

    surrogate, point_sets = create_surrogate(
        product_set_surrogate,
        smolyay.basis.ChebyshevFirstKind,
        smolyay.samples.NestedClenshawCurtisPointSet,
        num_level,
        domain,
    )
    grid = grid_obj(point_sets=point_sets)
    # fit with same number of points as terms
    branin_output = branin(grid.points)
    surrogate = surrogate.fit(grid, branin_output)
    test_points = smolyay.samples.LatinHypercubeRandomPointSet(domain, 5, 1234).points
    assert numpy.allclose(surrogate.points, grid.points)
    assert numpy.allclose(surrogate.data, branin_output)
    assert numpy.allclose(
        branin(test_points), surrogate.predict(test_points), rtol=1e-3
    )
    assert numpy.allclose(
        branin_gradient(test_points), surrogate.predict_gradient(test_points), rtol=1e-3
    )


@pytest.mark.parametrize(
    "product_set_surrogate,grid_obj",
    [
        (TensorProductSurrogate, smolyay.samples.TensorProductPointSet),
        (SmolyakSparseProductSurrogate, smolyay.samples.SmolyakSparseProductPointSet),
    ],
    ids=["Tensor", "Smolyak"],
)
def test_fit_2D_Trignometric(product_set_surrogate, grid_obj):
    """Test if class is fit."""
    domain = [[0, 2 * numpy.pi], [0, 2 * numpy.pi]]
    num_level = 2

    surrogate, point_sets = create_surrogate(
        product_set_surrogate,
        smolyay.basis.Trigonometric,
        smolyay.samples.NestedTrigonometricPointSet,
        num_level,
        domain,
    )
    grid = grid_obj(point_sets=point_sets)
    # fit with same number of points as terms
    sample_output = [function_5(x) for x in grid.points]
    surrogate = surrogate.fit(grid, sample_output)
    test_points = smolyay.samples.LatinHypercubeRandomPointSet(domain, 5, 1234).points
    assert numpy.allclose(surrogate.points, grid.points)
    assert numpy.allclose(surrogate.data, sample_output)
    assert numpy.allclose(
        [function_5(x) for x in test_points], surrogate.predict(test_points)
    )
    assert numpy.allclose(
        [function_5_gradient(x) for x in test_points],
        surrogate.predict_gradient(test_points),
    )


@pytest.mark.parametrize(
    "product_set_surrogate,grid_obj",
    [
        (TensorProductSurrogate, smolyay.samples.TensorProductPointSet),
        (SmolyakSparseProductSurrogate, smolyay.samples.SmolyakSparseProductPointSet),
    ],
    ids=["Tensor", "Smolyak"],
)
def test_fit_1D_Trignometric(product_set_surrogate, grid_obj):
    """Test if class is fit."""
    domain = [0, 2 * numpy.pi]
    num_level = 2

    surrogate, point_sets = create_surrogate(
        product_set_surrogate,
        smolyay.basis.Trigonometric,
        smolyay.samples.NestedTrigonometricPointSet,
        num_level,
        domain,
    )
    grid = grid_obj(point_sets=point_sets)
    # fit with same number of points as terms
    sample_output = [function_6(x) for x in grid.points]
    surrogate = surrogate.fit(grid, sample_output)
    test_points = smolyay.samples.LatinHypercubeRandomPointSet(domain, 5, 1234).points
    assert numpy.allclose(surrogate.points, grid.points)
    assert numpy.allclose(surrogate.data, sample_output)
    assert numpy.allclose(
        numpy.squeeze(function_6(test_points)), surrogate.predict(test_points)
    )
    assert numpy.allclose(
        [function_6_gradient(x) for x in test_points],
        surrogate.predict_gradient(test_points),
    )


@pytest.mark.parametrize(
    "product_set_surrogate,grid_obj",
    [
        (TensorProductSurrogate, smolyay.samples.TensorProductPointSet),
        (SmolyakSparseProductSurrogate, smolyay.samples.SmolyakSparseProductPointSet),
    ],
    ids=["Tensor", "Smolyak"],
)
def test_fit_2D_mixed_basis(product_set_surrogate, grid_obj):
    """Test if class is fit."""
    domain = [[-1, 1], [0, 2 * numpy.pi]]
    num_level = 3
    point_sets = [
        smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], num_level),
        smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], num_level),
    ]
    num_trig = numpy.arange(len(point_sets[1]) - 2, dtype=int)
    frequencies = numpy.where(num_trig % 2 == 1, (1 + num_trig) / 2, -num_trig / 2)
    basis_sets = [
        smolyay.basis.NestedBasisFunctionSet(
            [smolyay.basis.ChebyshevFirstKind(n) for n in range(len(point_sets[0]))],
            point_sets[0].num_per_level,
        ),
        smolyay.basis.NestedBasisFunctionSet(
            [smolyay.basis.Trigonometric(n) for n in frequencies]
            + [
                smolyay.basis.ChebyshevFirstKind(1),
                smolyay.basis.ChebyshevFirstKind(2),
            ],
            point_sets[1].num_per_level,
        ),
    ]
    surrogate = product_set_surrogate(domain, basis_sets)
    grid = grid_obj(point_sets=point_sets)
    # fit with same number of points as terms
    sample_output = [function_1(x) for x in grid.points]
    surrogate = surrogate.fit(grid, sample_output)
    test_points = smolyay.samples.LatinHypercubeRandomPointSet(domain, 5, 1234).points
    y = numpy.array([function_1(x) for x in test_points])
    y1 = surrogate.predict(test_points)
    assert numpy.allclose(surrogate.points, grid.points)
    assert numpy.allclose(surrogate.data, sample_output)
    assert numpy.allclose(
        [function_1(x) for x in test_points], surrogate.predict(test_points), rtol=0.01
    )
    assert numpy.allclose(
        [function_1_gradient(x) for x in test_points],
        surrogate.predict_gradient(test_points),
        rtol=0.01,
    )


@pytest.mark.parametrize(
    "product_set_surrogate",
    [TensorProductSurrogate, SmolyakSparseProductSurrogate],
    ids=["Tensor", "Smolyak"],
)
@pytest.mark.parametrize(
    "domain",
    [
        [-1, 1],
        [-5, 5],
        [-5, 10],
    ],
    ids=[
        "basis",
        "mirrored",
        "all different",
    ],
)
def test_fit_1D(product_set_surrogate, domain):
    num_level = 2
    # fit with a 1D function
    surrogate, point_sets = create_surrogate(
        product_set_surrogate,
        smolyay.basis.ChebyshevFirstKind,
        smolyay.samples.NestedClenshawCurtisPointSet,
        num_level,
        domain,
    )
    grid_points = numpy.array(point_sets[0].points, ndmin=2).reshape((-1, 1))
    fun2_output = function_2(grid_points)
    test_points = smolyay.samples.LatinHypercubeRandomPointSet(domain, 5, 1234).points
    fun2_gradient_output = numpy.array(function_2_gradient(test_points), ndmin=2)
    surrogate.fit(grid_points, fun2_output)
    assert numpy.allclose(surrogate.points, grid_points)
    assert numpy.allclose(surrogate.data, fun2_output)
    assert numpy.allclose(
        numpy.squeeze(function_2(test_points)), surrogate.predict(test_points)
    )
    assert numpy.allclose(fun2_gradient_output, surrogate.predict_gradient(test_points))


@pytest.mark.parametrize(
    "product_set_surrogate",
    [TensorProductSurrogate, SmolyakSparseProductSurrogate],
    ids=["Tensor", "Smolyak"],
)
@pytest.mark.parametrize(
    "regression,points",
    [("ridge", 1500), ("lasso", 2500), ("lstsq", 1500)],
    ids=["Ridge", "Lasso", "Least Squares"],
)
def test_fit_latin_2D(product_set_surrogate, regression, points):
    """Test if class is fit when number of terms doesn't match samples."""
    domain = [[-5, 5], [0, 10]]
    num_level = 4

    surrogate, _ = create_surrogate(
        product_set_surrogate,
        smolyay.basis.ChebyshevFirstKind,
        smolyay.samples.NestedClenshawCurtisPointSet,
        num_level,
        domain,
        regression=regression,
    )
    # fit to a 2D function with a different number of points as terms
    test_points = numpy.array([[-0.5, 0.8], [1, 1], [0.7, 0.9]])
    grid = smolyay.samples.LatinHypercubeRandomPointSet(domain, points, 1234)
    surrogate.fit(grid, branin(grid.points))
    assert numpy.allclose(surrogate.points, grid.points)
    assert numpy.allclose(
        branin(test_points), surrogate.predict(test_points), rtol=0.01
    )
    assert numpy.allclose(
        branin_gradient(test_points), surrogate.predict_gradient(test_points), rtol=0.01
    )


@pytest.mark.parametrize(
    "product_set_surrogate",
    [TensorProductSurrogate, SmolyakSparseProductSurrogate],
    ids=["Tensor", "Smolyak"],
)
@pytest.mark.parametrize(
    "regression", ["ridge", "lasso", "lstsq"], ids=["Ridge", "Lasso", "Least Squares"]
)
def test_fit_latin_1D(product_set_surrogate, regression):
    """Test if class is fit when number of terms doesn't match samples."""
    domain = [-5, 10]
    num_level = 4

    # fit with a 1D function
    surrogate, _ = create_surrogate(
        product_set_surrogate,
        smolyay.basis.ChebyshevFirstKind,
        smolyay.samples.NestedClenshawCurtisPointSet,
        num_level,
        domain=domain,
        regression=regression,
    )
    grid = smolyay.samples.LatinHypercubeRandomPointSet(domain, 100, 1234)
    fun2_output = function_2(grid.points)
    test_points = numpy.array([1, 2, 3], ndmin=2).reshape((-1, 1))
    fun2_gradient_output = numpy.array(function_2_gradient(test_points), ndmin=2)
    surrogate.fit(grid.points, fun2_output)
    assert numpy.allclose(surrogate.points, grid.points)
    assert numpy.allclose(surrogate.data, fun2_output)
    assert numpy.allclose(
        numpy.squeeze(function_2(test_points)),
        surrogate.predict(test_points),
        rtol=0.01,
    )
    assert numpy.allclose(
        fun2_gradient_output,
        surrogate.predict_gradient(test_points),
        rtol=0.01,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "product_set_surrogate",
    [
        (TensorProductSurrogate),
        (SmolyakSparseProductSurrogate),
    ],
    ids=["Tensor", "Smolyak"],
)
def test_fit_error(product_set_surrogate):
    """Test if fit raises an error if points are outside domain."""
    domain = [[-5, 10], [0, 15]]
    bs = [
        smolyay.basis.NestedBasisFunctionSet(
            [smolyay.basis.ChebyshevFirstKind(n) for n in range(5)],
            [1, 2, 2],
        )
        for _ in range(2)
    ]
    surrogate = product_set_surrogate(domain, bs)

    with pytest.raises(ValueError):
        test_points = numpy.array([[-0.5, 0.8], [0, 0], [0.7, -0.2]])
        surrogate.fit(test_points, branin(test_points))
    with pytest.raises(ValueError):
        test_points = numpy.array([[-0.5, 0.8], [-7, 0], [0.7, 0.2]])
        surrogate.fit(test_points, branin(test_points))
    with pytest.raises(ValueError):
        test_points = numpy.array([[-0.5, 18], [0, 0], [0.7, 0.2]])
        surrogate.fit(test_points, branin(test_points))
    with pytest.raises(ValueError):
        test_points = numpy.array([[-0.5, 0.8], [0, 0], [25, 0.2]])
        surrogate.fit(test_points, branin(test_points))


@pytest.mark.parametrize(
    "product_set_surrogate,grid_obj",
    [
        (TensorProductSurrogate, smolyay.samples.TensorProductPointSet),
        (SmolyakSparseProductSurrogate, smolyay.samples.SmolyakSparseProductPointSet),
    ],
    ids=["Tensor", "Smolyak"],
)
@pytest.mark.parametrize(
    "domain",
    [
        [[-1, 1], [-1, 1]],
        [[-5, 5], [-5, 5]],
        [[-9, 9], [-5, 5]],
        [[-5, 10], [-5, 10]],
        [[-5, 10], [0, 15]],
    ],
    ids=[
        "basis",
        "identical and mirrored",
        "different and mirrored",
        "identical",
        "all different",
    ],
)
def test_fit_gradient_2D(product_set_surrogate, grid_obj, domain):
    """Test if class is fit using the gradient."""
    num_level = 3
    surrogate, point_sets = create_surrogate(
        product_set_surrogate,
        smolyay.basis.ChebyshevFirstKind,
        smolyay.samples.NestedClenshawCurtisPointSet,
        num_level,
        domain,
    )
    grid = grid_obj(point_sets=point_sets)
    # fit with same number of points as terms
    fun3_gradient_samples = [function_3_gradient(x) for x in grid.points]
    surrogate = surrogate.fit_gradient(grid, fun3_gradient_samples)
    test_points = smolyay.samples.LatinHypercubeRandomPointSet(domain, 5, 1234).points
    fun3_output = [function_3(x) for x in test_points]
    fun3_gradient_output = [function_3_gradient(x) for x in test_points]
    if numpy.allclose(domain,[[-9,9],[-5,5]]):
        assert numpy.allclose(function_3_gradient((8, 0.75)), surrogate.predict_gradient([(8, 0.75)]))
    assert numpy.allclose(surrogate.points, grid.points)
    assert numpy.allclose(surrogate.data, fun3_gradient_samples)
    # assert numpy.allclose(fun3_output, surrogate.predict(test_points), rtol=0.1)
    assert numpy.allclose(fun3_gradient_output, surrogate.predict_gradient(test_points))


@pytest.mark.parametrize(
    "product_set_surrogate,grid_obj",
    [
        (TensorProductSurrogate, smolyay.samples.TensorProductPointSet),
        (SmolyakSparseProductSurrogate, smolyay.samples.SmolyakSparseProductPointSet),
    ],
    ids=["Tensor", "Smolyak"],
)
def test_fit_gradient_2D_Trignometric(product_set_surrogate, grid_obj):
    """Test if class is fit."""
    domain = [[0, 2 * numpy.pi], [0, 2 * numpy.pi]]
    num_level = 2

    surrogate, point_sets = create_surrogate(
        product_set_surrogate,
        smolyay.basis.Trigonometric,
        smolyay.samples.NestedTrigonometricPointSet,
        num_level,
        domain,
    )
    grid = grid_obj(point_sets=point_sets)
    # fit with same number of points as terms
    sample_output = [function_5_gradient(x) for x in grid.points]
    surrogate = surrogate.fit_gradient(grid, sample_output)
    test_points = smolyay.samples.LatinHypercubeRandomPointSet(domain, 5, 1234).points
    assert numpy.allclose(surrogate.points, grid.points)
    assert numpy.allclose(surrogate.data, sample_output)
    # assert numpy.allclose(
    #    [function_5(x) for x in test_points], surrogate.predict(test_points), rtol=0.1
    # )
    assert numpy.allclose(
        [function_5_gradient(x) for x in test_points],
        surrogate.predict_gradient(test_points),
    )


@pytest.mark.parametrize(
    "product_set_surrogate,grid_obj",
    [
        (TensorProductSurrogate, smolyay.samples.TensorProductPointSet),
        (SmolyakSparseProductSurrogate, smolyay.samples.SmolyakSparseProductPointSet),
    ],
    ids=["Tensor", "Smolyak"],
)
def test_fit_gradient_2D_mixed_basis(product_set_surrogate, grid_obj):
    """Test if class is fit."""
    domain = [[-1, 1], [0, 2 * numpy.pi]]
    num_level = 3
    point_sets = [
        smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], num_level),
        smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], num_level),
    ]
    num_trig = numpy.arange(len(point_sets[1]) - 2, dtype=int)
    frequencies = numpy.where(num_trig % 2 == 1, (1 + num_trig) / 2, -num_trig / 2)
    basis_sets = [
        smolyay.basis.NestedBasisFunctionSet(
            [smolyay.basis.ChebyshevFirstKind(n) for n in range(len(point_sets[0]))],
            point_sets[0].num_per_level,
        ),
        smolyay.basis.NestedBasisFunctionSet(
            [smolyay.basis.Trigonometric(n) for n in frequencies]
            + [
                smolyay.basis.ChebyshevFirstKind(1),
                smolyay.basis.ChebyshevFirstKind(2),
            ],
            point_sets[1].num_per_level,
        ),
    ]
    surrogate = product_set_surrogate(domain, basis_sets)
    grid = grid_obj(point_sets=point_sets)
    # fit with same number of points as terms
    sample_output = [function_1_gradient(x) for x in grid.points]
    surrogate = surrogate.fit_gradient(grid, sample_output)
    test_points = smolyay.samples.LatinHypercubeRandomPointSet(domain, 5, 1234).points
    assert numpy.allclose(surrogate.points, grid.points)
    assert numpy.allclose(surrogate.data, sample_output)
    # assert numpy.allclose(
    #    [function_1(x) for x in test_points], surrogate.predict(test_points), rtol=0.1
    # )
    assert numpy.allclose(
        [function_1_gradient(x) for x in test_points],
        surrogate.predict_gradient(test_points),
        rtol=0.01,
    )


@pytest.mark.parametrize(
    "product_set_surrogate",
    [TensorProductSurrogate, SmolyakSparseProductSurrogate],
    ids=["Tensor", "Smolyak"],
)
@pytest.mark.parametrize(
    "domain",
    [
        [-1, 1],
        [-5, 5],
        [-5, 10],
    ],
    ids=[
        "basis",
        "mirrored",
        "all different",
    ],
)
def test_fit_gradient_1D(product_set_surrogate, domain):
    num_level = 3
    surrogate, point_sets = create_surrogate(
        product_set_surrogate,
        smolyay.basis.ChebyshevFirstKind,
        smolyay.samples.NestedClenshawCurtisPointSet,
        num_level,
        domain,
    )
    # fit with a 1D function
    grid_points = numpy.array(point_sets[0].points, ndmin=2).reshape((-1, 1))
    fun4_gradient_samples = function_4_gradient(grid_points)
    surrogate.fit_gradient(grid_points, fun4_gradient_samples)
    test_points = numpy.array([0.1, 0.2, 0.3], ndmin=2).reshape((-1, 1))
    assert numpy.allclose(surrogate.points, grid_points)
    assert numpy.allclose(surrogate.data, fun4_gradient_samples)
    # assert numpy.allclose(
    #    numpy.squeeze(function_4(test_points)), surrogate.predict(test_points), rtol=0.1
    # )
    assert numpy.allclose(
        function_4_gradient(test_points),
        surrogate.predict_gradient(test_points),
    )


@pytest.mark.parametrize(
    "product_set_surrogate",
    [TensorProductSurrogate, SmolyakSparseProductSurrogate],
    ids=["Tensor", "Smolyak"],
)
@pytest.mark.parametrize(
    "regression", ["ridge", "lasso", "lstsq"], ids=["Ridge", "Lasso", "Least Squares"]
)
def test_fit_gradient_latin_2D(product_set_surrogate, regression):
    """Test if class is fit using gradient when number of terms != number of points."""
    domain = [[-5, 5], [-1, 1]]
    num_level = 3
    surrogate, _ = create_surrogate(
        product_set_surrogate,
        smolyay.basis.ChebyshevFirstKind,
        smolyay.samples.NestedClenshawCurtisPointSet,
        num_level,
        domain,
        regression=regression,
    )
    # fit to a 2D function with a different number of points as terms
    grid = smolyay.samples.LatinHypercubeRandomPointSet(domain, 1000, 1234)
    fun3_gradient_samples = [function_3_gradient(x) for x in grid.points]
    surrogate.fit_gradient(grid, fun3_gradient_samples)
    test_points = numpy.array([[-0.5, 0.8], [0, 0], [0.7, 0.2]])
    fun3_output = [function_3(x) for x in test_points]
    fun3_gradient_output = [function_3_gradient(x) for x in test_points]
    assert numpy.allclose(surrogate.points, grid.points)
    assert numpy.allclose(surrogate.data, fun3_gradient_samples)
    # assert numpy.allclose(
    #    fun3_output, surrogate.predict(test_points), rtol=0.1, atol=1e-3
    # )
    assert numpy.allclose(
        fun3_gradient_output,
        surrogate.predict_gradient(test_points),
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "product_set_surrogate",
    [TensorProductSurrogate, SmolyakSparseProductSurrogate],
    ids=["Tensor", "Smolyak"],
)
@pytest.mark.parametrize(
    "regression", ["ridge", "lasso", "lstsq"], ids=["Ridge", "Lasso", "Least Squares"]
)
def test_fit_gradient_latin_1D(product_set_surrogate, regression):
    num_level = 3
    domain = [-5, 6]
    # fit with a 1D function
    grid = smolyay.samples.LatinHypercubeRandomPointSet(domain, 100, 1234)
    surrogate, _ = create_surrogate(
        product_set_surrogate,
        smolyay.basis.ChebyshevFirstKind,
        smolyay.samples.NestedClenshawCurtisPointSet,
        num_level,
        domain,
        regression=regression,
    )
    fun4_gradient_samples = function_4_gradient(grid.points)
    surrogate.fit_gradient(grid, fun4_gradient_samples)
    test_points = numpy.array([0.1, 0.2, 0.3], ndmin=2).reshape((-1, 1))
    assert numpy.allclose(surrogate.points, grid.points)
    assert numpy.allclose(surrogate.data, fun4_gradient_samples)
    # assert numpy.allclose(
    #    numpy.squeeze(function_4(test_points)),
    #    surrogate.predict(test_points),
    #    rtol=0.1,
    #    atol=1e-3,
    # )
    assert numpy.allclose(
        function_4_gradient(test_points), surrogate.predict_gradient(test_points)
    )


@pytest.mark.parametrize(
    "product_set_surrogate",
    [TensorProductSurrogate, SmolyakSparseProductSurrogate],
    ids=["Tensor", "Smolyak"],
)
def test_fit_gradient_error(product_set_surrogate):
    """Test if fit_gradient raises an error if points are outside domain."""
    domain = [[-5, 10], [0, 15]]
    bs = [
        smolyay.basis.NestedBasisFunctionSet(
            [smolyay.basis.ChebyshevFirstKind(n) for n in range(5)],
            [1, 2, 2],
        )
        for _ in range(2)
    ]
    surrogate = product_set_surrogate(domain, bs)
    with pytest.raises(IndexError):
        test_points = numpy.array([[-0.5, 0.8], [0, 0], [0.7, -0.2]])
        surrogate.fit_gradient(test_points, branin(test_points))
    with pytest.raises(ValueError):
        test_points = numpy.array([[-0.5, 0.8], [0, 0], [0.7, -0.2]])
        fun3_gradient_output = [function_3_gradient(x) for x in test_points]
        surrogate.fit_gradient(test_points, fun3_gradient_output)
    with pytest.raises(ValueError):
        test_points = numpy.array([[-0.5, 0.8], [-7, 0], [0.7, 0.2]])
        fun3_gradient_output = [function_3_gradient(x) for x in test_points]
        surrogate.fit_gradient(test_points, fun3_gradient_output)
    with pytest.raises(ValueError):
        test_points = numpy.array([[-0.5, 18], [0, 0], [0.7, 0.2]])
        fun3_gradient_output = [function_3_gradient(x) for x in test_points]
        surrogate.fit_gradient(test_points, fun3_gradient_output)
    with pytest.raises(ValueError):
        test_points = numpy.array([[-0.5, 0.8], [0, 0], [25, 0.2]])
        fun3_gradient_output = [function_3_gradient(x) for x in test_points]
        surrogate.fit_gradient(test_points, fun3_gradient_output)


@pytest.mark.parametrize(
    "product_set_surrogate,grid_obj",
    [
        (TensorProductSurrogate, smolyay.samples.TensorProductPointSet),
        (SmolyakSparseProductSurrogate, smolyay.samples.SmolyakSparseProductPointSet),
    ],
    ids=["Tensor", "Smolyak"],
)
def test_predict_size_2D(product_set_surrogate, grid_obj):
    """Test predict returns answer of the appropriate shape"""
    domain = [[-5, 10], [0, 15]]
    num_level = 5
    surrogate, point_sets = create_surrogate(
        product_set_surrogate,
        smolyay.basis.ChebyshevFirstKind,
        smolyay.samples.NestedClenshawCurtisPointSet,
        num_level,
        domain,
    )
    grid = grid_obj(point_sets=point_sets)
    # fit with same number of points as terms
    surrogate.fit(grid, branin(grid.points))
    assert numpy.array_equal(
        numpy.shape(surrogate.predict([[-0.5, 0.8], [0, 0], [0.7, 0]])), (3,)
    )
    assert numpy.array_equal(numpy.shape(surrogate.predict([[0.7, 0]])), ())


@pytest.mark.parametrize(
    "product_set_surrogate",
    [TensorProductSurrogate, SmolyakSparseProductSurrogate],
    ids=["Tensor", "Smolyak"],
)
def test_predict_size_1D(product_set_surrogate):
    num_level = 4
    domain = [-5, 5]
    # fit with a 1D function
    surrogate, point_sets = create_surrogate(
        product_set_surrogate,
        smolyay.basis.ChebyshevFirstKind,
        smolyay.samples.NestedClenshawCurtisPointSet,
        num_level,
        domain,
    )
    grid_points = numpy.array(point_sets[0].points, ndmin=2).reshape((-1, 1))
    fun2_output = function_2(grid_points)
    surrogate.fit(grid_points, fun2_output)
    assert numpy.array_equal(numpy.shape(surrogate.predict([[-0.5], [0], [0.7]])), (3,))
    assert numpy.array_equal(numpy.shape(surrogate.predict([[0.7]])), ())


@pytest.mark.parametrize(
    "product_set_surrogate,basis_sets",
    [
        (
            TensorProductSurrogate,
            [
                smolyay.basis.BasisFunctionSet(
                    [smolyay.basis.ChebyshevFirstKind(n) for n in range(10)]
                )
                for _ in range(2)
            ],
        ),
        (
            SmolyakSparseProductSurrogate,
            [
                smolyay.basis.NestedBasisFunctionSet(
                    [smolyay.basis.ChebyshevFirstKind(n) for n in range(5)], [1, 2, 2]
                )
                for _ in range(2)
            ],
        ),
    ],
    ids=["Tensor", "Smolyak"],
)
def test_predict_error(product_set_surrogate, basis_sets):
    """Test that predict raises correct errors"""
    surrogate = product_set_surrogate([[-5, 10], [0, 15]], basis_sets)
    grid = smolyay.samples.LatinHypercubeRandomPointSet([[-5, 10], [0, 15]], 1500, 1234)
    with pytest.raises(RuntimeError):
        surrogate.predict([[0.7, 0.3]])
    with pytest.raises(ValueError):
        surrogate.fit(grid, branin(grid.points))
        surrogate.predict([[11, 5]])
    with pytest.raises(ValueError):
        surrogate.fit(grid, branin(grid.points))
        surrogate.predict([[5, 4], [3, 20], [0, 5]])
    with pytest.raises(ValueError):
        surrogate.fit(grid, branin(grid.points))
        surrogate.predict([[-19, 5], [3, 3]])
    with pytest.raises(ValueError):
        surrogate.fit(grid, branin(grid.points))
        surrogate.predict([[-4, -1], [3, 3]])
    with pytest.raises(NotImplementedError):
        surrogate.fit_gradient(grid, branin_gradient(grid.points))
        surrogate.predict([[9, 5]])

    # ensure using fit after fit_gradient resets flag
    surrogate.fit_gradient(grid, branin_gradient(grid.points))
    surrogate.fit(grid, branin(grid.points))
    surrogate.predict([[8, 5]])


@pytest.mark.parametrize(
    "product_set_surrogate,grid_obj",
    [
        (TensorProductSurrogate, smolyay.samples.TensorProductPointSet),
        (SmolyakSparseProductSurrogate, smolyay.samples.SmolyakSparseProductPointSet),
    ],
    ids=["Tensor", "Smolyak"],
)
def test_predict_gradient_size_2D(product_set_surrogate, grid_obj):
    """Test predict_gradient returns answer of the appropriate shape."""
    domain = [[-5, 10], [0, 15]]
    num_level = 5

    surrogate, point_sets = create_surrogate(
        product_set_surrogate,
        smolyay.basis.ChebyshevFirstKind,
        smolyay.samples.NestedClenshawCurtisPointSet,
        num_level,
        domain,
    )
    grid = grid_obj(point_sets=point_sets)
    # fit with same number of points as terms
    surrogate.fit(grid, branin(grid.points))
    assert numpy.array_equal(
        numpy.shape(surrogate.predict_gradient([[-0.5, 0.8], [0, 0], [0.7, 0]])), (3, 2)
    )
    assert numpy.array_equal(
        numpy.shape(surrogate.predict_gradient([[0.7, 0]])), (1, 2)
    )


@pytest.mark.parametrize(
    "product_set_surrogate",
    [TensorProductSurrogate, SmolyakSparseProductSurrogate],
    ids=["Tensor", "Smolyak"],
)
def test_predict_gradient_size_1D(product_set_surrogate):
    num_level = 4
    domain = [-5, 5]
    surrogate, point_sets = create_surrogate(
        product_set_surrogate,
        smolyay.basis.ChebyshevFirstKind,
        smolyay.samples.NestedClenshawCurtisPointSet,
        num_level,
        domain,
    )
    # fit with a 1D function
    grid_points = numpy.array(point_sets[0].points, ndmin=2).reshape((-1, 1))
    fun2_output = function_2(grid_points)
    surrogate.fit(grid_points, fun2_output)
    assert numpy.array_equal(
        numpy.shape(surrogate.predict_gradient([[-0.5], [0], [0.7]])), (3, 1)
    )
    assert numpy.array_equal(numpy.shape(surrogate.predict_gradient([[0.7]])), ())


@pytest.mark.parametrize(
    "product_set_surrogate,basis_sets",
    [
        (
            TensorProductSurrogate,
            [
                smolyay.basis.BasisFunctionSet(
                    [smolyay.basis.ChebyshevFirstKind(n) for n in range(10)]
                )
                for _ in range(2)
            ],
        ),
        (
            SmolyakSparseProductSurrogate,
            [
                smolyay.basis.NestedBasisFunctionSet(
                    [smolyay.basis.ChebyshevFirstKind(n) for n in range(5)], [1, 2, 2]
                )
                for _ in range(2)
            ],
        ),
    ],
    ids=["Tensor", "Smolyak"],
)
def test_predict_gradient_error(product_set_surrogate, basis_sets):
    """Test that predict_gradient raises correct errors"""
    surrogate = product_set_surrogate([[-5, 10], [0, 15]], basis_sets)
    grid = smolyay.samples.LatinHypercubeRandomPointSet([[-5, 10], [0, 15]], 1500, 1234)
    with pytest.raises(RuntimeError):
        surrogate.predict_gradient([[0.7, 0.3]])
    with pytest.raises(ValueError):
        surrogate.fit(grid, branin(grid.points))
        surrogate.predict_gradient([[11, 5]])
        surrogate.fit(grid, branin(grid.points))
        surrogate.predict_gradient([[11, 5]])
    with pytest.raises(ValueError):
        surrogate.fit(grid, branin(grid.points))
        surrogate.predict_gradient([[5, 4], [3, 20], [0, 5]])
        surrogate.fit(grid, branin(grid.points))
        surrogate.predict_gradient([[5, 4], [3, 20], [0, 5]])
    with pytest.raises(ValueError):
        surrogate.fit(grid, branin(grid.points))
        surrogate.predict_gradient([[-19, 5], [3, 3]])
        surrogate.fit(grid, branin(grid.points))
        surrogate.predict_gradient([[-19, 5], [3, 3]])
    with pytest.raises(ValueError):
        surrogate.fit(grid, branin(grid.points))
        surrogate.predict_gradient([[-4, -1], [3, 3]])
