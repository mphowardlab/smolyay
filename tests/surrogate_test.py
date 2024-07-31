import numpy
import pytest

import smolyay


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
            smolyay.basis.NestedClenshawCurtisBasisFunctionSet(num_level)
            for _ in range(len(point_sets))
        ]
    elif basis_function_class == smolyay.basis.Trigonometric:
        bs = [
            smolyay.basis.NestedTrigonometricBasisFunctionSet(num_level)
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
    """Test function 1 (gradient)."""
    x1, x2 = x
    return -numpy.sin(x1), 4 * x2


def function_1_hessian(x):
    """Test function 1 (hessian)."""
    x1, x2 = x
    return [[-numpy.cos(x1), 0], [0, 4]]


def function_2(x):
    """Test function 2."""
    return x**3 - 3 * (2 + x) - x


def function_2_gradient(x):
    """Test function 2 (gradient)."""
    return 3 * x**2 - 4


def function_2_hessian(x):
    """Test function 2 (hessian)."""
    return 6 * x


def function_3(x):
    """Test function 3"""
    x1, x2 = x
    return x1 * x2 - 2 * x2


def function_3_gradient(x):
    """Test function 3 (gradient)."""
    x1, x2 = x
    return x2, x1 - 2


def function_3_hessian(x):
    """Test function 3 (hessian)."""
    return [[0, 1], [1, 0]]


def function_4(x):
    """Test function 4."""
    x1, x2 = x
    return numpy.cos(x1) + numpy.sin(x2)


def function_4_gradient(x):
    """Test function 4 (gradient)."""
    x1, x2 = x
    return -numpy.sin(x1), numpy.cos(x2)


def function_4_hessian(x):
    """Test function 4 (hessian)."""
    x1, x2 = x
    return [[-numpy.cos(x1), 0], [0, -numpy.sin(x2)]]


def function_5(x):
    """Test function 5."""
    return numpy.cos(x)


def function_5_gradient(x):
    """Test function 5 (gradient)."""
    return -numpy.sin(x)


def function_5_hessian(x):
    """Test function 5 (hessian)."""
    return -numpy.cos(x)


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


def branin_hessian(x):
    """Hessian matrix of the branin function."""
    answer = numpy.zeros(list(numpy.shape(x)) + [numpy.shape(x)[-1]])
    # d2f/dx2
    answer[:, 0, 0] = -(
        (4000 * (numpy.pi**4) - 500 * (numpy.pi**3)) * numpy.cos(x[:, 0])
        - 7803 * (x[:, 0] ** 2)
        + 30600 * numpy.pi * x[:, 0]
        + 2040 * (numpy.pi**2) * x[:, 1]
        - 32240 * (numpy.pi**2)
    ) / (400 * numpy.pi**4)
    # d2f/dy2
    answer[:, 1, 1] = 2
    # d2f/dxdy
    answer[:, 0, 1] = answer[:, 1, 0] = 2 * (
        -51 * (x[:, 0]) / (20 * (numpy.pi**2)) + 5 / numpy.pi
    )
    return answer


@pytest.mark.parametrize(
    "surrogate_class,basis_sets,index_answer",
    [
        (
            smolyay.surrogate.TensorProductSurrogate,
            [
                smolyay.basis.ChebyshevFirstKindBasisFunctionSet(10),
                smolyay.basis.ChebyshevFirstKindBasisFunctionSet(10),
            ],
            numpy.array(numpy.meshgrid(list(range(10)), list(range(10)))).T.reshape(
                -1, 2
            ),
        ),
        (
            smolyay.surrogate.TensorProductSurrogate,
            [
                smolyay.basis.ChebyshevFirstKindBasisFunctionSet(3),
                smolyay.basis.TrigonometricBasisFunctionSet(5),
            ],
            numpy.array(numpy.meshgrid(list(range(3)), list(range(5)))).T.reshape(
                -1, 2
            ),
        ),
        (
            smolyay.surrogate.SmolyakSparseProductSurrogate,
            [
                smolyay.basis.NestedClenshawCurtisBasisFunctionSet(3),
                smolyay.basis.NestedClenshawCurtisBasisFunctionSet(3),
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
            smolyay.surrogate.SmolyakSparseProductSurrogate,
            [
                smolyay.basis.NestedClenshawCurtisBasisFunctionSet(2),
                smolyay.basis.NestedTrigonometricBasisFunctionSet(3),
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
def test_initialization_product_set(surrogate_class, basis_sets, index_answer):
    """Test if class is properly intiallized."""
    domain = [[-5, 10], [0, 15]]
    surrogate = surrogate_class(domain, basis_sets)
    assert numpy.allclose(surrogate.domain, domain)
    assert basis_sets[0] is surrogate.basis_sets[0]
    assert surrogate.num_dimensions == 2
    assert surrogate.regularization == None
    surrogate.fit(domain, [1, 2])
    assert numpy.array_equal(surrogate._index_combinations, index_answer)

    # test optional parameters
    surrogate = surrogate_class(
        domain, basis_sets, smolyay.surrogate.L2Regularization(alpha=1e-5)
    )
    assert isinstance(surrogate.regularization, smolyay.surrogate.L2Regularization)
    assert surrogate.regularization.alpha == 1e-5

    # test setting parameters
    surrogate.domain = [[-7, 15], [6, 14]]
    assert numpy.allclose(surrogate.domain, [[-7, 15], [6, 14]])
    surrogate.regularization = smolyay.surrogate.L1Regularization(alpha=1e-10)
    assert isinstance(surrogate.regularization, smolyay.surrogate.L1Regularization)
    assert surrogate.regularization.alpha == 1e-10


@pytest.mark.parametrize(
    "surrogate_class,basis_sets",
    [
        (
            smolyay.surrogate.TensorProductSurrogate,
            [
                smolyay.basis.ChebyshevFirstKindBasisFunctionSet(10),
                smolyay.basis.ChebyshevFirstKindBasisFunctionSet(10),
            ],
        ),
        (
            smolyay.surrogate.SmolyakSparseProductSurrogate,
            [
                smolyay.basis.NestedClenshawCurtisBasisFunctionSet(3),
                smolyay.basis.NestedClenshawCurtisBasisFunctionSet(3),
            ],
        ),
    ],
    ids=["Tensor", "Smolyak"],
)
def test_regularization_error(surrogate_class, basis_sets):
    """test error at invalid regularization method value"""
    surrogate = surrogate_class([[4, 5], [3, 5]], basis_sets)
    with pytest.raises(ValueError):
        surrogate.regularization = "not a regularization method"


@pytest.mark.parametrize(
    "surrogate_class,grid_obj",
    [
        (
            smolyay.surrogate.TensorProductSurrogate,
            smolyay.samples.TensorProductPointSet,
        ),
        (
            smolyay.surrogate.SmolyakSparseProductSurrogate,
            smolyay.samples.SmolyakSparseProductPointSet,
        ),
    ],
    ids=["Tensor", "Smolyak"],
)
@pytest.mark.incremental
class TestFit2D:

    def test_fit_2D(self, surrogate_class, grid_obj):
        """Test if class is fit to 2D function."""
        num_level = 5
        domain = [[-1, 1], [-1, 1]]
        surrogate, point_sets = create_surrogate(
            surrogate_class,
            smolyay.basis.ChebyshevFirstKind,
            smolyay.samples.NestedClenshawCurtisPointSet,
            num_level,
            domain,
        )
        grid = grid_obj(point_sets=point_sets)
        sample_output = branin(grid.points)
        surrogate = surrogate.fit(grid, sample_output)

        # test surrogate matches at some points
        test_points = smolyay.samples.LatinHypercubeRandomPointSet(
            domain, 5, 1234
        ).points
        predict_answer = branin(test_points)
        gradient_answer = branin_gradient(test_points)
        hessian_answer = branin_hessian(test_points)
        assert numpy.allclose(predict_answer, surrogate.predict(test_points), rtol=1e-3)
        assert numpy.allclose(
            gradient_answer, surrogate.predict_gradient(test_points), rtol=1e-3
        )
        assert numpy.allclose(
            hessian_answer, surrogate.predict_hessian(test_points), rtol=1e-3
        )

    def test_fit_2D_domain_shift(self, surrogate_class, grid_obj):
        """Test if class is fit to 2D function with different domain as basis."""
        num_level = 5
        domain = [[-1, 1], [-1, 1]]
        surrogate, point_sets = create_surrogate(
            surrogate_class,
            smolyay.basis.ChebyshevFirstKind,
            smolyay.samples.NestedClenshawCurtisPointSet,
            num_level,
            domain,
        )
        grid = grid_obj(point_sets=point_sets)
        sample_output = branin(grid.points)
        surrogate = surrogate.fit(grid, sample_output)

        # test surrogate matches at some points
        test_points = smolyay.samples.LatinHypercubeRandomPointSet(
            domain, 5, 1234
        ).points
        predict_answer = branin(test_points)
        gradient_answer = branin_gradient(test_points)
        hessian_answer = branin_hessian(test_points)
        assert numpy.allclose(predict_answer, surrogate.predict(test_points), rtol=1e-3)
        assert numpy.allclose(
            gradient_answer, surrogate.predict_gradient(test_points), rtol=1e-3
        )
        assert numpy.allclose(
            hessian_answer, surrogate.predict_hessian(test_points), rtol=1e-3
        )

    def test_fit_2D_complex(self, surrogate_class, grid_obj):
        """Test if class is fit to 2D function using periodic basis function with complex outputs."""
        domain = [[0, 2 * numpy.pi], [0, 2 * numpy.pi]]
        num_level = 2

        surrogate, point_sets = create_surrogate(
            surrogate_class,
            smolyay.basis.Trigonometric,
            smolyay.samples.NestedTrigonometricPointSet,
            num_level,
            domain,
        )
        grid = grid_obj(point_sets=point_sets)
        sample_output = [function_4(x) for x in grid.points]
        surrogate = surrogate.fit(grid, sample_output)

        # test surrogate matches at some points
        test_points = smolyay.samples.LatinHypercubeRandomPointSet(
            domain, 5, 1234
        ).points
        predict_answer = [function_4(x) for x in test_points]
        gradient_answer = [function_4_gradient(x) for x in test_points]
        hessian_answer = [function_4_hessian(x) for x in test_points]
        assert numpy.allclose(predict_answer, surrogate.predict(test_points))
        assert numpy.allclose(gradient_answer, surrogate.predict_gradient(test_points))
        assert numpy.allclose(hessian_answer, surrogate.predict_hessian(test_points))

    def test_fit_2D_mixed_basis(self, surrogate_class, grid_obj):
        """Test if class is fit using different basis functions."""
        domain = [[0, 2 * numpy.pi], [-1, 1]]
        num_level = 2
        point_sets = [
            smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], num_level),
            smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], num_level),
        ]
        basis_sets = [
            smolyay.basis.NestedTrigonometricBasisFunctionSet(2),
            smolyay.basis.NestedClenshawCurtisBasisFunctionSet(2),
        ]
        surrogate = surrogate_class(domain, basis_sets)
        grid = grid_obj(point_sets=point_sets)
        sample_output = [function_1(x) for x in grid.points]
        surrogate = surrogate.fit(grid, sample_output)

        # test surrogate matches at some points
        test_points = smolyay.samples.LatinHypercubeRandomPointSet(
            domain, 5, 1234
        ).points
        predict_answer = [function_1(x) for x in test_points]
        gradient_answer = [function_1_gradient(x) for x in test_points]
        hessian_answer = [function_1_hessian(x) for x in test_points]
        assert numpy.allclose(predict_answer, surrogate.predict(test_points))
        assert numpy.allclose(gradient_answer, surrogate.predict_gradient(test_points))
        assert numpy.allclose(hessian_answer, surrogate.predict_hessian(test_points))

    def test_fit_regularization_2D(self, surrogate_class, grid_obj):
        """Test if class is fit when number of terms doesn't match samples for 2D function."""
        domain = [[-5, 5], [0, 10]]
        num_level = 4
        basis_sets = [
            smolyay.basis.NestedClenshawCurtisBasisFunctionSet(num_level)
            for d in domain
        ]
        surrogate = surrogate_class(domain, basis_sets, None)
        grid = smolyay.samples.LatinHypercubeRandomPointSet(domain, 1000, 1234)
        sample_output = branin(grid.points)
        surrogate.fit(grid, sample_output)

        # test surrogate matches at some points
        test_points = numpy.array([[-0.5, 0.8], [1, 1], [0.7, 0.9]])
        predict_answer = branin(test_points)
        gradient_answer = branin_gradient(test_points)
        hessian_answer = branin_hessian(test_points)
        assert numpy.allclose(predict_answer, surrogate.predict(test_points), rtol=0.01)
        assert numpy.allclose(
            gradient_answer, surrogate.predict_gradient(test_points), rtol=0.01
        )
        assert numpy.allclose(
            hessian_answer, surrogate.predict_hessian(test_points), rtol=0.01, atol=1e-1
        )

    def test_fit_regularization_complex_2D(self, surrogate_class, grid_obj):
        """Test if class is fit when number of terms doesn't match samples for 2D function."""
        domain = [[0, 2*numpy.pi], [0, 2*numpy.pi]]
        num_level = 4
        basis_sets = [
            smolyay.basis.NestedTrigonometricBasisFunctionSet(num_level) for d in domain
        ]
        surrogate = surrogate_class(domain, basis_sets, None)
        grid = smolyay.samples.LatinHypercubeRandomPointSet(domain, 800, 1234)
        sample_output = [function_4(x) for x in grid.points]
        surrogate.fit(grid, sample_output)
        print(len(surrogate._index_combinations))
        # test surrogate matches at some points
        test_points = smolyay.samples.LatinHypercubeRandomPointSet(
            domain, 5, 1234
        ).points
        predict_answer = [function_4(x) for x in test_points]
        gradient_answer = [function_4_gradient(x) for x in test_points]
        hessian_answer = [function_4_hessian(x) for x in test_points]
        print(surrogate.predict(test_points))
        print(predict_answer)
        assert numpy.allclose(predict_answer, surrogate.predict(test_points), rtol=0.01)
        assert numpy.allclose(
            gradient_answer, surrogate.predict_gradient(test_points), rtol=0.01
        )
        assert numpy.allclose(
            hessian_answer, surrogate.predict_hessian(test_points), rtol=0.01, atol=1e-1
        )

    def test_fit_regularization_ridge_2D(self, surrogate_class, grid_obj):
        """Test if class is fit when number of terms doesn't match samples for 2D function."""
        domain = [[-5, 5], [0, 10]]
        num_level = 4
        basis_sets = [
            smolyay.basis.NestedClenshawCurtisBasisFunctionSet(num_level)
            for d in domain
        ]
        surrogate = surrogate_class(
            domain, basis_sets, smolyay.surrogate.L2Regularization(alpha=1e-10)
        )
        grid = smolyay.samples.LatinHypercubeRandomPointSet(domain, 1000, 1234)
        sample_output = branin(grid.points)
        surrogate.fit(grid, sample_output)

        # test surrogate matches at some points
        test_points = numpy.array([[-0.5, 0.8], [1, 1], [0.7, 0.9]])
        predict_answer = branin(test_points)
        gradient_answer = branin_gradient(test_points)
        hessian_answer = branin_hessian(test_points)
        assert numpy.allclose(predict_answer, surrogate.predict(test_points), rtol=0.01)
        assert numpy.allclose(
            gradient_answer, surrogate.predict_gradient(test_points), rtol=0.01
        )
        assert numpy.allclose(
            hessian_answer, surrogate.predict_hessian(test_points), rtol=0.01, atol=1e-1
        )

    def test_fit_regularization_lasso_2D(self, surrogate_class, grid_obj):
        """Test if class is fit when number of terms doesn't match samples for 2D function."""
        domain = [[-5, 5], [0, 10]]
        num_level = 4
        basis_sets = [
            smolyay.basis.NestedClenshawCurtisBasisFunctionSet(num_level)
            for d in domain
        ]
        surrogate = surrogate_class(
            domain, basis_sets, (smolyay.surrogate.L1Regularization(alpha=1e-10))
        )
        grid = smolyay.samples.LatinHypercubeRandomPointSet(domain, 2400, 1234)
        sample_output = branin(grid.points)
        surrogate.fit(grid, sample_output)

        # test surrogate matches at some points
        test_points = numpy.array([[-0.5, 0.8], [1, 1], [0.7, 0.9]])
        predict_answer = branin(test_points)
        gradient_answer = branin_gradient(test_points)
        hessian_answer = branin_hessian(test_points)
        assert numpy.allclose(predict_answer, surrogate.predict(test_points), rtol=0.01)
        assert numpy.allclose(
            gradient_answer, surrogate.predict_gradient(test_points), rtol=0.01
        )
        assert numpy.allclose(
            hessian_answer, surrogate.predict_hessian(test_points), rtol=0.01, atol=1e-1
        )


@pytest.mark.parametrize(
    "surrogate_class",
    [
        smolyay.surrogate.TensorProductSurrogate,
        smolyay.surrogate.SmolyakSparseProductSurrogate,
    ],
    ids=["Tensor", "Smolyak"],
)
@pytest.mark.incremental
class TestFit1D:

    def test_fit_1D(self, surrogate_class):
        """Test if class is fit to 1D function."""
        num_level = 3
        domain = [-1, 1]
        # fit with a 1D function
        surrogate, point_sets = create_surrogate(
            surrogate_class,
            smolyay.basis.ChebyshevFirstKind,
            smolyay.samples.NestedClenshawCurtisPointSet,
            num_level,
            domain,
        )
        grid_points = numpy.array(point_sets[0].points, ndmin=2).reshape((-1, 1))
        sample_output = function_2(grid_points)
        surrogate.fit(grid_points, sample_output)

        # test surrogate matches at some points
        test_points = smolyay.samples.LatinHypercubeRandomPointSet(
            domain, 5, 1234
        ).points
        predict_answer = numpy.squeeze(function_2(test_points))
        gradient_answer = numpy.array(function_2_gradient(test_points), ndmin=2)
        hessian_answer = numpy.array(function_2_hessian(test_points), ndmin=3).reshape(
            (-1, 1, 1)
        )
        assert numpy.allclose(predict_answer, surrogate.predict(test_points))
        assert numpy.allclose(gradient_answer, surrogate.predict_gradient(test_points))
        assert numpy.allclose(hessian_answer, surrogate.predict_hessian(test_points))

    def test_fit_1D_domain_shift(self, surrogate_class):
        """Test if class is fit to 1D function with different domain as basis."""
        num_level = 3
        domain = [-5, 10]
        # fit with a 1D function
        surrogate, point_sets = create_surrogate(
            surrogate_class,
            smolyay.basis.ChebyshevFirstKind,
            smolyay.samples.NestedClenshawCurtisPointSet,
            num_level,
            domain,
        )
        grid_points = numpy.array(point_sets[0].points, ndmin=2).reshape((-1, 1))
        sample_output = function_2(grid_points)
        surrogate.fit(grid_points, sample_output)

        # test surrogate matches at some points
        test_points = smolyay.samples.LatinHypercubeRandomPointSet(
            domain, 5, 1234
        ).points
        predict_answer = numpy.squeeze(function_2(test_points))
        gradient_answer = numpy.array(function_2_gradient(test_points), ndmin=2)
        hessian_answer = numpy.array(function_2_hessian(test_points), ndmin=3).reshape(
            (-1, 1, 1)
        )
        assert numpy.allclose(predict_answer, surrogate.predict(test_points))
        assert numpy.allclose(gradient_answer, surrogate.predict_gradient(test_points))
        assert numpy.allclose(hessian_answer, surrogate.predict_hessian(test_points))

    def test_fit_1D_complex(self, surrogate_class):
        """Test if class is fit to 1D function using periodic basis function with complex outputs."""
        domain = [0, 2 * numpy.pi]
        num_level = 2

        surrogate, point_sets = create_surrogate(
            surrogate_class,
            smolyay.basis.Trigonometric,
            smolyay.samples.NestedTrigonometricPointSet,
            num_level,
            domain,
        )
        grid_points = numpy.array(point_sets[0].points, ndmin=2).reshape((-1, 1))
        sample_output = [function_5(x) for x in grid_points]
        surrogate = surrogate.fit(grid_points, sample_output)

        # test surrogate matches at some points
        test_points = smolyay.samples.LatinHypercubeRandomPointSet(
            domain, 5, 1234
        ).points
        predict_answer = numpy.squeeze(function_5(test_points))
        gradient_answer = function_5_gradient(test_points)
        hessian_answer = function_5_hessian(test_points).reshape((-1, 1, 1))
        assert numpy.allclose(predict_answer, surrogate.predict(test_points))
        assert numpy.allclose(gradient_answer, surrogate.predict_gradient(test_points))
        assert numpy.allclose(hessian_answer, surrogate.predict_hessian(test_points))

    @pytest.mark.parametrize(
        "regularization",
        [
            smolyay.surrogate.L2Regularization(alpha=1e-10),
            smolyay.surrogate.L1Regularization(alpha=1e-10),
            None,
        ],
        ids=["Ridge", "Lasso", "Least Squares"],
    )
    def test_fit_regularization_1D(self, surrogate_class, regularization):
        """Test if class is fit when number of terms doesn't match samples for 1D function."""
        domain = [-5, 10]
        num_level = 4

        # fit with a 1D function
        basis_sets = [smolyay.basis.NestedClenshawCurtisBasisFunctionSet(num_level)]
        surrogate = surrogate_class(domain, basis_sets, regularization)
        grid = smolyay.samples.LatinHypercubeRandomPointSet(domain, 500, 1234)
        sample_output = function_2(grid.points)
        surrogate.fit(grid.points, sample_output)

        # test surrogate matches at some points
        test_points = numpy.array([1, 2, 3], ndmin=2).reshape((-1, 1))
        predict_answer = numpy.squeeze(function_2(test_points))
        gradient_answer = numpy.array(function_2_gradient(test_points), ndmin=2)
        hessian_answer = numpy.array(function_2_hessian(test_points), ndmin=3).reshape(
            (-1, 1, 1)
        )
        assert numpy.allclose(predict_answer, surrogate.predict(test_points), rtol=0.01)
        assert numpy.allclose(
            gradient_answer,
            surrogate.predict_gradient(test_points),
            rtol=0.01,
            atol=1e-3,
        )
        assert numpy.allclose(
            hessian_answer, surrogate.predict_hessian(test_points), rtol=0.01, atol=1e-3
        )


@pytest.mark.parametrize(
    "surrogate_class",
    [
        (smolyay.surrogate.TensorProductSurrogate),
        (smolyay.surrogate.SmolyakSparseProductSurrogate),
    ],
    ids=["Tensor", "Smolyak"],
)
def test_fit_error(surrogate_class):
    """Test if fit raises an error if points are outside domain."""
    domain = [[-5, 10], [0, 15]]
    bs = [
        smolyay.basis.NestedClenshawCurtisBasisFunctionSet(3),
        smolyay.basis.NestedClenshawCurtisBasisFunctionSet(3),
    ]
    surrogate = surrogate_class(domain, bs)

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
    "surrogate_class,grid_obj",
    [
        (
            smolyay.surrogate.TensorProductSurrogate,
            smolyay.samples.TensorProductPointSet,
        ),
        (
            smolyay.surrogate.SmolyakSparseProductSurrogate,
            smolyay.samples.SmolyakSparseProductPointSet,
        ),
    ],
    ids=["Tensor", "Smolyak"],
)
@pytest.mark.incremental
class TestFitGradient2D:

    def test_fit_gradient_2D(self, surrogate_class, grid_obj):
        """Test if class is fit to gradient for 2D function."""
        num_level = 3
        domain = numpy.array([[-1, 1], [-1, 1]])
        surrogate, point_sets = create_surrogate(
            surrogate_class,
            smolyay.basis.ChebyshevFirstKind,
            smolyay.samples.NestedClenshawCurtisPointSet,
            num_level,
            domain,
        )
        grid = grid_obj(point_sets=point_sets)
        sample_output = [function_3_gradient(x) for x in grid.points]
        surrogate = surrogate.fit_gradient(grid, sample_output)

        # test surrogate matches at some points
        test_points = smolyay.samples.LatinHypercubeRandomPointSet(
            domain, 5, 1234
        ).points
        predict_answer = [function_3(x) for x in test_points]
        gradient_answer = [function_3_gradient(x) for x in test_points]
        hessian_answer = [function_3_hessian(x) for x in test_points]
        assert numpy.allclose(gradient_answer, surrogate.predict_gradient(test_points))
        assert numpy.allclose(hessian_answer, surrogate.predict_hessian(test_points))

        # test predict
        difference_predict = numpy.subtract(
            predict_answer, surrogate.predict(test_points)
        )
        assert numpy.allclose(difference_predict, difference_predict[0])
        surrogate = surrogate.fit_gradient(
            grid, sample_output, y0=[function_3(domain[:, 0])]
        )
        assert numpy.allclose(predict_answer, surrogate.predict(test_points))
        surrogate = surrogate.fit_gradient(
            grid, sample_output, X0=[domain[:, 1]], y0=[function_3(domain[:, 1])]
        )
        assert numpy.allclose(predict_answer, surrogate.predict(test_points))

    def test_fit_gradient_2D_domain_shift(self, surrogate_class, grid_obj):
        """Test class is fit to gradient for 2D function with different domain as basis."""
        num_level = 3
        domain = numpy.array([[-5, 10], [0, 15]])
        surrogate, point_sets = create_surrogate(
            surrogate_class,
            smolyay.basis.ChebyshevFirstKind,
            smolyay.samples.NestedClenshawCurtisPointSet,
            num_level,
            domain,
        )
        grid = grid_obj(point_sets=point_sets)
        sample_output = [function_3_gradient(x) for x in grid.points]
        surrogate = surrogate.fit_gradient(grid, sample_output)

        # test surrogate matches at some points
        test_points = smolyay.samples.LatinHypercubeRandomPointSet(
            domain, 5, 1234
        ).points
        predict_answer = [function_3(x) for x in test_points]
        gradient_answer = [function_3_gradient(x) for x in test_points]
        hessian_answer = [function_3_hessian(x) for x in test_points]
        assert numpy.allclose(gradient_answer, surrogate.predict_gradient(test_points))
        assert numpy.allclose(hessian_answer, surrogate.predict_hessian(test_points))

        # test predict
        difference_predict = numpy.subtract(
            predict_answer, surrogate.predict(test_points)
        )
        assert numpy.allclose(difference_predict, difference_predict[0])
        surrogate = surrogate.fit_gradient(
            grid, sample_output, y0=[function_3(domain[:, 0])]
        )
        assert numpy.allclose(predict_answer, surrogate.predict(test_points))
        surrogate = surrogate.fit_gradient(
            grid, sample_output, X0=[domain[:, 1]], y0=[function_3(domain[:, 1])]
        )
        assert numpy.allclose(predict_answer, surrogate.predict(test_points))

    def test_fit_gradient_2D_complex(self, surrogate_class, grid_obj):
        """Test if class is fit to gradient using periodic basis function with complex outputs."""
        domain = numpy.array([[0, 2 * numpy.pi], [0, 2 * numpy.pi]])
        num_level = 2

        surrogate, point_sets = create_surrogate(
            surrogate_class,
            smolyay.basis.Trigonometric,
            smolyay.samples.NestedTrigonometricPointSet,
            num_level,
            domain,
        )
        grid = grid_obj(point_sets=point_sets)
        sample_output = [function_4_gradient(x) for x in grid.points]
        surrogate = surrogate.fit_gradient(grid, sample_output)

        # test surrogate matches at some points
        test_points = smolyay.samples.LatinHypercubeRandomPointSet(
            domain, 5, 1234
        ).points
        predict_answer = [function_4(x) for x in test_points]
        gradient_answer = [function_4_gradient(x) for x in test_points]
        hessian_answer = [function_4_hessian(x) for x in test_points]
        assert numpy.allclose(gradient_answer, surrogate.predict_gradient(test_points))
        assert numpy.allclose(hessian_answer, surrogate.predict_hessian(test_points))

        # test predict
        difference_predict = numpy.subtract(
            predict_answer, surrogate.predict(test_points)
        )
        assert numpy.allclose(difference_predict, difference_predict[0])
        surrogate = surrogate.fit_gradient(
            grid, sample_output, [domain[:, 0]], [function_4(domain[:, 0])]
        )
        assert numpy.allclose(predict_answer, surrogate.predict(test_points))

    def test_fit_gradient_2D_mixed_basis(self, surrogate_class, grid_obj):
        """Test if class is fit with gradient with different basis functions."""
        domain = numpy.array([[0, 2 * numpy.pi], [-1, 1]])
        num_level = 2
        point_sets = [
            smolyay.samples.NestedTrigonometricPointSet([0, 2 * numpy.pi], num_level),
            smolyay.samples.NestedClenshawCurtisPointSet([-1, 1], num_level),
        ]
        basis_sets = [
            smolyay.basis.NestedTrigonometricBasisFunctionSet(2),
            smolyay.basis.NestedClenshawCurtisBasisFunctionSet(2),
        ]
        surrogate = surrogate_class(domain, basis_sets)
        grid = grid_obj(point_sets=point_sets)
        sample_output = [function_1_gradient(x) for x in grid.points]
        surrogate = surrogate.fit_gradient(grid, sample_output)

        # test surrogate matches at some points
        test_points = smolyay.samples.LatinHypercubeRandomPointSet(
            domain, 5, 1234
        ).points
        predict_answer = [function_1(x) for x in test_points]
        gradient_answer = [function_1_gradient(x) for x in test_points]
        hessian_answer = [function_1_hessian(x) for x in test_points]
        assert numpy.allclose(gradient_answer, surrogate.predict_gradient(test_points))
        assert numpy.allclose(hessian_answer, surrogate.predict_hessian(test_points))

        # test predict
        difference_predict = numpy.subtract(
            predict_answer, surrogate.predict(test_points)
        )
        assert numpy.allclose(difference_predict, difference_predict[0])
        surrogate = surrogate.fit_gradient(
            grid, sample_output, [domain[:, 0]], [function_1(domain[:, 0])]
        )
        assert numpy.allclose(predict_answer, surrogate.predict(test_points))

    @pytest.mark.parametrize(
        "regularization",
        [
            smolyay.surrogate.L2Regularization(alpha=1e-10),
            smolyay.surrogate.L1Regularization(alpha=1e-10),
            None,
        ],
        ids=["Ridge", "Lasso", "Least Squares"],
    )
    def test_fit_gradient_regularization_2D(
        self, surrogate_class, grid_obj, regularization
    ):
        """Test if class is fit using gradient when n_terms != n_points for 2D function."""
        domain = numpy.array([[-5, 5], [-1, 1]])
        num_level = 3
        basis_sets = [
            smolyay.basis.NestedClenshawCurtisBasisFunctionSet(num_level)
            for d in domain
        ]
        surrogate = surrogate_class(domain, basis_sets, regularization)
        grid = smolyay.samples.LatinHypercubeRandomPointSet(domain, 1000, 1234)
        sample_output = [function_3_gradient(x) for x in grid.points]
        surrogate.fit_gradient(grid, sample_output)

        # test surrogate matches at some points
        test_points = numpy.array([[-0.5, 0.8], [0, 0], [0.7, 0.2]])
        predict_answer = [function_3(x) for x in test_points]
        gradient_answer = [function_3_gradient(x) for x in test_points]
        hessian_answer = [function_3_hessian(x) for x in test_points]
        assert numpy.allclose(
            gradient_answer,
            surrogate.predict_gradient(test_points),
            atol=1e-4,
        )
        assert numpy.allclose(
            hessian_answer, surrogate.predict_hessian(test_points), atol=5e-4
        )

        # test predict
        difference_predict = numpy.subtract(
            predict_answer, surrogate.predict(test_points)
        )
        assert numpy.allclose(
            difference_predict, difference_predict[0], rtol=0.01, atol=1e-4
        )
        surrogate = surrogate.fit_gradient(
            grid, sample_output, [domain[:, 0]], [function_3(domain[:, 0])]
        )
        assert numpy.allclose(
            predict_answer, surrogate.predict(test_points), rtol=0.01, atol=1e-3
        )


@pytest.mark.parametrize(
    "surrogate_class",
    [
        smolyay.surrogate.TensorProductSurrogate,
        smolyay.surrogate.SmolyakSparseProductSurrogate,
    ],
    ids=["Tensor", "Smolyak"],
)
@pytest.mark.incremental
class TestFitGradient1D:

    def test_fit_gradient_1D(self, surrogate_class):
        """Test class is fit using gradient for 1D function."""
        num_level = 3
        domain = numpy.array([-1, 1])
        surrogate, point_sets = create_surrogate(
            surrogate_class,
            smolyay.basis.ChebyshevFirstKind,
            smolyay.samples.NestedClenshawCurtisPointSet,
            num_level,
            domain,
        )
        grid_points = numpy.array(point_sets[0].points, ndmin=2).reshape((-1, 1))
        sample_output = function_2_gradient(grid_points)
        surrogate.fit_gradient(grid_points, sample_output)

        # test surrogate matches at some points
        test_points = numpy.array([0.1, 0.2, 0.3], ndmin=2).reshape((-1, 1))
        predict_answer = numpy.squeeze(function_2(test_points))
        gradient_answer = function_2_gradient(test_points)
        hessian_answer = numpy.array(function_2_hessian(test_points), ndmin=3).reshape(
            (-1, 1, 1)
        )
        assert numpy.allclose(
            gradient_answer,
            surrogate.predict_gradient(test_points),
        )
        assert numpy.allclose(hessian_answer, surrogate.predict_hessian(test_points))

        # test predict
        difference_predict = numpy.subtract(
            predict_answer, surrogate.predict(test_points)
        )
        assert numpy.allclose(difference_predict, difference_predict[0])
        surrogate = surrogate.fit_gradient(
            grid_points, sample_output, [[domain[0]]], [function_2(domain[0])]
        )
        assert numpy.allclose(predict_answer, surrogate.predict(test_points))

    def test_fit_gradient_1D_domain_shift(self, surrogate_class):
        """Test class is fit using gradient for 1D function with different domain as basis."""
        num_level = 3
        domain = numpy.array([-5, 10])
        surrogate, point_sets = create_surrogate(
            surrogate_class,
            smolyay.basis.ChebyshevFirstKind,
            smolyay.samples.NestedClenshawCurtisPointSet,
            num_level,
            domain,
        )
        grid_points = numpy.array(point_sets[0].points, ndmin=2).reshape((-1, 1))
        sample_output = function_2_gradient(grid_points)
        surrogate.fit_gradient(grid_points, sample_output)

        # test surrogate matches at some points
        test_points = numpy.array([0.1, 0.2, 0.3], ndmin=2).reshape((-1, 1))
        predict_answer = numpy.squeeze(function_2(test_points))
        gradient_answer = function_2_gradient(test_points)
        hessian_answer = numpy.array(function_2_hessian(test_points), ndmin=3).reshape(
            (-1, 1, 1)
        )
        assert numpy.allclose(
            gradient_answer,
            surrogate.predict_gradient(test_points),
        )
        assert numpy.allclose(hessian_answer, surrogate.predict_hessian(test_points))

        # test predict
        difference_predict = numpy.subtract(
            predict_answer, surrogate.predict(test_points)
        )
        assert numpy.allclose(difference_predict, difference_predict[0])
        surrogate = surrogate.fit_gradient(
            grid_points, sample_output, [[domain[0]]], [function_2(domain[0])]
        )
        assert numpy.allclose(predict_answer, surrogate.predict(test_points))

    def test_fit_gradient_1D_complex(self, surrogate_class):
        """Test if class is fit to gradient using periodic basis function with complex outputs."""
        domain = numpy.array([0, 2 * numpy.pi])
        num_level = 2

        surrogate, point_sets = create_surrogate(
            surrogate_class,
            smolyay.basis.Trigonometric,
            smolyay.samples.NestedTrigonometricPointSet,
            num_level,
            domain,
        )
        grid_points = numpy.array(point_sets[0].points, ndmin=2).reshape((-1, 1))
        sample_output = function_5_gradient(grid_points)
        surrogate = surrogate.fit_gradient(grid_points, sample_output)

        # test surrogate matches at some points
        test_points = smolyay.samples.LatinHypercubeRandomPointSet(
            domain, 5, 1234
        ).points
        predict_answer = numpy.squeeze(function_5(test_points))
        gradient_answer = function_5_gradient(test_points)
        hessian_answer = numpy.array(function_5_hessian(test_points), ndmin=3).reshape(
            (-1, 1, 1)
        )
        assert numpy.allclose(gradient_answer, surrogate.predict_gradient(test_points))
        assert numpy.allclose(hessian_answer, surrogate.predict_hessian(test_points))

        # test predict
        difference_predict = numpy.subtract(
            predict_answer, surrogate.predict(test_points)
        )
        assert numpy.allclose(difference_predict, difference_predict[0])
        surrogate = surrogate.fit_gradient(
            grid_points, sample_output, [[domain[0]]], [function_5(domain[0])]
        )
        assert numpy.allclose(predict_answer, surrogate.predict(test_points))

    @pytest.mark.parametrize(
        "regularization",
        [
            smolyay.surrogate.L2Regularization(alpha=1e-10),
            smolyay.surrogate.L1Regularization(alpha=1e-10),
            None,
        ],
        ids=["Ridge", "Lasso", "Least Squares"],
    )
    def test_fit_gradient_regularization_1D(self, surrogate_class, regularization):
        """Test if class is fit using gradient when n_terms != n_points for 1D function."""
        num_level = 3
        domain = [-5, 6]
        # fit with a 1D function
        basis_sets = [smolyay.basis.NestedClenshawCurtisBasisFunctionSet(num_level)]
        surrogate = surrogate_class(domain, basis_sets, regularization)
        grid = smolyay.samples.LatinHypercubeRandomPointSet(domain, 100, 1234)
        sample_output = function_2_gradient(grid.points)
        surrogate.fit_gradient(grid, sample_output)

        # test surrogate matches at some points
        test_points = numpy.array([0.1, 0.2, 0.3], ndmin=2).reshape((-1, 1))
        predict_answer = numpy.squeeze(function_2(test_points))
        gradient_answer = function_2_gradient(test_points)
        hessian_answer = numpy.array(function_2_hessian(test_points), ndmin=3).reshape(
            (-1, 1, 1)
        )
        assert numpy.allclose(gradient_answer, surrogate.predict_gradient(test_points))
        assert numpy.allclose(hessian_answer, surrogate.predict_hessian(test_points))

        # test predict
        difference_predict = numpy.subtract(
            predict_answer, surrogate.predict(test_points)
        )
        assert numpy.allclose(difference_predict, difference_predict[0])
        surrogate = surrogate.fit_gradient(
            grid, sample_output, [[domain[0]]], [function_2(domain[0])]
        )
        assert numpy.allclose(predict_answer, surrogate.predict(test_points), rtol=0.01)


@pytest.mark.parametrize(
    "surrogate_class",
    [
        smolyay.surrogate.TensorProductSurrogate,
        smolyay.surrogate.SmolyakSparseProductSurrogate,
    ],
    ids=["Tensor", "Smolyak"],
)
def test_fit_gradient_error(surrogate_class):
    """Test if fit_gradient raises an error if points are outside domain."""
    domain = [[-5, 10], [0, 15]]
    bs = [
        smolyay.basis.NestedClenshawCurtisBasisFunctionSet(3),
        smolyay.basis.NestedClenshawCurtisBasisFunctionSet(3),
    ]
    surrogate = surrogate_class(domain, bs)
    with pytest.raises(IndexError):
        test_points = numpy.array([[-0.5, 0.8], [0, 0], [0.7, 0.2]])
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
    "surrogate_class,grid_obj",
    [
        (
            smolyay.surrogate.TensorProductSurrogate,
            smolyay.samples.TensorProductPointSet,
        ),
        (
            smolyay.surrogate.SmolyakSparseProductSurrogate,
            smolyay.samples.SmolyakSparseProductPointSet,
        ),
    ],
    ids=["Tensor", "Smolyak"],
)
def test_successive_fits(surrogate_class, grid_obj):
    """Test that predict and predict_gradient are unaffected by previous fittings"""
    domain = numpy.array([[-5, 10], [0, 15]])
    num_level = 5
    surrogate, point_sets = create_surrogate(
        surrogate_class,
        smolyay.basis.ChebyshevFirstKind,
        smolyay.samples.NestedClenshawCurtisPointSet,
        num_level,
        domain,
    )
    grid = grid_obj(point_sets=point_sets)
    sample_output_1 = branin(grid.points)
    sample_output_gradient_1 = branin_gradient(grid.points)
    sample_output_gradient_2 = [function_3_gradient(x) for x in grid.points]
    X0 = [domain[:, 1]]
    y0_1 = [branin(domain[:, 1])]
    y0_2 = [function_3(domain[:, 1])]
    # test surrogate matches at some points
    test_points = smolyay.samples.LatinHypercubeRandomPointSet(domain, 5, 1234).points
    predict_answer_1 = branin(test_points)
    gradient_answer_1 = branin_gradient(test_points)
    predict_answer_2 = [function_3(x) for x in test_points]
    gradient_answer_2 = [function_3_gradient(x) for x in test_points]
    surrogate = surrogate.fit_gradient(grid, sample_output_gradient_1, X0, y0_1)
    surrogate = surrogate.fit_gradient(grid, sample_output_gradient_2)
    assert numpy.allclose(gradient_answer_2, surrogate.predict_gradient(test_points))
    difference_predict = numpy.subtract(
        predict_answer_2, surrogate.predict(test_points)
    )
    assert numpy.allclose(difference_predict, difference_predict[0])
    surrogate = surrogate.fit_gradient(grid, sample_output_gradient_2, X0, y0_2)
    assert numpy.allclose(predict_answer_2, surrogate.predict(test_points))
    surrogate = surrogate.fit(grid, sample_output_1)
    assert numpy.allclose(
        gradient_answer_1, surrogate.predict_gradient(test_points), rtol=1e-3
    )
    assert numpy.allclose(predict_answer_1, surrogate.predict(test_points), rtol=1e-3)


@pytest.mark.parametrize(
    "surrogate_class,grid_obj",
    [
        (
            smolyay.surrogate.TensorProductSurrogate,
            smolyay.samples.TensorProductPointSet,
        ),
        (
            smolyay.surrogate.SmolyakSparseProductSurrogate,
            smolyay.samples.SmolyakSparseProductPointSet,
        ),
    ],
    ids=["Tensor", "Smolyak"],
)
@pytest.mark.incremental
class TestPredictSize:
    def test_predict_size_2D(self, surrogate_class, grid_obj):
        """Test predict returns answer of the appropriate shape"""
        domain = [[-5, 10], [0, 15]]
        num_level = 5
        surrogate, point_sets = create_surrogate(
            surrogate_class,
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

    def test_predict_size_1D(self, surrogate_class, grid_obj):
        num_level = 4
        domain = [-5, 5]
        # fit with a 1D function
        surrogate, point_sets = create_surrogate(
            surrogate_class,
            smolyay.basis.ChebyshevFirstKind,
            smolyay.samples.NestedClenshawCurtisPointSet,
            num_level,
            domain,
        )
        grid_points = numpy.array(point_sets[0].points, ndmin=2).reshape((-1, 1))
        sample_output = function_2(grid_points)
        surrogate.fit(grid_points, sample_output)
        assert numpy.array_equal(
            numpy.shape(surrogate.predict([[-0.5], [0], [0.7]])), (3,)
        )
        assert numpy.array_equal(numpy.shape(surrogate.predict([[0.7]])), ())
        assert numpy.array_equal(numpy.shape(surrogate.predict([0.7])), ())


@pytest.mark.parametrize(
    "surrogate_class,basis_sets",
    [
        (
            smolyay.surrogate.TensorProductSurrogate,
            [
                smolyay.basis.ChebyshevFirstKindBasisFunctionSet(10),
                smolyay.basis.ChebyshevFirstKindBasisFunctionSet(10),
            ],
        ),
        (
            smolyay.surrogate.SmolyakSparseProductSurrogate,
            [
                smolyay.basis.NestedClenshawCurtisBasisFunctionSet(3),
                smolyay.basis.NestedClenshawCurtisBasisFunctionSet(3),
            ],
        ),
    ],
    ids=["Tensor", "Smolyak"],
)
def test_predict_error(surrogate_class, basis_sets):
    """Test that predict raises correct errors"""
    surrogate = surrogate_class([[-5, 10], [0, 15]], basis_sets)
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


@pytest.mark.parametrize(
    "surrogate_class,grid_obj",
    [
        (
            smolyay.surrogate.TensorProductSurrogate,
            smolyay.samples.TensorProductPointSet,
        ),
        (
            smolyay.surrogate.SmolyakSparseProductSurrogate,
            smolyay.samples.SmolyakSparseProductPointSet,
        ),
    ],
    ids=["Tensor", "Smolyak"],
)
@pytest.mark.incremental
class TestGradientSize:

    def test_predict_gradient_size_2D(self, surrogate_class, grid_obj):
        """Test predict_gradient returns answer of the appropriate shape."""
        domain = [[-5, 10], [0, 15]]
        num_level = 5

        surrogate, point_sets = create_surrogate(
            surrogate_class,
            smolyay.basis.ChebyshevFirstKind,
            smolyay.samples.NestedClenshawCurtisPointSet,
            num_level,
            domain,
        )
        grid = grid_obj(point_sets=point_sets)
        # fit with same number of points as terms
        surrogate.fit(grid, branin(grid.points))
        assert numpy.array_equal(
            numpy.shape(surrogate.predict_gradient([[-0.5, 0.8], [0, 0], [0.7, 0]])),
            (3, 2),
        )
        assert numpy.array_equal(
            numpy.shape(surrogate.predict_gradient([[0.7, 0]])), (1, 2)
        )

    def test_predict_gradient_size_1D(self, surrogate_class, grid_obj):
        num_level = 4
        domain = [-5, 5]
        surrogate, point_sets = create_surrogate(
            surrogate_class,
            smolyay.basis.ChebyshevFirstKind,
            smolyay.samples.NestedClenshawCurtisPointSet,
            num_level,
            domain,
        )
        # fit with a 1D function
        grid_points = numpy.array(point_sets[0].points, ndmin=2).reshape((-1, 1))
        sample_output = function_2(grid_points)
        surrogate.fit(grid_points, sample_output)
        assert numpy.array_equal(
            numpy.shape(surrogate.predict_gradient([[-0.5], [0], [0.7]])), (3, 1)
        )
        assert numpy.array_equal(numpy.shape(surrogate.predict_gradient([[0.7]])), ())


@pytest.mark.parametrize(
    "surrogate_class,basis_sets",
    [
        (
            smolyay.surrogate.TensorProductSurrogate,
            [
                smolyay.basis.ChebyshevFirstKindBasisFunctionSet(10),
                smolyay.basis.ChebyshevFirstKindBasisFunctionSet(10),
            ],
        ),
        (
            smolyay.surrogate.SmolyakSparseProductSurrogate,
            [
                smolyay.basis.NestedClenshawCurtisBasisFunctionSet(3),
                smolyay.basis.NestedClenshawCurtisBasisFunctionSet(3),
            ],
        ),
    ],
    ids=["Tensor", "Smolyak"],
)
@pytest.mark.incremental
def test_predict_gradient_error(surrogate_class, basis_sets):
    """Test that predict_gradient raises correct errors"""
    surrogate = surrogate_class([[-5, 10], [0, 15]], basis_sets)
    grid = smolyay.samples.LatinHypercubeRandomPointSet([[-5, 10], [0, 15]], 1500, 1234)
    with pytest.raises(RuntimeError):
        surrogate.predict_gradient([[0.7, 0.3]])
    with pytest.raises(ValueError):
        surrogate.fit(grid, branin(grid.points))
        surrogate.predict_gradient([[11, 5]])
    with pytest.raises(ValueError):
        surrogate.fit(grid, branin(grid.points))
        surrogate.predict_gradient([[5, 4], [3, 20], [0, 5]])
    with pytest.raises(ValueError):
        surrogate.fit(grid, branin(grid.points))
        surrogate.predict_gradient([[-19, 5], [3, 3]])
    with pytest.raises(ValueError):
        surrogate.fit(grid, branin(grid.points))
        surrogate.predict_gradient([[-4, -1], [3, 3]])


@pytest.mark.parametrize(
    "surrogate_class,grid_obj",
    [
        (
            smolyay.surrogate.TensorProductSurrogate,
            smolyay.samples.TensorProductPointSet,
        ),
        (
            smolyay.surrogate.SmolyakSparseProductSurrogate,
            smolyay.samples.SmolyakSparseProductPointSet,
        ),
    ],
    ids=["Tensor", "Smolyak"],
)
@pytest.mark.incremental
class TestHessianSize:

    def test_predict_hessian_size_2D(self, surrogate_class, grid_obj):
        """Test predict_hessian returns answer of the appropriate shape."""
        domain = [[-5, 10], [0, 15]]
        num_level = 5

        surrogate, point_sets = create_surrogate(
            surrogate_class,
            smolyay.basis.ChebyshevFirstKind,
            smolyay.samples.NestedClenshawCurtisPointSet,
            num_level,
            domain,
        )
        grid = grid_obj(point_sets=point_sets)
        # fit with same number of points as terms
        surrogate.fit(grid, branin(grid.points))
        assert numpy.array_equal(
            numpy.shape(surrogate.predict_hessian([[-0.5, 0.8], [0, 0], [0.7, 0]])),
            (3, 2, 2),
        )
        assert numpy.array_equal(
            numpy.shape(surrogate.predict_hessian([[0.7, 0]])), (1, 2, 2)
        )

    def test_predict_hessian_size_1D(self, surrogate_class, grid_obj):
        num_level = 4
        domain = [-5, 5]
        surrogate, point_sets = create_surrogate(
            surrogate_class,
            smolyay.basis.ChebyshevFirstKind,
            smolyay.samples.NestedClenshawCurtisPointSet,
            num_level,
            domain,
        )
        # fit with a 1D function
        grid_points = numpy.array(point_sets[0].points, ndmin=2).reshape((-1, 1))
        sample_output = function_2(grid_points)
        surrogate.fit(grid_points, sample_output)
        assert numpy.array_equal(
            numpy.shape(surrogate.predict_hessian([[-0.5], [0], [0.7]])), (3, 1, 1)
        )
        assert numpy.array_equal(numpy.shape(surrogate.predict_hessian([[0.7]])), ())


@pytest.mark.parametrize(
    "surrogate_class,basis_sets",
    [
        (
            smolyay.surrogate.TensorProductSurrogate,
            [
                smolyay.basis.ChebyshevFirstKindBasisFunctionSet(10),
                smolyay.basis.ChebyshevFirstKindBasisFunctionSet(10),
            ],
        ),
        (
            smolyay.surrogate.SmolyakSparseProductSurrogate,
            [
                smolyay.basis.NestedClenshawCurtisBasisFunctionSet(3),
                smolyay.basis.NestedClenshawCurtisBasisFunctionSet(3),
            ],
        ),
    ],
    ids=["Tensor", "Smolyak"],
)
def test_predict_hessian_error(surrogate_class, basis_sets):
    """Test that predict_hessian raises correct errors"""
    surrogate = surrogate_class([[-5, 10], [0, 15]], basis_sets)
    grid = smolyay.samples.LatinHypercubeRandomPointSet([[-5, 10], [0, 15]], 1500, 1234)
    with pytest.raises(RuntimeError):
        surrogate.predict_hessian([[0.7, 0.3]])
    with pytest.raises(ValueError):
        surrogate.fit(grid, branin(grid.points))
        surrogate.predict_hessian([[11, 5]])
    with pytest.raises(ValueError):
        surrogate.fit(grid, branin(grid.points))
        surrogate.predict_hessian([[5, 4], [3, 20], [0, 5]])
    with pytest.raises(ValueError):
        surrogate.fit(grid, branin(grid.points))
        surrogate.predict_hessian([[-19, 5], [3, 3]])
    with pytest.raises(ValueError):
        surrogate.fit(grid, branin(grid.points))
        surrogate.predict_hessian([[-4, -1], [3, 3]])
