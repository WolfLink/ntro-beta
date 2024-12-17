from __future__ import annotations

import numpy as np
import numpy.typing as npt
from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost import CostFunction
from bqskit.ir.opt.cost.differentiable import DifferentiableCostFunction
from bqskit.ir.opt.cost.differentiable import DifferentiableResidualsFunction
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


def get_arr(
    params: npt.NDArray[np.float64],
    period: float,
) -> npt.NDArray[np.float64]:
    shifted_params = np.mod(params - period / 2, period)
    deviation = np.abs(shifted_params - period / 2)
    return (2 / period) * deviation


class RelaxedTCountCostGenerator(CostFunctionGenerator):
    def __init__(self, period: float = np.pi / 4) -> None:
        """
        Constructor for RelaxedTCountCostGenerator.

        Args:
            period (float): The period of the triangle wave cost function.
                Parameter values will be pushed towards a multiple this
                (Default: np.pi / 4)
        """
        super().__init__()
        self.period = period

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> CostFunction:
        return RelaxedTCountCost(self.period)


class RelaxedTCountCost(DifferentiableCostFunction):
    """
    A CostFunction that pushes parameters towards multiples of `period`.

    Sweeping across the domain of [0, 2pi), this cost function looks like
    a triangle wave. Valley points are at multiples of `period` and peak
    points are at multiples of `period / 2`.
    """

    def __init__(self, period: float) -> None:
        super().__init__()
        self.period = period

    def get_cost(self, params: RealVector) -> float:
        if len(params) < 1:
            return 0
        if not isinstance(params, np.ndarray):
            params = np.array(params)
        deviation = get_arr(params, self.period)
        cost = np.sum(deviation)
        return float(cost)

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        deviation = np.mod(params, self.period) - self.period / 2
        signs = np.sign(deviation)
        return -1 * (2 / self.period) * signs


class SumCostGenerator(CostFunctionGenerator):
    def __init__(
        self,
        A: CostFunctionGenerator,
        B: CostFunctionGenerator,
    ) -> None:
        super().__init__()
        self.A = A
        self.B = B

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> CostFunction:
        cost_a = self.A.gen_cost(circuit, target)
        cost_b = self.B.gen_cost(circuit, target)
        return SumCost(cost_a, cost_b)


class SumCost(DifferentiableCostFunction):
    """A CostFunction that is the sum of two other CostFunctions."""

    def __init__(
        self,
        A: DifferentiableCostFunction,
        B: DifferentiableCostFunction,
    ) -> None:
        super().__init__()
        self.A = A
        self.B = B

    def get_cost(self, params: RealVector) -> float:
        return self.A.get_cost(params) + self.B.get_cost(params)

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        return self.A.get_grad(params) + self.B.get_grad(params)


class SumResidualsGenerator(CostFunctionGenerator):
    def __init__(
        self,
        A: CostFunctionGenerator,
        B: CostFunctionGenerator,
    ) -> None:
        super().__init__()
        self.A = A
        self.B = B

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
    ) -> CostFunction:
        cost_a = self.A.gen_cost(circuit, target)
        cost_b = self.B.gen_cost(circuit, target)
        return SumResiduals(cost_a, cost_b, circuit.params)


class SumResiduals(DifferentiableResidualsFunction):
    def __init__(
        self,
        A: DifferentiableResidualsFunction,
        B: DifferentiableResidualsFunction,
        test_params: RealVector,
    ) -> None:
        super().__init__()
        self.A = A
        self.B = B
        self.test_params = test_params

    def get_cost(self, params: RealVector) -> float:
        return np.sum(np.square(self.get_residuals(params)))

    def num_residuals(self) -> int:
        try:
            numa = self.A.num_residuals()
        except AttributeError:
            numa = len(self.A.get_residuals(self.test_params))
        try:
            numb = self.B.num_residuals()
        except AttributeError:
            numb = len(self.B.get_residuals(self.test_params))
        return numa + numb

    def get_residuals(self, params: RealVector) -> RealVector:
        res_a = self.A.get_residuals(params)
        res_b = self.B.get_residuals(params)
        return np.concatenate((res_a, res_b), axis=None)

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        grad_a = self.A.get_grad(params)
        grad_b = self.B.get_grad(params)
        return np.concatenate((grad_a, grad_b), axis=0)


class RoundSmallestNCostGenerator(CostFunctionGenerator):
    def __init__(self, N: int, period: float = np.pi / 4) -> None:
        """
        Constructor for RelaxedTCountCostGenerator.

        Args:
            period (float): The period of the triangle wave cost function.
                Parameter values will be pushed towards a multiple this
                (Default: np.pi / 4)
        """
        super().__init__()
        self.period = period
        self.N = N

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> CostFunction:
        return RoundSmallestNCost(self.N, self.period)

    # TODO probably need an argsort or something for the grad etc.


class RoundSmallestNCost(DifferentiableCostFunction):
    """
    Computes the cost of rounding the smallest N values of the triangle wave.

    Cost here refers to change in Hilbert-Schmidt distance, as opposed to the
    magnitude of the change in parameter space.
    """

    def __init__(self, N: int, period: float) -> None:
        super().__init__()
        self.period = period
        self.N = N

    def get_cost(self, params: RealVector) -> float:
        if len(params) < 1 or self.N < 1:
            return 0
        if not isinstance(params, np.ndarray):
            params = np.array(params)
        deviation = get_arr(params, self.period)
        deviation = 0.5 - 0.5 * np.cos(deviation)
        deviation = np.sort(deviation)

        cost = np.sum(deviation[: self.N])
        return float(cost)

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        if self.N < 1:
            return np.zeros_like(params)
        deviation = np.mod(params, self.period) - self.period / 2
        signs = np.sign(deviation)

        deviation = get_arr(params, self.period)
        mask = np.zeros_like(params)
        sort_dev = 0.5 - 0.5 * np.cos(deviation)
        indices = np.argsort(sort_dev)
        mask[indices[:self.N]] = 1

        mult = 0.5 * np.sin(deviation)
        return -1 * (2 / self.period) * signs * mult * mask


class RoundSmallestNResidualsGenerator(CostFunctionGenerator):
    def __init__(self, N: int, period: float) -> None:
        super().__init__()
        self.period = period
        self.N = N

    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
    ) -> CostFunction:
        return RoundSmallestNResiduals(self.N, self.period, circuit.dim)

    # TODO probably need an argsort or something for the grad etc.


class RoundSmallestNResiduals(DifferentiableResidualsFunction):
    def __init__(self, N: int, period: float, dim: int) -> None:
        super().__init__()
        self.period = period
        self.N = N
        self.dim = dim

    def get_cost(self, params: RealVector) -> float:
        return np.sum(np.square(self.get_residuals(params)))

    def num_residuals(self) -> float:
        return self.N

    def get_residuals(self, params: RealVector) -> npt.NDArray[np.float64]:
        if len(params) < 1 or self.N < 1:
            return np.array([], dtype=np.float64)
        if not isinstance(params, np.ndarray):
            params = np.array(params)
        deviation = get_arr(params, self.period)
        deviation = 0.5 - 0.5 * np.cos(deviation)
        deviation = np.sort(deviation)
        return np.array(self.dim * deviation[:self.N], dtype=np.float64)

    def get_grad(self, params: RealVector) -> npt.NDArray[np.float64]:
        if self.N < 1:
            return np.zeros((self.N, len(params)))
        deviation = np.mod(params, self.period) - self.period / 2
        signs = np.sign(deviation)

        deviation = get_arr(params, self.period)
        sort_dev = 0.5 - 0.5 * np.cos(deviation)
        indices = np.argsort(sort_dev)

        output = np.zeros((self.N, len(params)))
        for i in range(self.N):
            j = indices[i]
            sin = 0.5 * np.sin(deviation[j])
            output[i][j] = -1 * (2 / self.period) * signs[j] * sin
        return self.dim * output
