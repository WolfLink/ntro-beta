from __future__ import annotations

from typing import Sequence

import numpy as np
from bqskit.ir.circuit import Circuit
from bqskit.ir.opt import HilbertSchmidtCostGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator as CostGen
from bqskit.ir.opt.cost.residual import ResidualsFunction
from bqskit.ir.opt.instantiater import Instantiater
from bqskit.ir.opt.minimizer import Minimizer
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.multistartgens.random import RandomStartGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.runtime import get_runtime

from ntro.clift import clifford_gates
from ntro.clift import rz_gates
from ntro.clift import t_gates
from ntro.futurequeue import FutureQueue


def run_minimization(
    circuit: Circuit,
    minimizer: Minimizer,
    cost_gen: CostGen,
    target: UnitaryMatrix | StateVector | StateSystem,
    x0: RealVector,
) -> RealVector:
    """
    Separate minimization function to avoid pickling CostGenerators.

    Args:
        circuit (Circuit): The circuit to optimize.

        minimizer (Minimizer): The minimization algorithm to use.

        cost_gen (CostGen): Generator for the cost function to minimize.

        target (UnitaryMatrix | StateVector | StateSystem): The target
            unitary matrix or state vector.

        x0 (RealVector): Starting parameters for minimization.

    Returns:
        (RealVector): The best parameters found.
    """
    cost = cost_gen.gen_cost(circuit, target)
    return minimizer.minimize(cost, x0)


class MultiStartMinimization(Instantiater):
    def __init__(
        self,
        cost_gen: CostGen = HilbertSchmidtCostGenerator(),
        threshold: float | None = None,
        multistarts: int = 32,
        minimizer: Minimizer = LBFGSMinimizer(),
    ) -> None:
        """
        Construct a new MultiStartMinimization Instantiater.

        Args:
            cost_gen (CostGen): The instantiator will attempt to minimize this
                cost function. (Default: HilbertSchmidtCostGenerator())

            threshold (Optional[float]): If provided, the instantiator has the
                option to return early if the cost function is below this
                threshold. (Default: None)

            multistarts (int): The number of multi-starts to use. (Default: 32)

            minimizer (Minimizer): The minimization algorithm to use.
                (Default: LBFGSMinimizer())
        """
        self.cost_gen = cost_gen
        self.threshold = threshold
        self.multistarts = multistarts
        self.minimizer = minimizer

    def is_capable(self, circuit: Circuit) -> bool:
        """Return True only if all gates in the circuit are Clifford+T+Rz."""
        return all(
            g in clifford_gates + t_gates + rz_gates for g in circuit.gate_set
        )

    def get_violation_report(self, circuit: Circuit) -> str:
        if not self.is_capable(circuit):
            violating_gates = []
            for g in circuit.gate_set:
                if g not in clifford_gates + t_gates + rz_gates:
                    violating_gates.append(g)
            if len(violating_gates) > 0:
                m = f"Found gates ({violating_gates}) in circuit that are"
                m += 'not Clifford+T+Rz'
                return m
        return 'Unknown error'

    def get_method_name(self) -> str:
        return 'multi-start-minimization'

    async def multi_start_instantiate_async(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
        starts: Sequence[RealVector] | None = None,
    ) -> Circuit:
        """
        Run the two-pass minimization multiple times and return best result.

        Args:
            circuit (Circuit): The circuit to optimize.

            target (UnitaryMatrix | StateVector | StateSystem): The target
                unitary matrix or state vector.

            starts (Sequence[RealVector]): The number of times to run the
                two-pass minimization. (Default: None)

        Returns:
            (Circuit) The best circuit found.
        """
        if starts is None:
            starts = RandomStartGenerator().gen_starting_points(
                self.multistarts,
                circuit,
                target,
            )
        elif len(starts) < self.multistarts:
            starts = list(starts)
            num_starts_to_gen = self.multistarts - len(starts)
            new_starts = RandomStartGenerator().gen_starting_points(
                num_starts_to_gen,
                circuit,
                target,
            )
            starts.extend(new_starts)

        num_starts = len(starts)
        result_future = get_runtime().map(
            run_minimization,
            [circuit] * num_starts,
            [self.minimizer] * num_starts,
            [self.cost_gen] * num_starts,
            [target] * num_starts,
            starts,
        )

        cost = self.cost_gen.gen_cost(circuit, target)
        best_result, best_cost = None, None
        async for _, result in FutureQueue(result_future, num_starts):
            distance = cost(result)

            if isinstance(cost, ResidualsFunction):
                distance = np.sum(np.square(distance))

            if best_cost is None or distance < best_cost:
                best_cost = distance
                best_result = result

            if self.threshold is not None and best_cost < self.threshold:
                get_runtime().cancel(result_future)
                break

        circuit.set_params(best_result)
        return circuit

    def instantiate(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
        starts: Sequence[RealVector] | None = None,
    ) -> RealVector | None:
        """
        Instantiate a circuit so that it implements `target`.

        The parameters of the circuit are modified in place.

        Args:
            circuit (Circuit): The circuit to instantiate. If instantiation
                succeeds, the parameters of this circuit will be modified. If
                it does not, the circuit will be left unchanged.

            target (UnitaryMatrix | StateVector | StateSystem): The target
                unitary matrix, state vector, or state system.

            starts (Optional[Sequence[RealVector]]): The starting parameters
                for minimization. Random starts will be generated if None are
                specified. (Default: None)

        Returns:
            (RealVector | None): The best parameters found.
        """
        num_starts = self.multistarts

        if starts is None:
            starts = RandomStartGenerator().gen_starting_points(
                num_starts, circuit, target,
            )

        num_starts = len(starts)
        cost = self.cost_gen.gen_cost(circuit, target)
        best_result, best_cost = None, None

        for start in starts:
            result = run_minimization(
                circuit,
                self.minimizer,
                self.cost_gen,
                target,
                start,
            )
            distance = cost(result)
            if best_cost is None or distance < best_cost:
                best_cost = distance
                best_result = result
            if self.threshold is not None and best_cost < self.threshold:
                break

        if best_cost is None or self.threshold is not None and \
                best_cost >= self.threshold:
            return None

        circuit.set_params(best_result)
        return best_result
