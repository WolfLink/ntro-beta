"""This module implements the NumericalTReductionPass."""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost import HilbertSchmidtCostGenerator
from bqskit.ir.opt.cost import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.minimizers.ceres import CeresMinimizer
from bqskit.qis.unitary import RealVector
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.runtime import get_runtime

from ntro.clift import circuit_for_rounded_val
from ntro.clift import clifford_gates
from ntro.clift import rz_gates
from ntro.clift import t_gates
from ntro.multi_start_minimization import MultiStartMinimization
from ntro.tcount import get_arr
from ntro.tcount import RoundSmallestNCostGenerator
from ntro.tcount import RoundSmallestNResidualsGenerator
from ntro.tcount import SumCostGenerator
from ntro.tcount import SumResidualsGenerator

_logger = logging.getLogger(__name__)


class NumericalTReductionPass(BasePass):
    def __init__(
        self,
        success_threshold: float = 1e-8,
        full_loops: int = 1,
        first_pass_multistarts: int = 128,
        second_pass_multi_starts: int = 16,
        target_periods: Sequence[float] | None = None,
    ) -> None:
        """
        Construct a NumericalTReductionPass.

        Args:
            success_threshold (float): The synthesis success threshold.
                (Default: 1e-8)

            full_loops (int): Number of times to run rounding style instant-
                iation from generated starting points. (Default: 1)

            first_pass_multistarts (int): Number of starting points to use
                for the first pass of optimization. This will help find
                multiple starting points for the second pass. (Default: 128)

            second_pass_multi_starts (int): Number of starting points to use
                for the second pass of optimization. (Default: 16)

            target_periods (Optional[Sequence[float]]): The target periods
                for the optimization. (Default: [np.pi/2, np.pi/4])
        """
        if target_periods is None:
            target_periods = [0.5 * np.pi, 0.25 * np.pi]

        self.success_threshold = success_threshold
        self.full_loops = full_loops
        self.first_pass_multistarts = first_pass_multistarts
        self.second_pass_multi_starts = second_pass_multi_starts
        self.target_periods = target_periods

        self.acceptable_gates = clifford_gates + t_gates + rz_gates

    async def optimize_for_period(
        self,
        circuit: Circuit,
        target: UnitaryMatrix,
        period: float,
    ) -> Circuit:
        """
        Minimization attempts to find the largest subset of parameters that can
        be rounded to multiples of `period` without increasing the cost beyond
        the success threshold.

        Args:
            circuit (Circuit): The circuit to optimize.

            target (UnitaryMatrix): The target unitary.

            period (float): Push parameters towards multiples of this period.
        """
        trial_circuit = circuit.copy()
        high, low = len(circuit.params) + 1, 0

        d_gen = HilbertSchmidtCostGenerator()
        d_res = HilbertSchmidtResidualsGenerator()
        best_params, best_N = circuit.params, 0
        first_min = CeresMinimizer()

        # Find the largest subset of parameters that can be rounded to multiples
        # of `period` without increasing the cost beyond the success threshold.
        while high > low:
            N = (low + high) // 2
            n_gen = RoundSmallestNCostGenerator(N, period)
            n_res = RoundSmallestNResidualsGenerator(N, period)
            sum_gen = SumCostGenerator(d_gen, n_gen)
            sum_res = SumResidualsGenerator(d_res, n_res)
            trial_params = first_min.minimize(
                sum_res.gen_cost(trial_circuit, target),
                best_params,
            )
            score = sum_gen.gen_cost(trial_circuit, target)(trial_params)
            # If the rounded circuit does not meet the success threshold, try
            # running full minimization.
            if score >= self.success_threshold:
                miser = MultiStartMinimization(
                    cost_gen=sum_res,
                    threshold=self.success_threshold,
                    multistarts=self.first_pass_multistarts,
                    minimizer=CeresMinimizer(),
                )
                result = await miser.multi_start_instantiate_async(
                    trial_circuit,
                    target,
                )
                trial_params = result.params
                score = sum_gen.gen_cost(trial_circuit, target)(trial_params)

            if score >= self.success_threshold:
                high = N - 1
            else:
                low = N + 1
                best_params = trial_params
                best_N = N

        best_circuit = circuit.copy()
        best_circuit.set_params(best_params)
        # Replace rounded parameters with discrete gates
        for _ in range(best_N):
            if len(best_circuit.params) < 1:
                break
            index = np.argmin(get_arr(trial_circuit.params, period))
            trial_circuit = best_circuit.copy()
            for cycle, op in trial_circuit.operations_with_cycles():
                if len(op.params) != 1:
                    continue
                if op.params[0] == trial_circuit.params[index]:
                    t_replace = period < np.pi * 0.5
                    rounded = circuit_for_rounded_val(op.params[0], t_replace)
                    trial_circuit.replace_gate(
                        (cycle, op.location[0]),
                        rounded,
                        op.location,
                    )
                    break
            best_circuit = trial_circuit

        # Update un-rounded parameters
        cost = HilbertSchmidtResidualsGenerator().gen_cost(best_circuit, target)
        mizer = CeresMinimizer(ftol=5e-16, gtol=1e-15)
        test_params = mizer.minimize(cost, best_circuit.params)
        best_circuit.set_params(test_params)
        return best_circuit

    async def optimize_all_periods(
        self,
        circuit: Circuit,
        target: UnitaryMatrix,
        x0: RealVector,
    ) -> Circuit:
        """Starting from the largest period, try rounding parameters to
        multiples of some valid_period."""
        candidate_circuit = circuit.copy()
        candidate_circuit.set_params(x0)
        for period in self.target_periods:
            result = await get_runtime().submit(
                self.optimize_for_period,
                candidate_circuit,
                target,
                period,
            )
            candidate_circuit = candidate_circuit if result is None else result
        if candidate_circuit is not None:
            candidate_circuit.unfold_all()
        return candidate_circuit

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """
        Re-instantiate the parameters of `circuit` to minimize the number of
        RZGates.

        RZGates will be pushed towards multiples of `target_period`
        while still fitting the Hilbert-Schmidt distance constraint.
        """
        # Check that circuit has been converrted to Clifford+T+Rz
        if any(g not in self.acceptable_gates for g in circuit.gate_set):
            m = (
                'Circuit must be converted to Clifford+T+Rz before running'
                f" NumericalTReductionPass. Got {circuit.gate_set}."
            )
            raise ValueError(m)

        utry = circuit.get_unitary()
        best_circuit = circuit
        best_circuit.unfold_all()
        candidate_circuit = best_circuit

        def update_stats(circuit: Circuit) -> tuple[int, int, float]:
            rz = sum(circuit.count(gate) for gate in rz_gates)
            t = sum(circuit.count(gate) for gate in t_gates)
            d = circuit.get_unitary().get_distance_from(utry, degree=1)
            return rz, t, d

        param_list = [circuit.params]
        # If multiple full loops will be executed, generate multiple valid
        # starting points for optimization.
        if self.full_loops > 1:
            miser = MultiStartMinimization(
                cost_gen=HilbertSchmidtResidualsGenerator(),
                minimizer=CeresMinimizer(),
                threshold=self.success_threshold,
                multistarts=self.second_pass_multi_starts,
            )
            seed_circuits = await get_runtime().map(
                miser.multi_start_instantiate_async,
                [candidate_circuit] * (self.full_loops - 1),
                [utry] * (self.full_loops - 1),
            )
            param_list.extend(
                [seed_circuit.params for seed_circuit in seed_circuits],
            )

        candidate_circuits: list[Circuit] = await get_runtime().map(
            self.optimize_all_periods,
            [best_circuit] * self.full_loops,
            [utry] * self.full_loops,
            param_list,
        )

        for candidate_circuit in candidate_circuits:
            curr_rz, curr_t, curr_d = update_stats(candidate_circuit)
            best_rz, best_t, best_d = update_stats(best_circuit)

            if curr_d >= self.success_threshold or curr_rz > best_rz:
                continue

            if curr_rz < best_rz:
                best_circuit = candidate_circuit
            elif curr_rz == best_rz and curr_t < best_t:
                best_circuit = candidate_circuit
            elif curr_rz == best_rz and curr_t == best_t and curr_d < best_d:
                best_circuit = candidate_circuit

        circuit.become(best_circuit)
