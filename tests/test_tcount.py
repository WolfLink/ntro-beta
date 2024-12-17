from __future__ import annotations

import numpy as np
from bqskit.ir.circuit import Circuit
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from numpy.random import choice
from numpy.random import randint
from numpy.random import randn

from ntro.tcount import RelaxedTCountCostGenerator
from ntro.tcount import RoundSmallestNCostGenerator


def parameter_vector(
    num_params: int,
    period: float,
    num_rounded_params: int,
    rounded_offset: float = 0.0,
) -> np.ndarray:
    params = np.array([randn() for _ in range(num_params)])
    num_bins = int(np.ceil(2 * np.pi / period))
    to_round = choice(num_params, size=num_rounded_params, replace=False)
    for i in to_round:
        if randint(0, 2) == 1:
            params[i] = randint(1, num_bins + 1) * period + rounded_offset
    return params


def triangular_shift(params: np.ndarray, period: float) -> np.ndarray:
    shifted_params = np.mod(params - period / 2, period)
    deviation = np.abs(shifted_params - period / 2)
    return (2 / period) * deviation


class TestTCount:
    def test_RelaxedTCountCost(self) -> None:
        num_qudits = 3
        period_2 = np.pi / 2
        period_4 = np.pi / 4
        period_8 = np.pi / 8

        circuit = Circuit(num_qudits)
        target = UnitaryMatrix.random(num_qudits)
        cost_2 = RelaxedTCountCostGenerator(period_2).gen_cost(circuit, target)
        cost_4 = RelaxedTCountCostGenerator(period_4).gen_cost(circuit, target)
        cost_8 = RelaxedTCountCostGenerator(period_8).gen_cost(circuit, target)

        num_params = 32
        num_rounded_params = 15
        params_2 = parameter_vector(num_params, period_2, num_rounded_params)
        params_4 = parameter_vector(num_params, period_4, num_rounded_params)
        params_8 = parameter_vector(num_params, period_8, num_rounded_params)

        val_2_2 = cost_2.get_cost(params_2)
        val_2_4 = cost_4.get_cost(params_2)
        val_2_8 = cost_8.get_cost(params_2)
        val_4_4 = cost_4.get_cost(params_4)
        val_4_8 = cost_8.get_cost(params_4)
        val_8_8 = cost_8.get_cost(params_8)

        assert np.isclose(val_2_2, triangular_shift(params_2, period_2).sum())
        assert np.isclose(val_2_4, triangular_shift(params_2, period_4).sum())
        assert np.isclose(val_2_8, triangular_shift(params_2, period_8).sum())
        assert np.isclose(val_4_4, triangular_shift(params_4, period_4).sum())
        assert np.isclose(val_4_8, triangular_shift(params_4, period_8).sum())
        assert np.isclose(val_8_8, triangular_shift(params_8, period_8).sum())

    def test_RoundSmallestNCost(self) -> None:
        num_qudits = 3
        period_2 = np.pi / 2
        period_4 = np.pi / 4
        period_8 = np.pi / 8
        num_params, num_rounded_params, offset = 32, 15, 0.05

        circuit = Circuit(num_qudits)
        target = UnitaryMatrix.random(num_qudits)
        cost_gen_2 = RoundSmallestNCostGenerator(num_rounded_params, period_2)
        cost_gen_4 = RoundSmallestNCostGenerator(num_rounded_params, period_4)
        cost_gen_8 = RoundSmallestNCostGenerator(num_rounded_params, period_8)
        cost_2 = cost_gen_2.gen_cost(circuit, target)
        cost_4 = cost_gen_4.gen_cost(circuit, target)
        cost_8 = cost_gen_8.gen_cost(circuit, target)

        params_2 = parameter_vector(
            num_params,
            period_2,
            num_rounded_params,
            offset,
        )
        params_4 = parameter_vector(
            num_params,
            period_4,
            num_rounded_params,
            offset,
        )
        params_8 = parameter_vector(
            num_params,
            period_8,
            num_rounded_params,
            offset,
        )

        val_2_2 = cost_2.get_cost(params_2)
        val_2_4 = cost_4.get_cost(params_2)
        val_2_8 = cost_8.get_cost(params_2)
        val_4_4 = cost_4.get_cost(params_4)
        val_4_8 = cost_8.get_cost(params_4)
        val_8_8 = cost_8.get_cost(params_8)

        shift_2_2 = triangular_shift(params_2, period_2).sum()
        shift_2_4 = triangular_shift(params_2, period_4).sum()
        shift_2_8 = triangular_shift(params_2, period_8).sum()
        shift_4_4 = triangular_shift(params_4, period_4).sum()
        shift_4_8 = triangular_shift(params_4, period_8).sum()
        shift_8_8 = triangular_shift(params_8, period_8).sum()

        assert val_2_2 <= shift_2_2
        assert val_2_4 <= shift_2_4
        assert val_2_8 <= shift_2_8
        assert val_4_4 <= shift_4_4
        assert val_4_8 <= shift_4_8
        assert val_8_8 <= shift_8_8
