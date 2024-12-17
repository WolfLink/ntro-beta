"""This module implements the RzToTPass."""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from bqskit import Circuit
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.gates import RZGate

from ntro.clift import circuit_for_rounded_val

_logger = logging.getLogger(__name__)


class RzToTPass(BasePass):
    def __init__(
        self,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        conversion_method: Callable[[float], Circuit] | None = None,
    ) -> None:
        """
        Contructor for the RzToTPass.

        Args:
            rtol (float): The relative tolerance parameter for angle rounding.
                (Default: 1e-5)

            atol (float): The absolute tolerance parameter for angle rounding.
                (Default: 1e-8)

            conversion_method (Optional[Callable[[float], Circuit]]): A map
                from Rz angles to circuits. If None, no conversion is done but
                angles will still be rounded. (Default: None)
        """
        self.rtol = rtol
        self.atol = atol
        self.conversion_method = conversion_method

    async def run(self, circuit: Circuit, data: PassData) -> None:
        # Find RZGates
        points = [
            (cycle, op.location[0])
            for cycle, op in circuit.operations_with_cycles()
            if isinstance(op.gate, RZGate)
        ]

        shift = 0
        # Round angles and optionally replace RZGates
        for point in points:
            point = (point[0] - shift, point[1])

            op = circuit[point]
            og_cycle_count = circuit.num_cycles

            val = op.params[0]
            rounded_val = np.round(val * 4 / np.pi) * np.pi / 4
            if np.isclose(val, rounded_val, self.rtol, self.atol):
                rounded = circuit_for_rounded_val(val % (np.pi * 2), True)
                circuit.replace_gate(point, rounded, op.location)
            elif self.conversion_method:
                new_gate = self.conversion_method(val % (np.pi * 2))
                circuit.replace_gate(point, new_gate, op.location)

            shift += og_cycle_count - circuit.num_cycles
