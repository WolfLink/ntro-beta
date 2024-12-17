"""This module implements the ConvertToCliffordPlusTPlusRZPass class."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import CZGate
from bqskit.ir.gates import SwapGate
from bqskit.passes.alias import PassAlias
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.passes.control.ifthenelse import IfThenElsePass
from bqskit.passes.control.predicate import PassPredicate
from bqskit.passes.group import PassGroup
from bqskit.passes.noop import NOOPPass
from bqskit.passes.partitioning import GroupSingleQuditGatePass
from bqskit.passes.partitioning import QuickPartitioner
from bqskit.passes.rules import ZXZXZDecomposition
from bqskit.passes.synthesis import QSearchSynthesisPass
from bqskit.passes.util.unfold import UnfoldPass


class CliffordGateSetPredicate(PassPredicate):
    """Predicate that returns true if all multi-qubit gates are clifford."""

    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        """Return the truth value, see :class:`PassAlias` for more."""
        return all(
            g in [CNOTGate(), SwapGate(), CZGate()]
            for g in circuit.gate_set
            if g.num_qudits >= 2
        )


class ConvertToCliffordPlusTPlusRZPass(PassAlias):
    """Retarget a circuit's gates to be Clifford + T + Rz."""

    def __init__(self, optimization_level: int = 0) -> None:
        """
        Initialize the pass.

        Args:
            optimization_level (int): How much optimization to perform
                during gate set transpilation. Level 0 does ... # TODO
        """
        if optimization_level < 0 or optimization_level > 4:
            raise ValueError('Expected optimization level 0, 1, 2, 3 or 4.')

        # TODO: Resynthesis should take into account optimization_level
        self.optimization_level = optimization_level
        self.resynthesis = PassGroup(
            [
                QuickPartitioner(3),
                ForEachBlockPass(QSearchSynthesisPass()),
                UnfoldPass(),
            ],
        )

    def get_passes(self) -> list[BasePass]:
        """Return the aliased workflow, see :class:`PassAlias` for more."""
        return [
            # To start, unfold, removing any existing block structure
            UnfoldPass(),
            # If target two-qubit gates are not clifford, resynthesize
            IfThenElsePass(
                CliffordGateSetPredicate(),
                [NOOPPass()],  # Don't do anything if cxs and swaps
                [self.resynthesis],  # Otherwise resynthesize to cxs
            ),
            # Convert all single-qubit gates to ZXZXZ
            GroupSingleQuditGatePass(),
            ForEachBlockPass(ZXZXZDecomposition()),
            UnfoldPass(),
            # TODO scanning gate removal if optimization level high enough
        ]
