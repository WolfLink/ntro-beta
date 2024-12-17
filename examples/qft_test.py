from __future__ import annotations

from timeit import default_timer as timer

import numpy as np
from bqskit.compiler import Compiler
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate as CXG
from bqskit.ir.gates.constant.sx import SXGate as SXG
from bqskit.ir.gates.parameterized.rz import RZGate as RZG
from bqskit.passes import ForEachBlockPass
from bqskit.passes import GroupSingleQuditGatePass
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes import SetModelPass
from bqskit.passes import UnfoldPass
from bqskit.passes import ZXZXZDecomposition

from ntro import NumericalTReductionPass


def qft(n):
    # this is the qft unitary generator code from qsearch
    root = np.e ** (2j * np.pi / n)
    x = np.fromfunction(lambda x, y: root**(x * y), (n, n))
    return np.array(x) / np.sqrt(n)


if __name__ == '__main__':
    # example: qft circuit
    q = 3
    U = qft(2**q)
    U_S = np.array([[1, 0], [0, 1j]], dtype='complex128')
    start = timer()
    gateset = {CXG(), SXG(), RZG()}

    compiler = Compiler(num_workers=4)
    synthesized_circuit = compiler.compile(
        Circuit.from_unitary(U),
        [
            SetModelPass(MachineModel(q, gate_set=gateset)),
            QSearchSynthesisPass(),
            GroupSingleQuditGatePass(),
            ForEachBlockPass(
                ZXZXZDecomposition(),
                collection_filter=lambda x: x.num_qudits == 1,
            ),
            UnfoldPass(),
        ],
    )
    print(synthesized_circuit.gate_counts)
    print(synthesized_circuit.get_unitary().get_distance_from(U))
    print(f"Synthesis took {timer() - start}s")


    start = timer()
    synthesized_circuit = compiler.compile(
        synthesized_circuit,
        [
            SetModelPass(MachineModel(q, gate_set=gateset)),
            NumericalTReductionPass(full_loops=1),
        ],
    )
    synthesized_circuit.unfold_all()
    print(f"Optimization took {timer() - start}s")

    t_count = 0
    rz_count = 0
    for gate in synthesized_circuit.gate_set:
        print(f"{gate} Count:", synthesized_circuit.count(gate))
        if f"{gate}" in ['TGate', 'TdgGate']:
            t_count += synthesized_circuit.count(gate)
        elif f"{gate}" in ['RZGate']:
            rz_count += synthesized_circuit.count(gate)
    dist = synthesized_circuit.get_unitary().get_distance_from(U, 1)
    print(f"\nDistance: {dist}")
    print(f"T-Count: {t_count}\tRz-Count: {rz_count}")
