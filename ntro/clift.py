from __future__ import annotations

import numpy as np
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import CZGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import SdgGate
from bqskit.ir.gates import SGate
from bqskit.ir.gates import SqrtXGate
from bqskit.ir.gates import SwapGate
from bqskit.ir.gates import TdgGate
from bqskit.ir.gates import TGate
from bqskit.ir.gates import XGate
from bqskit.ir.gates import YGate
from bqskit.ir.gates import ZGate


clifford_gates = [
    CNOTGate(),
    CZGate(),
    HGate(),
    SdgGate(),
    SGate(),
    SqrtXGate(),
    SwapGate(),
    XGate(),
    YGate(),
    ZGate(),
]

t_gates = [TGate(), TdgGate()]

rz_gates = [RZGate()]


def circuit_for_rounded_val(val: float, enable_t: bool) -> CircuitGate:
    """
    Returns a CircuitGate with a {Z, S, Sdg, T, Tdg} depending on val.

    Args:
        val (float): A parameter value to round.

        period (float): The period of the rounding function.

    Returns:
        (CircuitGate): A CircuitGate containing a Z rotation gate depending
            on val.
    """
    circuit = Circuit(1)
    val = val % (2 * np.pi)
    if not enable_t:
        # cliffords
        rounded_val = np.round(val * 2 / np.pi)
        if rounded_val == 1:
            circuit.append_gate(SGate(), 0)
        elif rounded_val == 2:
            circuit.append_gate(ZGate(), 0)
        elif rounded_val == 3:
            circuit.append_gate(SdgGate(), 0)
    else:
        rounded_val = np.round(val * 4 / np.pi)
        if rounded_val < 4:
            if rounded_val >= 2:
                circuit.append_gate(SGate(), 0)
            if rounded_val % 2:
                circuit.append_gate(TGate(), 0)
        elif rounded_val > 4:
            if rounded_val <= 6:
                circuit.append_gate(SdgGate(), 0)
            if rounded_val % 2:
                circuit.append_gate(TdgGate(), 0)
        elif rounded_val == 4:
            circuit.append_gate(ZGate(), 0)
    return CircuitGate(circuit)
