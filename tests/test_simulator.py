from qiskit import QuantumCircuit, transpile
from sheet_1 import quantum_simulator
from qiskit.quantum_info import Statevector
import numpy as np
from qiskit.circuit.random import random_circuit
import pytest

def construct_simple_circuit():
    number_of_qubits = 3
    qc = QuantumCircuit(number_of_qubits)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    return qc


def construct_random_circuit():
    return random_circuit(num_qubits=4, depth=10, measure=False)


@pytest.mark.parametrize("construction_function", [construct_simple_circuit, construct_random_circuit])
def test_comparision(construction_function):
    qc = construction_function()
    qc = transpile(qc, basis_gates=["u", "cx"])

    aer_state = Statevector.from_instruction(qc)
    own_state = quantum_simulator.simulate(qc, None)
    idx = np.argmax(np.abs(aer_state))
    global_phase = aer_state.data[idx] / own_state[idx]
    assert np.allclose(aer_state, own_state * global_phase)
