from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
import numpy as np
from qiskit.circuit.random import random_circuit
from sheet_1.apply_cx_numba import apply_cx_numba

control=0
target=1
number_of_qubits=9

qc = random_circuit(num_qubits=number_of_qubits, depth=10, measure=False, seed=42)
tqc = transpile(qc, basis_gates=['u', 'cx'])
initial_state = Statevector.from_instruction(tqc)

# simulate result using AerSimulator
qc2 = QuantumCircuit(number_of_qubits)
qc2.cx(control, target)
Aer_result = initial_state.evolve(qc2)

result = apply_cx_numba(initial_state.data, control, target)

global_phase=1

assert np.allclose(Aer_result.data, result*global_phase, atol=1e-12)
