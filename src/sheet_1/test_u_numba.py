from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, UGate, CXGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np
from qiskit.circuit.random import random_circuit
from sheet_1.apply_u_numba import apply_u_numba

theta = np.pi/3
phi=np.pi/8
lam=0
apply_to=4
number_of_qubits=9

qc = random_circuit(num_qubits=number_of_qubits, depth=10, measure=False, seed=42)
tqc = transpile(qc, basis_gates=['u', 'cx'])
initial_state = Statevector.from_instruction(tqc)

# simulate result using AerSimulator
qc2 = QuantumCircuit(number_of_qubits)
qc2.u(theta,phi,lam,apply_to)
Aer_result = initial_state.evolve(qc2)

# set up cx gate
u=UGate(theta,phi,lam).to_matrix()

result = apply_u_numba(initial_state.data, u, apply_to)

#print(Aer_result.data)
#print()
#print(result)


global_phase=1

assert np.allclose(Aer_result.data, result*global_phase, atol=1e-12)
