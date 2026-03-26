from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, UGate, CXGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np
import numba
from sheet_1.apply_u_numba import apply_u_numba


theta = np.pi/3
phi=np.pi/8
lam=0

# initialize state
number_of_qubits=5
qc = QuantumCircuit(number_of_qubits)
qc.x(0)
'''
qc.h(0)
qc.h(1)
qc.h(2)
qc.h(3)
qc.append(QFTGate(4), [0, 1, 2, 3])'''

initial_state = Statevector.from_instruction(qc)

# simulate result using AerSimulator
qc2 = QuantumCircuit(number_of_qubits)
qc2.u(theta,phi,lam,2)
Aer_result = initial_state.evolve(qc2)

# set up cx gate
u=UGate(theta,phi,lam).to_matrix()

result = apply_u_numba(initial_state, u, 2)

global_phase=1

assert np.allclose(Aer_result.data, result*global_phase, atol=1e-12)
