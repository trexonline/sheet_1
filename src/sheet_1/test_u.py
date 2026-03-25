from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, UGate, CXGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np

from sheet_1 import apply_u_on_state

print()
print('testing')

theta = np.pi/3
phi=np.pi/8
lam=0

# initialize state
number_of_qubits=4
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

print()
print(initial_state.data)
print()
print(Aer_result.data)

# set up cx gate
u=UGate(theta,phi,lam).to_matrix()

initial_state_converted = np.reshape(initial_state.data,[2]*number_of_qubits)
result = apply_u_on_state.apply_u_on_state(initial_state_converted, u, 2)
result_reshaped = np.reshape(result,(2**number_of_qubits))

print()
print(result_reshaped)

global_phase=1

assert np.allclose(Aer_result.data, result_reshaped*global_phase, atol=1e-12)
