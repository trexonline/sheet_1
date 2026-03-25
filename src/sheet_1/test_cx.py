from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, UGate, CXGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np

from sheet_1 import apply_cx_on_state


qbit1=0
qbit2=2

# initialize state
number_of_qubits=3
qc = QuantumCircuit(number_of_qubits)
#qc.x(0)
qc.h(0)
qc.h(1)
qc.h(2)
qc.append(QFTGate(3), [0, 1, 2])

initial_state = Statevector.from_instruction(qc)

# simulate result using AerSimulator
qc2 = QuantumCircuit(number_of_qubits)
qc2.cx(qbit1,qbit2)
Aer_result = initial_state.evolve(qc2)

# set up cx gate
cx_gate = CXGate()        # Instanz erzeugen
cx_matrix = cx_gate.to_matrix()  # ruft die Matrix auf
cx = np.reshape(cx_matrix,(2,2,2,2))

initial_state_converted = np.reshape(initial_state.data,[2]*number_of_qubits)
result = apply_cx_on_state(initial_state_converted, cx, qbit1,qbit2)
result_reshaped = np.reshape(result,(2**number_of_qubits))

global_phase=1

assert np.allclose(Aer_result.data, result_reshaped*global_phase, atol=1e-18)
