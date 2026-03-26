from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, QFTGate
from qiskit.quantum_info import Statevector
import numpy as np

from sheet_1 import apply_cx_on_state


def test_apply_cx_on_state_matches_qiskit_evolution():
	qbit1 = 0
	qbit2 = 2
	number_of_qubits = 3

	qc = QuantumCircuit(number_of_qubits)
	qc.h(0)
	qc.h(1)
	qc.h(2)
	qc.append(QFTGate(3), [0, 1, 2])
	initial_state = Statevector.from_instruction(qc)

	qc2 = QuantumCircuit(number_of_qubits)
	qc2.cx(qbit1, qbit2)
	aer_result = initial_state.evolve(qc2)

	cx = np.reshape(CXGate().to_matrix(), (2, 2, 2, 2))
	initial_state_converted = np.reshape(initial_state.data, [2] * number_of_qubits)
	result = apply_cx_on_state(initial_state_converted, cx, qbit1, qbit2)
	result_reshaped = np.reshape(np.asarray(result), (2**number_of_qubits,))

	assert np.allclose(aer_result.data, result_reshaped, atol=1e-18)
