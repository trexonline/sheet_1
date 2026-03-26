from qiskit import QuantumCircuit
from qiskit.circuit.library import UGate
from qiskit.quantum_info import Statevector
import numpy as np

from sheet_1 import apply_u_on_state


def test_apply_u_on_state_matches_qiskit_evolution():
	theta = np.pi / 3
	phi = np.pi / 8
	lam = 0
	number_of_qubits = 5

	qc = QuantumCircuit(number_of_qubits)
	qc.x(0)
	initial_state = Statevector.from_instruction(qc)

	qc2 = QuantumCircuit(number_of_qubits)
	qc2.u(theta, phi, lam, 2)
	aer_result = initial_state.evolve(qc2)

	u = UGate(theta, phi, lam).to_matrix()
	initial_state_converted = np.reshape(initial_state.data, [2] * number_of_qubits)
	result = apply_u_on_state(initial_state_converted, u, 2)
	result_reshaped = np.reshape(np.asarray(result), (2**number_of_qubits,))

	assert np.allclose(aer_result.data, result_reshaped, atol=1e-12)
