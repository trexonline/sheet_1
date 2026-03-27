from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UGate
from sheet_1.apply_cx_numba import apply_cx_numba
from sheet_1.apply_u_numba import apply_u_numba
import numpy as np

#change apply_cx_on_state to numba version

def simulate_numba(qc: QuantumCircuit, parameters: dict | None = None) -> np.ndarray:
    """Simulates a transpiled quantum circuit using Numba and returns the final state vector.
    The circuit is expected to contain only 'cx' and 'u' gates.
    """
    number_of_qubits = qc.num_qubits
    psi = np.zeros(2**number_of_qubits, dtype=np.complex128)
    psi[0] = 1.0 + 0.0j

    for circuit_instruction in qc.data:
        instr = circuit_instruction.operation
        qargs = circuit_instruction.qubits
        qubit_indices = [qc.find_bit(q).index for q in qargs]

        if instr.name == 'cx':
            psi = apply_cx_numba(psi, qubit_indices[0], qubit_indices[1])
        elif instr.name == 'u':
            u = UGate(instr.params[0], instr.params[1], instr.params[2]).to_matrix()
            psi = apply_u_numba(psi, np.asarray(u), qubit_indices[0])
        else:
            continue
    
    return psi

            







