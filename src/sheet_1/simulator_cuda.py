from qiskit import QuantumCircuit
from qiskit.circuit.library import UGate
import numpy as np

from sheet_1.apply_cx_cuda import apply_cx_cuda
from sheet_1.apply_u_cuda import apply_u_cuda


def simulate_cuda(qc: QuantumCircuit, parameters: dict | None = None) -> np.ndarray:
    """Simulates a transpiled quantum circuit using CUDA gate kernels and returns the final state vector.
    The circuit is expected to contain only 'cx' and 'u' gates.
    """
    number_of_qubits = qc.num_qubits
    psi = np.zeros(2**number_of_qubits, dtype=np.complex128)
    psi[0] = 1.0 + 0.0j

    for circuit_instruction in qc.data:
        instr = circuit_instruction.operation
        qargs = circuit_instruction.qubits
        qubit_indices = [qc.find_bit(q).index for q in qargs]

        if instr.name == "cx":
            psi = apply_cx_cuda(psi, qubit_indices[0], qubit_indices[1])
        elif instr.name == "u":
            u = UGate(instr.params[0], instr.params[1], instr.params[2]).to_matrix()
            psi = apply_u_cuda(psi, np.asarray(u, dtype=np.complex128), qubit_indices[0])
        else:
            continue

    return psi

