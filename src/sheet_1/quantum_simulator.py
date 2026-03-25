from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, UGate, CXGate
from qiskit_aer import AerSimulator
from sheet_1.apply_cx_on_state import apply_cx_on_state
from sheet_1.apply_u_on_state import apply_u_on_state
import numpy as np
from qiskit.quantum_info import Statevector



def simulate(qc: QuantumCircuit, parameters: dict | None = None) -> np.ndarray:
    """Simulates the given quantum circuit and returns the final state vector."""
    number_of_qubits=qc.num_qubits
    psi=np.zeros(2**number_of_qubits)
    psi[0]=1
    psi_converted = np.reshape(psi,[2]*number_of_qubits)

    for instr, qargs, cargs in qc.data:
        qubit_indices = [qc.find_bit(q).index for q in qargs]

        if instr.name == 'cx':
            cx=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
            psi_converted=apply_cx_on_state(psi_converted,cx,qubit_indices[0], qubit_indices[1])
        elif instr.name == 'u':
            u=UGate(instr.params[0],instr.params[1],instr.params[2]).to_matrix()
            psi_converted=apply_u_on_state(psi_converted,u,qubit_indices[0])
        else:
            continue
    
    psi_final = np.reshape(np.asarray(psi_converted), (2**number_of_qubits,))
    return psi_final

            

'''# initialize state
number_of_qubits=4
qc = QuantumCircuit(number_of_qubits)
qc.h(0)
qc.h(1)
qc.h(2)
qc.h(3)
qc.append(QFTGate(4), [0, 1, 2, 3])

transpiled_qc = transpile(qc, basis_gates=["cx", "u"])


parameters= {
    'seed': None
}


result = simulate(transpiled_qc, parameters)
'''








