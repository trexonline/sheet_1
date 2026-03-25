from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, UGate, CXGate
from qiskit_aer import AerSimulator
import numpy as np


def simulate_mock(qc, parameters):
    simulator = AerSimulator()
    job = simulator.run(qc, shots=parameters["shots"],seed_simulator=parameters["seed"])
    result = job.result()
    counts = result.get_counts()
    return counts

def simulate(qc, parameters):
    for instr, qargs, cargs in qc.data:
        print("Gate:", instr.name)
        qubit_indices = [qc.find_bit(q).index for q in qargs]
        print("Qubits:",qubit_indices)
        print("Params:", instr.params)
        print()

        #if instr.name == 'cx':
            


qc = QuantumCircuit(4)
#----------------------------------------------------------------------------------------


for gate in qc.data:
    acting_on=[qc.find_bit(q).index for q in gate.qubits]


#def vector_to_tensor(psi, number_of_qubits):
 #   return psi.reshape([2]*number_of_qubits).transpose(list(reversed(range(number_of_qubits))))

#def tensor_to_vector(tensor, number_of_qubits):
 #   return tensor.transpose(list(reversed(range(number_of_qubits)))).reshape(2**number_of_qubits)

def apply_u_on_state(state: np.ndarray, u: np.ndarray, acting_on: int) -> None:
    number_of_qubits=state.ndim
    acting_on=number_of_qubits-acting_on-1
    print(acting_on)
    old_indices = [i for i in range(number_of_qubits)]
    new_indices = old_indices.copy()
    new_indices[acting_on] = 51
    result=np.einsum(u,[51,acting_on],state,old_indices,new_indices)
    return result

u=UGate(np.pi/4,0,0).to_matrix()