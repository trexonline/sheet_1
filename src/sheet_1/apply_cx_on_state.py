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



#--------------------------------------------------------


def map_qubits(number_of_qubits, q):
    return number_of_qubits-1-q

for gate in qc.data:
    acting_on=[qc.find_bit(q).index for q in gate.qubits]


def apply_cx_on_state(state: np.ndarray, cx: np.ndarray, acting_on1: int, acting_on2: int) -> None:
    number_of_qubits=state.ndim
    cx_gate = CXGate()        # Instanz erzeugen
    cx_matrix = cx_gate.to_matrix()  # ruft die Matrix auf
    cx = np.reshape(cx_matrix,(2,2,2,2))
    old_indices = [i for i in range(number_of_qubits)]
    new_indices = old_indices.copy()
    new_indices[acting_on1] = 51
    new_indices[acting_on2] = 50
    result=np.einsum(cx,[51,50,acting_on1,acting_on2],state,old_indices,new_indices) #müsste state nehmen, funktion mit state definiert
    return result





#result=apply_cx_on_state(psi_converted, cx, 2, 0) 
#print(np.reshape(result,(2**n)))
