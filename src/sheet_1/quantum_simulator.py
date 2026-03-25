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

def apply_u_on_state(state: np.ndarray, u: np.ndarray, acting_on: int) -> None:
    number_of_qubits=state.ndim
    old_indices = [i for i in range(number_of_qubits)]
    new_indices = old_indices.copy()
    new_indices[acting_on] = 51
    result=np.einsum(u,[51,acting_on],psi_converted,old_indices,new_indices)
    return result

def apply_cx_on_state(state: np.ndarray, cx: np.ndarray, acting_on1: int, acting_on2: int) -> None:
    number_of_qubits=state.ndim
    old_indices = [i for i in range(number_of_qubits)]
    new_indices = old_indices.copy()
    new_indices[acting_on1] = 51
    new_indices[acting_on2] = 50
    result=np.einsum(cx,[51,50,acting_on1,acting_on2],psi_converted,old_indices,new_indices)
    return result
            

n=3

psi = np.zeros(2**n)
psi[4]=1


X = np.zeros([2]*2)
X[0,1]=1
X[1,0]=1

psi_converted = np.zeros([2]*n)
for idx, value in np.ndenumerate(psi_converted):
    idx2=0
    for i in range(n):
        idx2+=idx[i]*2**(n-i-1)
    psi_converted[idx]=psi[idx2]

u=UGate(np.pi/4,0,0).to_matrix()

cx_gate = CXGate()        # Instanz erzeugen
cx_matrix = cx_gate.to_matrix()  # ruft die Matrix auf


cx = np.reshape(cx_matrix,(2,2,2,2))
#print(cx)
print(psi)
print()

#print(apply_u_on_state(psi_converted, u, 0).flatten())
result=apply_cx_on_state(psi_converted, cx, 2, 0)
print(np.reshape(result,(2**n)))



qc = QuantumCircuit(4)
#qc.x(3) # |1000> state

qc.h(0)
qc.cx(0,1)
qc.cx(1,2)

#qc.h(0) # 1/4*(|0> + |1>)^4 state
#qc.h(1)
#qc.h(2)
#qc.h(3)

#qc.append(QFTGate(4), [0, 1, 2, 3])
#qc.measure_all()
transpiled_qc = transpile(qc, basis_gates=["cx", "u"])


parameters= {
    'shots':1000,
    'seed': None
}


#result = simulate(transpiled_qc, parameters)
#simulate(transpiled_qc, parameters)







