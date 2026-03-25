from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, UGate, CXGate
from qiskit_aer import AerSimulator
import numpy as np

from sheet_1 import quantum_simulator


n=4

qc = QuantumCircuit(n)

qc.h(0)
qc.h(1)
qc.h(2)
qc.h(3)

qc.append(QFTGate(4), [0, 1, 2, 3])
qc.measure_all()
transpiled_qc = transpile(qc, basis_gates=["cx", "u"])


cx_gate = CXGate()        # Instanz erzeugen
cx_matrix = cx_gate.to_matrix()  # ruft die Matrix auf
cx = np.reshape(cx_matrix,(2,2,2,2))


result = quantum_simulator.apply_cx_on_state(psi, cx, 2, 0)
print(np.reshape(result,(2**n)))
