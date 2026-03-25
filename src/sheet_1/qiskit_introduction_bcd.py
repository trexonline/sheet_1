from qiskit import QuantumCircuit, transpile
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFTGate
from qiskit_aer import AerSimulator



qc = QuantumCircuit(4)
#qc.x(3) # |1000> state

qc.h(0) # 1/4*(|0> + |1>)^4 state
qc.h(1)
qc.h(2)
qc.h(3)

qc.append(QFTGate(4), [0, 1, 2, 3])
qc.measure_all()
transpiled_qc = transpile(qc, basis_gates=["cx", "u"])

simulator = AerSimulator()
job = simulator.run(transpiled_qc, shots=1024)
result = job.result()
counts = result.get_counts()

print(counts)

transpiled_qc.draw("mpl")

plot_histogram(counts)

plt.show()