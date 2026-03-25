from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit_aer import AerSimulator
from sheet_1 import quantum_simulator



def simulate(qc, parameters):
    simulator = AerSimulator()
    job = simulator.run(qc, shots=parameters["shots"],seed_simulator=parameters["seed"])
    result = job.result()
    counts = result.get_counts()
    return counts





'''qc = QuantumCircuit(4)
qc.x(3) # |1000> state

#qc.h(0) # 1/4*(|0> + |1>)^4 state
#qc.h(1)
#qc.h(2)
#qc.h(3)

qc.append(QFTGate(4), [0, 1, 2, 3])
qc.measure_all()
transpiled_qc = transpile(qc, basis_gates=["cx", "u"])


parameters= {
    'shots':1000,
    'seed': 2
}

parameters_2= {
    'shots':1000,
    'seed': 1
}


mock_result = simulate(transpiled_qc, parameters)
test_result = quantum_simulator.simulate(transpiled_qc, parameters)

assert mock_result == test_result'''
