from qiskit import QuantumCircuit, transpile
#from qiskit.circuit.library import QFTGate, UGate, CXGate
#from qiskit_aer import AerSimulator
from sheet_1 import quantum_simulator
from qiskit.quantum_info import Statevector
import numpy as np
from qiskit.circuit.random import random_circuit
import pytest

#from sheet_1 import apply_cx_on_state
#from sheet_1 import apply_u_on_state





'''def simulate(qc, parameters):
    simulator = AerSimulator()
    job = simulator.run(qc, shots=parameters["shots"],seed_simulator=parameters["seed"])
    result = job.result()
    statevector = result.get_statevector()
    return statevector



def simulate_aer_statevector(qc):
    simulator = AerSimulator(method="statevector")
    job = simulator.run(qc)
    result = job.result()
    return result.get_statevector()'''

def construct_simple_circuit():
    number_of_qubits=3
    qc = QuantumCircuit(number_of_qubits)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    return qc

def construct_random_circuit():
    return random_circuit(num_qubits=4,depth=10,measure=False)

@pytest.mark.parametrize("construction_function", [construct_simple_circuit, construct_random_circuit])
def test_comparision(construction_function):
    qc = construction_function()
    qc=transpile(qc,basis_gates=['u','cx'])

    #calculate statevector with aer
    aer_state = Statevector.from_instruction(qc)

    #calculate statevector with own 
    own_state = quantum_simulator.simulate(qc, None) #eigener

    #compare states
    idx = np.argmax(np.abs(aer_state))
    global_phase = aer_state.data[idx] / own_state[idx]
    assert np.allclose(aer_state, own_state*global_phase)

    '''result_reshaped = np.reshape(result, (2 ** number_of_qubits,))




    global_phase=1

    #assert np.allclose(aer_state, result_reshaped*global_phase, atol=1e-12)



    #qc = QuantumCircuit(4)
    #qc.x(3) # |1000> state

    #qc.h(0) # 1/4*(|0> + |1>)^4 state
    #qc.h(1)
    #qc.h(2)
    #qc.h(3)

    #qc.append(QFTGate(4), [0, 1, 2, 3])
    qc.measure_all()
    transpiled_qc = transpile(qc, basis_gates=["cx", "u"])


    parameters= {
        'shots':1000,
        'seed': 2
    }

    #parameters_2= {
    #   'shots':1000,
    #  'seed': 1
    #}


    mock_result = simulate(transpiled_qc, parameters)
    test_result = quantum_simulator.simulate(transpiled_qc, parameters)

    assert mock_result == test_result'''
