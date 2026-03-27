from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.circuit.random import random_circuit
from sheet_1 import simulator_numba
from sheet_1 import quantum_simulator
from sheet_1 import simulator_cuda
from qiskit_aer import AerSimulator

import numpy as np
import pytest


# ----------------------------
# Circuit construction helpers
# ----------------------------

def construct_simple_circuit():
    number_of_qubits = 4
    qc = QuantumCircuit(number_of_qubits)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    return qc


def construct_random_circuit():
    # fixed seed for reproducibility
    return random_circuit(num_qubits=24, depth=20, measure=False, seed=42)


# ----------------------------
# Shared parametrization
# ----------------------------

@pytest.fixture(params=[construct_random_circuit])
def circuit(request):
    qc = request.param()
    qc = transpile(qc, basis_gates=['u', 'cx'])
    return qc


# ----------------------------
# Benchmarks
# ----------------------------

# sollte gleich bleiben, oder?

def test_aer_simulation(benchmark, circuit):
    """Benchmark Qiskit Aer statevector simulation"""
    circuit.save_statevector()
    simulator = AerSimulator(method="statevector", fusion_enable=False, max_parallel_threads=1)
    compiled = transpile(circuit, simulator)

    def run():
        result = simulator.run(compiled).result()
        return result.get_statevector()

    result = benchmark(run)

    # basic sanity check (not strict comparison here)
    assert result is not None



def test_numba_simulation(benchmark, circuit):
    """Benchmark your custom simulator"""
    def run():
        return simulator_numba.simulate_numba(circuit, None)

    result = benchmark(run)

    assert result is not None


def test_einsum_simulation(benchmark, circuit):
    """Benchmark your custom simulator"""
    def run():
        return quantum_simulator.simulate(circuit, None)

    result = benchmark(run)

    assert result is not None

def test_cuda_simulation(benchmark, circuit):
    """Benchmark your custom simulator"""
    def run():
        return simulator_cuda.simulate_cuda(circuit, None)

    result = benchmark(run)

    assert result is not None


# ----------------------------
# Correctness test (no benchmark)
# ----------------------------

@pytest.mark.parametrize("construction_function", [construct_simple_circuit, construct_random_circuit])
def test_correctness(construction_function):
    qc = construction_function()
    qc = transpile(qc, basis_gates=['u', 'cx'])

    aer_state = Statevector.from_instruction(qc)
    #numba_state = simulator_numba.simulate_numba(qc, None)
    cuda_state = simulator_cuda.simulate_cuda(qc, None)
    # compare states up to global phase
    idx = np.argmax(np.abs(aer_state))
    global_phase = aer_state.data[idx] / cuda_state[idx]

    assert np.allclose(aer_state, cuda_state * global_phase)