import time
import numpy as np
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator
from sheet_1 import quantum_simulator, simulator_numba, simulator_cuda


def construct_random_circuit_for_benchmark():
    # Same setup as tests/test_benchmark_cuda.py
    qc = random_circuit(num_qubits=24, depth=20, measure=False, seed=42)
    return transpile(qc, basis_gates=['u', 'cx'])


def run_aer(circuit, simulator):
    qc_for_aer = circuit.copy()
    qc_for_aer.save_statevector()
    compiled = transpile(qc_for_aer, simulator)
    result = simulator.run(compiled).result()
    return result.get_statevector()


def benchmark_runtime(func, repeats=3):
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        timings.append(t1 - t0)
    timings = np.asarray(timings, dtype=float)
    return float(np.mean(timings)), float(np.std(timings))


qc = construct_random_circuit_for_benchmark()
aer = AerSimulator(method='statevector', fusion_enable=False, max_parallel_threads=1)

# Warm-up for JIT/compile paths.
_ = simulator_numba.simulate_numba(qc, None)
_ = simulator_cuda.simulate_cuda(qc, None)

aer_mean, aer_std = benchmark_runtime(lambda: run_aer(qc, aer), repeats=3)
numba_mean, numba_std = benchmark_runtime(lambda: simulator_numba.simulate_numba(qc, None), repeats=3)
einsum_mean, einsum_std = benchmark_runtime(lambda: quantum_simulator.simulate(qc, None), repeats=3)
cuda_mean, cuda_std = benchmark_runtime(lambda: simulator_cuda.simulate_cuda(qc, None), repeats=3)

labels = ['Aer', 'Numba', 'Einsum', 'CUDA']
means = [aer_mean, numba_mean, einsum_mean, cuda_mean]
stds = [aer_std, numba_std, einsum_std, cuda_std]

x = np.arange(len(labels))

plt.figure(figsize=(8, 5))
plt.bar(x, means, yerr=stds, capsize=5)
plt.xticks(x, labels)
plt.ylabel('Runtime (seconds)')
plt.title('Runtime Comparison on Random Circuit (24 qubits, depth 20)')
plt.grid(axis='y', linestyle='--', alpha=0.35)
plt.tight_layout()
plt.savefig('runtime_comparison.png')
plt.show()

print('Average runtime over 3 runs (seconds):')
for label, mean, std in zip(labels, means, stds):
    print(f"{label:>6} | mean: {mean:.6f} | std: {std:.6f}")