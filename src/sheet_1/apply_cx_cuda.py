import numpy as np
from numba import cuda


@cuda.jit
def _apply_cx_kernel(state, control, target, number_of_qubits, size):
    swap_id = cuda.grid(1)
    number_of_swaps = size >> 2
    if swap_id >= number_of_swaps:
        return

    idx0 = 0
    src = swap_id

    # Reconstruct one basis index where control=1 and target=0.
    for bit_pos in range(number_of_qubits):
        if bit_pos == control:
            bit_val = 1
        elif bit_pos == target:
            bit_val = 0
        else:
            bit_val = src & 1
            src >>= 1
        idx0 |= bit_val << bit_pos

    idx1 = idx0 + (1 << target)

    tmp = state[idx0]
    state[idx0] = state[idx1]
    state[idx1] = tmp


def apply_cx_cuda(state: np.ndarray, control: int, target: int, threads_per_block: int = 256) -> np.ndarray:
    '''Applies a CNOT gate using Cuda to the given state vector and it acts on the qubits target and control. 
    threads_per_block specifies the number of threads per block for the CUDA kernel.'''
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available. Install CUDA toolkit/driver and use a CUDA-enabled GPU.")

    if state.ndim != 1:
        raise ValueError("state must be a 1D array with shape (2**n,)")

    size = state.size
    if size == 0 or (size & (size - 1)) != 0:
        raise ValueError("state length must be a non-zero power of two")

    number_of_qubits = int(np.log2(size))
    if control < 0 or control >= number_of_qubits:
        raise ValueError("control is out of range for the given statevector")
    if target < 0 or target >= number_of_qubits:
        raise ValueError("target is out of range for the given statevector")
    if control == target:
        return np.asarray(state, dtype=np.complex128).copy()

    host_state = np.asarray(state, dtype=np.complex128).copy()

    d_state = cuda.to_device(host_state)

    number_of_swaps = size >> 2
    blocks_per_grid = (number_of_swaps + threads_per_block - 1) // threads_per_block
    _apply_cx_kernel[blocks_per_grid, threads_per_block](d_state, control, target, number_of_qubits, size)
    cuda.synchronize()

    return d_state.copy_to_host()


__all__ = ["apply_cx_cuda"]
