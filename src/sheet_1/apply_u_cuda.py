import numpy as np
from numba import cuda


@cuda.jit
def _apply_u_kernel(state, u, acting_on, number_of_qubits, size):
    pair_id = cuda.grid(1)
    half_size = size >> 1
    if pair_id >= half_size:
        return

    idx0 = 0
    src = pair_id

    # Reconstruct basis index from pair_id while fixing acting_on bit to 0.
    for bit_pos in range(number_of_qubits):
        if bit_pos == acting_on:
            continue
        bit_val = src & 1
        src >>= 1
        idx0 |= bit_val << bit_pos

    idx1 = idx0 + (1 << acting_on)

    s0 = state[idx0]
    s1 = state[idx1]

    state[idx0] = u[0, 0] * s0 + u[0, 1] * s1
    state[idx1] = u[1, 0] * s0 + u[1, 1] * s1


def apply_u_cuda(state: np.ndarray, u: np.ndarray, acting_on: int, threads_per_block: int = 256) -> np.ndarray:
    """Apply a single-qubit U gate to a statevector on the GPU using Numba CUDA.

    Args:
        state: Statevector with shape (2**n,) and complex dtype.
        u: 2x2 unitary matrix.
        acting_on: Qubit index in little-endian convention.
        threads_per_block: CUDA threads per block.

    Returns:
        Updated statevector copied back to host memory.
    """
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available. Install CUDA toolkit/driver and use a CUDA-enabled GPU.")

    if state.ndim != 1:
        raise ValueError("state must be a 1D array with shape (2**n,)")
    if u.shape != (2, 2):
        raise ValueError("u must have shape (2, 2)")

    size = state.size
    if size == 0 or (size & (size - 1)) != 0:
        raise ValueError("state length must be a non-zero power of two")

    number_of_qubits = int(np.log2(size))
    if acting_on < 0 or acting_on >= number_of_qubits:
        raise ValueError("acting_on is out of range for the given statevector")

    host_state = np.asarray(state, dtype=np.complex128).copy()
    host_u = np.asarray(u, dtype=np.complex128)

    d_state = cuda.to_device(host_state)
    d_u = cuda.to_device(host_u)

    half_size = size >> 1
    blocks_per_grid = (half_size + threads_per_block - 1) // threads_per_block
    _apply_u_kernel[blocks_per_grid, threads_per_block](d_state, d_u, acting_on, number_of_qubits, size)
    cuda.synchronize()

    return d_state.copy_to_host()


__all__ = ["apply_u_cuda"]
