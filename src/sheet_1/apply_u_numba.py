import numba 
import numpy as np
from qiskit.circuit.library import UGate



@numba.jit(cache=True)
def apply_u_numba(state: np.ndarray, u: np.ndarray, acting_on: int) -> np.ndarray:
    number_of_qubits=state.ndim
    #acting_on=number_of_qubits-acting_on-1

    for idx_lower in range(0, 2**acting_on):
        for idx_upper in range(0, 2*number_of_qubits, 2**acting_on):
            idx0 = idx_lower + idx_upper
            idx1 = idx_upper + idx_lower + 2**acting_on
            s0 = state[idx0]
            s1 = state[idx1]
            state[idx0] = u[0, 0] * s0 + u[0, 1] * s1
            state[idx1] = u[1, 0] * s0 + u[1, 1] * s1
    return state







state = np.array([1, 0, 0, 0], dtype=np.complex128)
u=UGate(1, 0, 0).to_matrix()
result = apply_u_numba(state, u , 1)

print(result)