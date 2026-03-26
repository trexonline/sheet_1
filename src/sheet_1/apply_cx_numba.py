import numba
import numpy as np

@numba.njit(cache=True)
def apply_cx_numba(state: np.ndarray, control: int, target: int) -> np.ndarray:
    if control == target:
        return state.copy()

    result = state.copy()
    number_of_qubits = int(np.log2(state.size))

    lower_qubit = control if control < target else target
    higher_qubit = target if control < target else control

    lower_stride = 1 << lower_qubit
    higher_stride = 1 << higher_qubit

    for idx_lower in range(0, lower_stride):
        for idx_middle in range(0, higher_stride, 1 << (lower_qubit + 1)):
            for idx_higher in range(0, 1 << number_of_qubits, 1 << (higher_qubit + 1)):
                base = idx_higher + idx_middle + idx_lower
                idx_01 = base + lower_stride
                idx_10 = base + higher_stride
                idx_11 = idx_10 + lower_stride

                if control > target:
                    tmp = result[idx_10]
                    result[idx_10] = result[idx_11]
                    result[idx_11] = tmp
                else:
                    tmp = result[idx_01]
                    result[idx_01] = result[idx_11]
                    result[idx_11] = tmp

    return result
