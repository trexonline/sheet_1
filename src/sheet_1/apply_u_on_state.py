import numpy as np

def apply_u_on_state(state: np.ndarray, u: np.ndarray, acting_on: int) -> np.ndarray:
    """Applies a single-qubit gate to the given state vector. The gate is defined 
    by the 2x2 matrix u, and it acts on the qubit specified by acting_on"""
    number_of_qubits=state.ndim
    acting_on=number_of_qubits-acting_on-1
    old_indices = [i for i in range(number_of_qubits)]
    new_indices = old_indices.copy()
    new_indices[acting_on] = 51
    result=np.einsum(u,[51,acting_on],state,old_indices,new_indices)
    return result