import numpy as np

def apply_u_on_state(state: np.ndarray, u: np.ndarray, acting_on: int) -> np.ndarray:
    '''Applies a U gate to the given state vector and it acts on the qubit specified by acting_on.
    The state vector is expected to be in the form of a multi-dimensional array where each dimension 
    corresponds to a qubit. For example, for a 3-qubit state, the state vector should be of shape (2, 2, 2).
    u ist the matrix reprensentation of the U gate in form of a np.array.'''
    number_of_qubits=state.ndim
    acting_on=number_of_qubits-acting_on-1
    old_indices = [i for i in range(number_of_qubits)]
    new_indices = old_indices.copy()
    new_indices[acting_on] = 51
    result=np.einsum(u,[51,acting_on],state,old_indices,new_indices)
    return result