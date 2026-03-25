import numpy as np

def apply_cx_on_state(state: np.ndarray, cx: np.ndarray, acting_on1: int, acting_on2: int) -> None:
    '''Applies a CNOT gate to the given state vector. The CNOT gate is defined 
    by the 4x4 matrix cx, and it acts on the qubits specified by acting_on1 (control) 
    and acting_on2 (target). The state vector is modified in-place.'''
    number_of_qubits=state.ndim
    acting_on1=number_of_qubits-1-acting_on1
    acting_on2=number_of_qubits-1-acting_on2
    cx_matrix=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    cx = np.reshape(cx_matrix,(2,2,2,2))
    old_indices = [i for i in range(number_of_qubits)]
    new_indices = old_indices.copy()
    new_indices[acting_on1] = 51
    new_indices[acting_on2] = 50
    result=np.einsum(cx,[51,50,acting_on1,acting_on2],state,old_indices,new_indices)
    return result