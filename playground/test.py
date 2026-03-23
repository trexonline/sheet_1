from sheet_1 import hello
from sheet_1.pauli_maticies import pauli_x, pauli_y, pauli_z

print(hello())

print(f"{pauli_x() @ pauli_y()} = {1j * pauli_z()}")
