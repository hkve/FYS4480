from Utils import MatrixElements, Atom
import numpy as np

M = MatrixElements(N=3, Z=2).read()

Helium = Atom(M, 5).fill()
print(Helium)
Helium.solve()
print(Helium.E)