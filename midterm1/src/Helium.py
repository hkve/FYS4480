from Utils import MatrixElements, Atom
import numpy as np

V = MatrixElements(N=3, Z=2).read()

Helium = Atom(V, 5).fill()
print(Helium)
