from Utils import MatrixElements, Atom, Indices
import numpy as np

M = MatrixElements(N=3, Z=2).read()

a_order = [(1,True),(1,False),(2,True),(2,False)]
i_order = [(0,True),(0,False),(0,True),(0,False)]
Fermi = Indices(0)

Helium = Atom(M, Fermi, i_order, a_order).fill()
print(Helium)
Helium.solve()
print(Helium.E)