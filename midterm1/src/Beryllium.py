from Utils import Indices, MatrixElements, Atom
import numpy as np

M = MatrixElements(N=3, Z=4).read()

a_order = [(2,True),(2,False),(2,True),(2,False)]
i_order = [(0,True),(0,False),(1,True),(1,False)]
Fermi = Indices(1)

Beryllium = Atom(M, Fermi, i_order, a_order).fill()
print(Beryllium)
Beryllium.solve()
print(Beryllium.E)
