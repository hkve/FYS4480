from Utils import Indices, MatrixElements, Atom, HartreeFock
import numpy as np

M = MatrixElements(N=3, Z=4).read()

a_order = [(2,True),(2,False),(2,True),(2,False)]
i_order = [(0,True),(0,False),(1,True),(1,False)]
Fermi = Indices(1)

# Beryllium = Atom(M, Fermi, i_order, a_order).fill()
# print(Beryllium)
# Beryllium.solve()
# print(Beryllium.E)

HaFo = HartreeFock(M, Fermi)
HaFo.solve(tol=1e-12, maxiters=1)
print(HaFo)


# Z = 4
# v = M.elms
# E = - 5*Z**2 / 4
# E += (
#     v[0,0,0,0] + v[1,1,1,1] + 2*v[0,1,0,1] - v[0,1,1,0] + 2 *v[1,0,1,0] - v[1,0,0,1]
# )

# print(E)