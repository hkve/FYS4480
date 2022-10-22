from Utils import MatrixElements, Atom, Indices, HartreeFock
import numpy as np

M = MatrixElements(N=3, Z=2).read()

# a_order = [(1,True),(1,False),(2,True),(2,False)]
# i_order = [(0,True),(0,False),(0,True),(0,False)]
Fermi = Indices(0)

# Helium = Atom(M, Fermi, i_order, a_order).fill()
# print(Helium)
# Helium.solve()
# print(Helium.E)


HaFo = HartreeFock(M)
HaFo.solve(tol=1e-16, maxiters=10000)
print(HaFo)
print(HaFo.E_gs)

# hf0 = np.zeros(shape=(6,6))
# All = [_ for _ in Fermi.below()] + [_ for _ in Fermi.above()]

# # Expected after first iteration
# for i, I in enumerate(All):
#     for j, J in enumerate(All):
#         hf0[i,j] = M.get_onebody(I)*(i == j)
#         for beta in Fermi.below():
#             hf0[i,j] += M.getAS(I, beta, J, beta)

# print(hf0)