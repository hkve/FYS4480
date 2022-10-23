from Utils import MatrixElements, Atom, Indices, HartreeFock
import numpy as np

M = MatrixElements(N=3, Z=2).read()

a_order = [(1,True),(1,False),(2,True),(2,False)]
i_order = [(0,True),(0,False),(0,True),(0,False)]
Fermi = Indices(0)

print("Configuration interaction")
Helium = Atom(M, Fermi, i_order, a_order).fill()
print(Helium)
Helium.solve()
print(f"CI: E = {Helium.E.min()}\n")

print("Hartree-Fock 1 iter")
HaFo = HartreeFock(M)
HaFo.solve(tol=1e-12, maxiters=2)
print(HaFo)
print(f"HF, 1 iter: E = {HaFo.E_gs}")
print(f"Diff after 1 iter = {HaFo.diff_conv:.4e}")

print("Hartree-Fock converged")
HaFo = HartreeFock(M)
HaFo.solve(tol=1e-12, maxiters=10000)
print(HaFo)
print(f"HF, converged: E = {HaFo.E_gs}")
print(f"Converged after {HaFo.iters_conv} iters, with diff = {HaFo.diff_conv:.4e}")