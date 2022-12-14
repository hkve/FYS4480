from Utils import MatrixElements, ConfigurationInteraction, Indices, HartreeFock
import numpy as np

M = MatrixElements(N=3, Z=2).read()

a_order = [(1,True),(1,False),(2,True),(2,False)]
i_order = [(0,True),(0,False),(0,True),(0,False)]
Fermi = Indices(0)

print("Configuration interaction")
Helium = ConfigurationInteraction(M, Fermi, i_order, a_order).fill()
print(Helium)
Helium.solve()
print(f"CI: E = {Helium.E.min()}\n")

print("Hartree-Fock 1 iter")
HaFo = HartreeFock(M)
HaFo.solve(tol=1e-12, maxiters=1)
print(HaFo)
print(f"HF, {HaFo.iters_conv} iter: E = {HaFo.E_gs}\n")
print(f"Diff after {HaFo.iters_conv} iter = {HaFo.diff_conv:.4e}")
print(f"Sp energies after {HaFo.iters_conv}: {HaFo.sp_conv}\n")

print("Hartree-Fock converged")
HaFo = HartreeFock(M)
HaFo.solve(tol=1e-12, maxiters=10000)
print(HaFo)
print(f"HF, converged: E = {HaFo.E_gs}\n")
print(f"Converged after {HaFo.iters_conv} iters, with diff = {HaFo.diff_conv:.4e}")