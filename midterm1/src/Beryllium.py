from Utils import Indices, MatrixElements, Atom, HartreeFock
import numpy as np

M = MatrixElements(N=3, Z=4).read()

a_order = [(2,True),(2,False),(2,True),(2,False)]
i_order = [(0,True),(0,False),(1,True),(1,False)]
Fermi = Indices(1)

Z = 4
v = M.elms
E = - 5*Z**2 / 4
E += (
    v[0,0,0,0] + v[1,1,1,1] + 2*v[0,1,0,1] - v[0,1,1,0] + 2 *v[1,0,1,0] - v[1,0,0,1]
)

print(f"E[Phi0] = {E:.4f}")

print("Configuration interaction")
Beryllium = Atom(M, Fermi, i_order, a_order).fill()
print(Beryllium)
Beryllium.solve()
print(f"CI: E = {Beryllium.E.min()}\n")

print("Hartree-Fock 1 iter")
HaFo = HartreeFock(M, Np=4)
HaFo.solve(tol=1e-12, maxiters=2)
print(HaFo)
print(f"HF, 1 iter: E = {HaFo.E_gs}\n")
print(f"Diff after 1 iter = {HaFo.diff_conv:.4e}")

print("Hartree-Fock converged")
HaFo = HartreeFock(M, Np=4)
HaFo.solve(tol=1e-12, maxiters=10000)
print(HaFo)
print(f"HF, 1 iter: E = {HaFo.E_gs}\n")
print(f"Converged after {HaFo.iters_conv} iters, with diff = {HaFo.diff_conv:.4e}")