import enum
from socket import MsgFlag
import numpy as np
import matplotlib.pyplot as plt
import re
from sympy import sympify

class Indices:
    """
    Simple class used to set the Fermi level. Then one can index below or above this level
    """
    def __init__(self, n, roof=3):
        self.nF = n
        self.roof = roof

    def below(self):
        i, s = 0, True
        while i <= self.nF:
            yield i, s
            
            if s:
                s = False
            else:
                s = True
                i+=1 

    def above(self):
        i, s = self.nF +1, True
        while i < self.roof:
            yield i, s
            
            if s:
                s = False
            else:
                s = True
                i+=1 
            

class MatrixElements:
    """
    Contains one and two body interactions. Contains functions to read the text file,
    and fetch matrix elements based on n and ms
    """
    def __init__(self, N=3, Z=2):
        self.elms = np.zeros(shape=(N,N,N,N))
        self.Z = Z # Nucleus charge
        self.N = N # Number of n states


    # Read and store matrix elements
    def read(self, filename="matrix_elements.txt"): 
        with open(filename, "r") as file:
            for line in file:
                line = line.strip()
                bra, = re.findall("(?<=\<)(\d\d)(?=\|)", line)
                ket, = re.findall("(?<=\|)(\d\d)(?=\>)", line)
                [i,j] = [int(c)-1 for c in bra]
                [k,l] = [int(c)-1 for c in ket]
                
                elm = line.split("=")[-1].strip()
                elm = re.sub("Sqrt\[", "sqrt(", elm)
                elm = re.sub("\]", ")", elm)
                elm = sympify(elm)
                
                
                self.elms[i,j,k,l] = float(elm.subs("Z", self.Z))
        
        return self

    # Get the energy corresponding to a one body matrix element.
    # Only need one index due to basis funcs being eig funcs of h0 
    def get_1body(self, i):
        ni, si = i
        return -self.Z**2 / (2*(ni+1)**2)

    # Get 2 body matrix elements 
    def get_2body(self, i,j,k,l):
        ni, si = i
        nj, sj = j
        nk, sk = k
        nl, sl = l
        
        spin_delta = (si == sk)*(sj == sl) 
        
        return self.elms[ni, nj, nk, nl]*spin_delta
        
    # Asymmetric wrapper for matrix elements 
    def getAS(self, i, j, k, l):
        return (self.get_2body(i,j,k,l)-self.get_2body(i,j,l,k))

class ConfigurationInteraction:
    """
    Preform configuration interaction calculations including gs and 1p1h excitations.
    Used the MatrixElement and Indicies classes.
    """
    def __init__(self, M, Fermi, i_order, a_order, D=5):
        self.M = M
        self.Fermi = Fermi
        self.i_order = i_order # How holes are ordered in matrix
        self.a_order = a_order # How particles are ordered in matrix

        self.D = D # Dimmensionality of constructed hamiltonian
        self.H = np.zeros(shape=(D,D))

    # E[Phi_0], expectation value of gs
    def fill_phi0_ph0(self):
        elm = 0
        M = self.M
        Z = M.Z
        
        for i in self.Fermi.below():
            # <i|h01|i>
            elm += M.get_1body(i) 

            for j in self.Fermi.below():
                # 0.5*(<ij|M|ij>_AS)
                elm += 0.5*self.M.getAS(i,j,i,j)

        return elm


    # Overlap between gs and 1h1h
    def fill_phi0_phi_ia(self, i, a):
        elm = 0
        M = self.M
        Z = M.Z

        for j in self.Fermi.below():
            # <ij|v|aj>AS
            elm += (M.getAS(i,j,a,j))
        return elm
    

    # Overlap between 2 different 1p1h's
    def fill_phi_ia_phi_ib(self, i, a, j, b):
        M = self.M
        Z = M.Z

        T1 = M.getAS(a,j,i,b) # <aj|v|ib>AS

        T2 = 0
        for k in self.Fermi.below():
            # <ak|v|bk>AS - <jk|v|ik>AS including delta functions
            T2 += M.getAS(a,k,b,k)*(i==j) - M.getAS(j,k,i,k)*(a==b)

        # One body hole/particle contribution
        T3 = (i==j)*(a==b)*(M.get_1body(a) - M.get_1body(i)) 

        T4 = 0
        if (i==j) and (a==b):
            for k in self.Fermi.below():
                T4 += M.get_1body(k)
                
                for l in self.Fermi.below():
                    T4 += 0.5*M.getAS(k,l,k,l)

        return T1 + T2 + T3 + T4


    # Fill the 5x5 Hamiltonian matrix
    def fill(self):
        H, D_sub = self.H, self.D-1 

        H[0,0] = self.fill_phi0_ph0()
        
        for n in range(D_sub):
            i = self.i_order[n]
            a = self.a_order[n]
            H[n+1,0] = H[0,n+1] = self.fill_phi0_phi_ia(i,a)
        
            for m in range(n, D_sub):
                j = self.i_order[m]
                b = self.a_order[m]
                H[n+1,m+1] = H[m+1,n+1] = self.fill_phi_ia_phi_ib(j,b,i,a)

        return self


    # Find eigen vectors and values and store them
    def solve(self):
        vals, vecs = np.linalg.eig(self.H)
        ids = np.argsort(vals)
        self.E = vals[ids]
        self.vecs = vecs[:, ids]


    # For pretty print
    def __str__(self):
        to_return = ""
        for i in range(self.D):
            for j in range(self.D):
                to_return += f"{self.H[i,j]:6.3f} "

            to_return += "\n"

        return to_return


class HartreeFock:
    """
    Class to perform Hartree-Fock calculations. Matrix elements, number
    of particles and number of single particle states must be supplied.
    """
    def __init__(self, M, Ns=6, Np=2):
        self.M = M

        Fermi = Indices(0)
        # Make list of all states...
        self.States = [_ for _ in Fermi.below()] + [_ for _ in Fermi.above()]
        
        self.Ns = Ns
        self.Np = Np


    # Calculate the density matrix based on coef matrix C
    def density_matrix_(self, C):
        rho = np.zeros_like(C)

        for b in range(self.Ns):
            for d in range(self.Ns):
                s = 0
                for q in range(self.Np):
                    s += C[q,b] * C[q,d]
                rho[b, d] = s

        return rho

    # Perform the actual HF algo
    def solve(self, tol=1e-8, maxiters=1000):
        M, States = self.M, self.States
        Ns, Np = self.Ns, self.Np

        diff = 1
        C = np.eye(Ns, Ns)
        rho = self.density_matrix_(C)
        
        iters = 0

        sp_E_old = np.zeros(Ns)    
        sp_E_new = np.zeros(Ns)
        while iters < maxiters and diff > tol:
            HFmat = np.zeros_like(rho)

            # SP mat elms. Works since the new basis is also ortonormal
            for l, lmbda in enumerate(States):
                HFmat[l,l] = M.get_1body(lmbda)

            # Construct HFmat of l,g sum
            for l, lmbda in enumerate(States):
                for g, gamma in enumerate(States):
                    
                    # Calculate l,g matrix elm of HFmat
                    elmSum = 0
                    for b, beta in enumerate(States):
                        for d, delta in enumerate(States):
                            elmSum += rho[b,d]*M.getAS(lmbda, beta, gamma, delta)

                    HFmat[l,g] += elmSum

            # Get new single particle energies and coefs
            sp_E_new, C = np.linalg.eigh(HFmat)

            # Calculate new density matrix
            rho = self.density_matrix_(C.T)

            # Check sp energy differences from last iter to see if converged
            diff = np.sum(np.abs(sp_E_new-sp_E_old))/Ns

            sp_E_old = sp_E_new
            iters += 1

        self.HFmat_conv = HFmat
        self.iters_conv = iters
        self.diff_conv = diff
        self.sp_conv = sp_E_new

        # Calculate gs expectation value
        self.evaluate_Energy_(rho)

    # Based on the converged density matrix, calculate the gs energy
    def evaluate_Energy_(self, rho):
        M, States = self.M, self.States

        E_gs_1body = 0
        for a, alpha in enumerate(States):
            E_gs_1body += rho[a,a] * M.get_1body(alpha)

        E_gs_2body = 0
        for a, alpha in enumerate(States):
            for b, beta in enumerate(States):
                for g, gamma in enumerate(States):
                    for d, delta in enumerate(States):
                        E_gs_2body += rho[a,g]*rho[b,d]*M.getAS(alpha, beta, gamma, delta)

        self.E_gs = E_gs_1body + 0.5*E_gs_2body


    def __str__(self):
        to_return = ""
        for i in range(self.Ns):
            for j in range(self.Ns):
                to_return += f"{self.HFmat_conv[i,j]:6.3f} "

            to_return += "\n"

        return to_return