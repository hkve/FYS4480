import enum
from socket import MsgFlag
import numpy as np
import matplotlib.pyplot as plt
import re
from sympy import sympify

class Indices:
    def __init__(self, n, roof=3):
        self.nF = n
        self.roof = roof

    def below(self):
        i, up = 0, True
        while i <= self.nF:
            yield i, up
            
            if up:
                up = False
            else:
                up = True
                i+=1 

    def above(self):
        i, up = self.nF +1, True
        while i < self.roof:
            yield i, up
            
            if up:
                up = False
            else:
                up = True
                i+=1 
            

class MatrixElements:
    def __init__(self, N=3, Z=2):
        self.elms = np.zeros(shape=(N,N,N,N))
        self.Z = Z
        self.N = N

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

    def get_onebody(self, i):
        ni, si = i
        return -self.Z**2 / (2*(ni+1)**2)

    def get(self, i,j,k,l):
        ni, si = i
        nj, sj = j
        nk, sk = k
        nl, sl = l
        
        spin_delta = (si == sk)*(sj == sl) 
        
        return self.elms[ni, nj, nk, nl]*spin_delta
        
    def getAS(self, i, j, k, l):
        return (self.get(i,j,k,l)-self.get(i,j,l,k))

class Atom:
    def __init__(self, M, Fermi, i_order, a_order, D=5):
        self.M = M
        self.Fermi = Fermi
        self.i_order = i_order
        self.a_order = a_order

        self.D = D
        self.H = np.zeros(shape=(D,D))

    def fill_phi0_ph0(self):
        elm = 0
        M = self.M
        Z = M.Z
        
        for i in self.Fermi.below():
            # <i|h01|i>
            elm += M.get_onebody(i) 

            for j in self.Fermi.below():
                # 0.5*(<ij|M|ij>_AS)
                elm += 0.5*self.M.getAS(i,j,i,j)

        return elm

    def fill_phi0_phi_ia(self, i, a):
        elm = 0
        M = self.M
        Z = M.Z

        for j in self.Fermi.below():
            elm += (M.getAS(i,j,a,j))
        return elm
    
    def fill_phi_ia_phi_ib(self, i, a, j, b):
        M = self.M
        Z = M.Z

        T1 = M.getAS(a,j,i,b)

        T2 = 0
        for k in self.Fermi.below():
            T2 += M.getAS(a,k,b,k)*(i==j) - M.getAS(j,k,i,k)*(a==b)

        T3 = (i==j)*(a==b)*(M.get_onebody(a) - M.get_onebody(i)) 

        T4 = 0
        if (i==j) and (a==b):
            for k in self.Fermi.below():
                T4 += M.get_onebody(k)
                
                for l in self.Fermi.below():
                    T4 += 0.5*M.getAS(k,l,k,l)

        return T1 + T2 + T3 + T4

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

    def solve(self):
        vals, vecs = np.linalg.eig(self.H)
        ids = np.argsort(vals)
        self.E = vals[ids]
        self.vecs = vecs[:, ids]

    def __str__(self):
        to_return = ""
        for i in range(self.D):
            for j in range(self.D):
                to_return += f"{self.H[i,j]:6.3f} "

            to_return += "\n"

        return to_return


class HartreeFock:
    def __init__(self, M, Ns=6, Np=2):
        self.M = M

        Fermi = Indices(0)
        self.States = [_ for _ in Fermi.below()] + [_ for _ in Fermi.above()]
        
        self.Ns = Ns
        self.Np = Np


    def density_matrix_(self, C):
        rho = np.zeros_like(C)

        for b in range(self.Ns):
            for d in range(self.Ns):
                s = 0
                for q in range(self.Np):
                    s += C[b, q] * C[d, q]
                rho[b, d] = s

        return rho


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

            for l, lmbda in enumerate(States):
                for g, gamma in enumerate(States):
                    elmSum = (l==g)*M.get_onebody(lmbda)

                    for b, beta in enumerate(States):
                        for d, delta in enumerate(States):
                            elmSum += rho[b,d]*M.getAS(lmbda, beta, gamma, delta)

                    HFmat[l,g] = elmSum

            sp_E_new, C = np.linalg.eigh(HFmat)
            rho = self.density_matrix_(C)

            diff = np.sum(np.abs(sp_E_new-sp_E_old))/Ns

            sp_E_old = sp_E_new
            iters += 1
            if( iters%100 == 0):
                print(diff)

        self.HFmat_conv = HFmat
        self.evaluateE(C)

    def evaluateE(self, C):
        M, States = self.M, self.States
        Np = self.Np
        self.E_gs = 0

        E_gs_2body = 0
        for p in range(Np):
            for a, alpha in enumerate(States):
                for b, beta in enumerate(States):
                    if(a == b):
                        self.E_gs += C[a,p] * C[b,p] * M.get_onebody(alpha) 
                    
                    for q in range(Np):
                        for g, gamma in enumerate(States):
                            for d, delta in enumerate(States):
                                E_gs_2body += C[a,p]*C[b,q]*C[g,p]*C[d,q]*M.getAS(alpha, beta, gamma, delta)

        self.E_gs += 0.5*E_gs_2body


    def __str__(self):
        to_return = ""
        for i in range(self.Ns):
            for j in range(self.Ns):
                to_return += f"{self.HFmat_conv[i,j]:6.3f} "

            to_return += "\n"

        return to_return