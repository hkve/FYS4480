import numpy as np
import matplotlib.pyplot as plt
import re
from sympy import sympify

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

    def n(self, i):
        return i//2 + 1

    def get_onebody(self, i):
        return -self.Z**2 / (2*self.n(i)**2)

    def get(self, i,j,k,l):
        n_i, n_j, n_k, n_l  = i//2, j//2, k//2, l//2
        s_i, s_j, s_k, s_l  = i%2, j%2, k%2, l%2
        

        if s_i == s_k and s_j == s_l:
            return self.elms[n_i, n_j, n_k, n_l]
        else:
            return 0

    def getAS(self, i, j, k, l):
        return (self.get(i,j,k,l)-self.get(i,j,l,k))

class Atom:
    def __init__(self, M, D=5):
        self.D = D
        self.M = M
        self.H = np.zeros(shape=(D,D))

    def idx_map(self, n,m):
        i = (m+1)%2
        a = m+1
        return i, a

    def fill_phi0_ph0(self):
        elm = 0
        M = self.M
        Z = M.Z
        
        for i in range(Z):
            # <i|h01|i>
            n = i//2 + 1
            elm += M.get_onebody(i) 

            for j in range(Z):
                # 0.5*(<ij|M|ij>_AS)
                elm += 0.5*self.M.getAS(i,j,i,j)

        return elm

    def fill_phi0_phi_ia(self, i, a):
        elm = 0
        M = self.M
        Z = M.Z

        for j in range(Z):
            elm += (M.getAS(i,j,a,j))
        return elm

    def fill_phi_ia_phi_ib(self, i, a, j, b):
        M = self.M
        Z = M.Z

        T1 = M.getAS(a,j,i,b)

        T2 = 0
        for k in range(Z):
            T2 += M.getAS(a,k,b,k)*(i==j) - M.getAS(j,k,i,k)*(a==b)

        T3 = (i==j)*(a==b)*(M.get_onebody(a) - M.get_onebody(i)) 

        T4 = 0
        if (i==j) and (a==b):
            for k in range(Z):
                T4 += M.get_onebody(k)
                
                for l in range(Z):
                    T4 += 0.5*M.getAS(k,l,k,l)

        return T1 + T2 + T3 + T4

    def fill(self):
        H, D = self.H, self.D 

        for n in range(D):
            for m in range(n,D):
                i, a  = self.idx_map(n,m)
                j, b  = self.idx_map(m,n)
                
                if n == m == 0:
                    H[n,m] = self.fill_phi0_ph0()
                elif n == 0:
                    H[n,m] = H[m,n] = self.fill_phi0_phi_ia(i,a)
                else:
                    H[n,m] = H[m,n] = self.fill_phi_ia_phi_ib(j,b,i,a)

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