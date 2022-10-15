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
    def __init__(self, V, D=5):
        self.D = D
        self.V = V
        self.H = np.zeros(shape=(D,D))

    def idx_map(self, n,m):
        i = (m+1)%2
        a = m+1
        return i, a

    def fill_phi0_ph0(self):
        elm = 0
        Z = self.V.Z
        
        for i in range(Z):
            # <i|h01|i>
            n = i//2 + 1
            elm += -Z**2 / (2*n**2) 

            for j in range(Z):
                # 0.5*(<ij|v|ij>_AS)
                elm += 0.5*self.V.getAS(i,j,i,j)

        return elm

    def fill_phi0_phi_ia(self, i, a):
        elm = 0
        Z = self.V.Z
        for j in range(Z):
            elm += (self.V.getAS(i,j,a,j))
        return elm

    def fill_phi_ia_phi_ib(self, i, a, j, b):
        V = self.V
        Z = V.Z

        T1 = V.getAS(a,j,i,b)

        T2 = 0
        for k in range(Z):
            T2 += (V.getAS(j,k,b,k)*(i==j) - V.getAS(j,k,i,k)*(a==b))

        n_ij = V.n(i)
        n_ab = V.n(a)
        T3 = (i==j)*(a==b)*(-Z**2)*(1/(2*n_ab**2) - 1/(2*n_ij**2)) 
        T3 = 0


        T4 = 0
        for k in range(Z):
            for l in range(Z):
                T4 += V.getAS(k,l,k,l)
        T4 *= 0.5
        
        if (i==j) and (a==b):
            for k in range(Z):
                n = k//2 + 1
                T4 += -Z**2 / (2*n**2)

     
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

    def __str__(self):
        to_return = ""
        for i in range(self.D):
            for j in range(self.D):
                to_return += f"{self.H[i,j]:6.3f} "

            to_return += "\n"

        return to_return