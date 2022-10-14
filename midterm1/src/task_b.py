import numpy as np
import matplotlib.pyplot as plt

class MatrixElements:
    def __init__(self, N):
        self.elms = np.zeros(shape=(N,N,N,N))

    def read(self, filename):
        with open(filename, "r") as file:
            for line in file:
                line = line.strip()
                line = line.replace("<", "")
                line = line.replace(">", "")
                line = line.replace("|V|", ",")
                line = line.replace(" = ", ",")
                
                idx_in, idx_out, elm = line.split(",")
                i,j = int(idx_in[0]), int(idx_in[1])
                k,l = int(idx_out[0]), int(idx_out[1])
                
                self.elms[i-1,j-1,k-1,l-1] = 1

    def get(self, i, j, k, l):
        pass

M = MatrixElements(3)

M.read("matrix_elements.txt")