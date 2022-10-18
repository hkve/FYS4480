from Utils import Indicies, MatrixElements, Atom
import numpy as np

M = MatrixElements(N=3, Z=4).read()


# Beryllium = Atom(M, 5).fill()
# print(Beryllium)
# Beryllium.solve()
# print(Beryllium.E)

# n = [0,1,2,3]
# x = -20

# for i in n:
#     for j in n:
#         for k in n:
#             for l in n:
#                 if M.get(i,j,k,l) != 0:
#                     print(M.state(i), M.state(j), M.state(k), M.state(l), M.get(i,j,k,l))
#                 x += 0.5*M.get(i,j,k,l)

i = Indicies(0, True)

for _ in range(4):
    j = Indicies(0, True)
    
    for _ in range(4):
        print(f"{i}\t{j}")
        j = next(j)
    
    i = next(i)