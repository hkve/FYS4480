import numpy as np

g = -1

I = [0,1]
A = [2,3]

T1 = 0
for i in I:
    for a in A:
        for c in A:
            T1 += -g**3 / ((i-a)*(i-c))

for i in I:
    for k in I:
        for a in A:
            T1 += -g**3 / ((i-a)*(k-a))

T1 /= 32

T2 = 0
for i in I:
    for a in A:
        T2 += (i-a)**(-2) 
T2 = (g**3 / 16)*T2

print(T1, T2)

T3 = 0
for i in I:
    for a in A:
        T3 += (i-a)**(-2)

T3 = (-g**3 / 16) * T3

print(T3)