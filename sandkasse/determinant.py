import numpy as np
import itertools
import scipy.linalg

n = 3
A = np.random.uniform(low=-1, high=1, size=(n,n))

def determinant(A):
	n, m = A.shape
	cols = np.arange(m)
	det = 0

	# Loop over permutations
	# For all permutation, get sign(?), and multiply over i A[i][permuted index]
	for perm in itertools.permutations(cols):
		row = 1
		diffs = 0

		for i, j in enumerate(perm):
			row *= A[i,j]
			diffs += (i != j)

		sgn = (-1)**np.abs(diffs-1)
		if diffs == 0: sgn *= -1

		det += sgn * row

	return det 

def determinant_lu(A):
	n, m = A.shape
	P, L, U = scipy.linalg.lu(A)
	det = 1
	for i in range(n):
		det *= L[i,i]*U[i,i]

	return det*np.linalg.det(P)

a = np.linalg.det(A)
b = determinant(A)
c = determinant_lu(A)

print(a, b, c)