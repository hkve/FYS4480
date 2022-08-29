import numpy as np
import itertools

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
		
		det += (-1)**(diffs) * row

	return det 

a = np.linalg.det(A)
b = determinant(A)

print(a, b)