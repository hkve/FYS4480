import numpy as np
import matplotlib.pyplot as plt


def diagonalize2(g, d=1):
	lam = np.sqrt(d**2+g**2)
	vp = np.array([d+lam, g])/g
	vm = np.array([d-lam, g])/g

	# Change of basis matrix
	U = np.c_[vp, vm]

	# Hamiltonian in Slater basis
	H = np.array([
		[2*d-g, -g],
		[-g, 4*d-g]
	])

	# Decompose Hamiltonian as H = U * D * U^-1
	# => D = U^-1 * H * U where D is the hamiltonian expressed in its eigenbasis 
	H_diag = np.linalg.inv(U) @ H @ U

	E1, E2 = np.diagonal(H_diag)
	
	lam1 = 3*d - g - lam
	lam2 = 3*d - g + lam
	print(np.abs(E1-lam1), np.abs(E2-lam2))

	return E1, E2

def diagonalize3(g, d=1):
	H = np.array([
		[2*d-g, -g, -g],
		[-g, 4*d-g, -g],
		[-g, -g, 6*d-g]
	])

	vals, vecs = np.linalg.eig(H)
	v1, v2, v3 = vecs[:,0], vecs[:,1], vecs[:,2]

	U = np.c_[v1, v2, v3]
	H_diag = np.linalg.inv(U) @ H @ U
	E1, E2, E3 = np.diagonal(H_diag)

	print(np.abs(np.diagonal(H_diag)-vals))

	return E1, E2, E3

def plot2():
	G = np.linspace(-1, 1, 100)
	E1 = np.zeros_like(G)
	E2 = np.zeros_like(G)

	fig, ax = plt.subplots(nrows=1, ncols=2)

	for i, g in enumerate(G):
		e1, e2 = diagonalize2(g)
		E1[i] = e1
		E2[i] = e2

	delta_E = E2 - E1
	ax[0].plot(G, delta_E, label=r"$\Delta E = E_2 - E_1$")
	ax[1].plot(G, E1, label=r"$E_1$")
	ax[1].plot(G, E2, label=r"$E_2$")
	ax[0].set(xlabel="g", ylabel="Energy")
	ax[1].set(xlabel="g")
	ax[0].legend()
	ax[1].legend()
	plt.show()


def plot3():
	G = np.linspace(0, 2, 100)
	E1 = np.zeros_like(G)
	E2 = np.zeros_like(G)
	E3 = np.zeros_like(G)

	fig, ax = plt.subplots(nrows=1, ncols=2)

	for i, g in enumerate(G):
		e1, e2, e3 = diagonalize3(g)
		E1[i] = e1
		E2[i] = e2
		E3[i] = e3

	delta_E21 = E2 - E1
	delta_E31 = E3 - E1
	delta_E32 = E3 - E2

	ax[0].plot(G, delta_E21, label=r"$\Delta E_{21} = E_2 - E_1$")
	ax[0].plot(G, delta_E31, label=r"$\Delta E_{31} = E_3 - E_1$")
	ax[0].plot(G, delta_E32, label=r"$\Delta E_{32} = E_3 - E_3$")
	
	ax[1].plot(G, E1, label=r"$E_1$")
	ax[1].plot(G, E2, label=r"$E_2$")
	ax[1].plot(G, E3, label=r"$E_3$")
	ax[0].set(xlabel="g", ylabel="Energy")
	ax[1].set(xlabel="g")
	ax[0].legend()
	ax[1].legend()
	plt.show()

if __name__ == "__main__":
	plot2()
	plot3()