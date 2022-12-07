import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import plot_utils

def H0(pair_A, pair_B):
    k,l = pair_A
    i,j = pair_B

    print((k==i)*(l==j) + (k==j)*(l==i))
    return 2*((i-1)+(j-1))*( (k==i)*(l==j) + (k==j)*(l==i) )


def V(pair_A, pair_B, g):
    k, l = pair_A
    i, j = pair_B

    x = (k==j) + (k==i) + (l==j) + (l==i)
    print(pair_A, pair_B, x)

    return -0.5*g*( (k==j) + (k==i) + (l==j) + (l==i) )


def eigvals(pairs, g):
    d = len(pairs)
    H = np.zeros((d,d))

    for i in range(d):
        for j in range(d):

            H[i,j] = H0(pairs[i], pairs[j])
            H[i,j] += V(pairs[i], pairs[j], g=g)

    
    exit()
    vals, vecs = np.linalg.eigh(H)
    
    return vals

def plot_eigvals(gs):
    p = 4
    d = int(comb(p,2))
    Es = np.zeros(shape=(*gs.shape, d))

    pairs = []
    for i in range(p):
        for j in range(i+1,p):
            pairs.append((i+1,j+1))

    for i, g in enumerate(gs):
        vals = eigvals(pairs, g=g)
        Es[i] = vals

    fig, ax = plt.subplots()

    for i in range(d):
        ax.plot(gs, Es[:,i], label=f"$E_{i}$")

    ax.legend(ncol=3, loc="upper center")
    ax.set(xlabel="g", ylabel="E(g)")
    plt.show()

if __name__ == "__main__":
    g = np.linspace(-1, 1, 10)
    plot_eigvals(g)