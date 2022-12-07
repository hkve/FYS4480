import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import plot_utils

def H0(pair):
    p1, p2 = pair
    return 2*((p1-1) + (p2-1))


def V(pair_A, pair_B, g):
    k, l = pair_A
    i, j = pair_B

    return -0.5*g*( (k==j) + (k==i) + (l==j) + (l==i) )


def eigvals(pairs, g):
    d = len(pairs)
    H = np.zeros((d,d))

    for i in range(d):
        for j in range(d):
            if i == j:
                H[i,j] += H0(pairs[i])

            H[i,j] += V(pairs[i], pairs[j], g=g)


    vals, vecs = np.linalg.eigh(H)
    
    return vals

def CI(gs, skip_4p4h=False):
    p = 4
    d = int(comb(p,2))

    pairs = []
    for i in range(p):
        for j in range(i+1,p):
            pairs.append((i+1,j+1))

    pairs[2], pairs[3] = pairs[3], pairs[2]
    if skip_4p4h:
        pairs.pop()
        d -= 1

    Es = np.zeros(shape=(*gs.shape, d))
    for i, g in enumerate(gs):
        vals = eigvals(pairs, g)
        Es[i] = vals

    return Es

def plot_eigvals(gs, plot=True, filename=None, skip_4p4h=False):
    Es = CI(gs, skip_4p4h=skip_4p4h)
    d = Es.shape[1]

    if plot:
        fig, ax = plt.subplots()

        for i, E in enumerate(Es.T):
            ax.plot(gs, E, label=f"$E_{i}$")

        ax.legend(ncol=3, loc="upper center")
        ylim_old = ax.get_ylim()
        ylim = (ylim_old[0], 1.15*ylim_old[1])

        ax.set(xlabel="g", ylabel="E(g)", ylim=ylim)
        
        plot_utils.save(filename)
        plt.show()


    return Es

def plot_diff(g, E1, E2, ylabel=r"$|E_{FCI} - E_{CID}|$", filename=None):
    fig, ax = plt.subplots()

    diff = np.abs(E1-E2)

    ax.plot(g, diff)
    ax.set(xlabel="g", ylabel=ylabel)

    plot_utils.save(filename)
    plt.show()

if __name__ == "__main__":
    g = np.linspace(-1, 1, 10)
    E_FCI = plot_eigvals(g, plot=True, filename="FCI", skip_4p4h=False)
    E_CID = plot_eigvals(g, plot=True, filename="CID", skip_4p4h=True)

    Egs_FCI = E_FCI[:,0]
    Egs_CID = E_CID[:,0]

    plot_diff(g, Egs_FCI, Egs_CID, filename="FCI_CID_diff")