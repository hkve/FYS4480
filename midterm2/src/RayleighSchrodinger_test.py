import numpy as np
import matplotlib.pyplot as plt
import plot_utils

def V(p, q, r, s, g=0):
    spin_delta_1 = (p[1]==r[1])*(q[1]==s[1])
    spin_delta_2 = (p[1]!=q[1])*(r[1]!=s[1])
    energy_delta = (p[0] == q[0])*(r[0]==s[0])

    print(spin_delta_1, spin_delta_2, energy_delta)

    return spin_delta_1*spin_delta_2*energy_delta*(-0.5*g)

def order2(hrange=(0,2), prange=(2,4), g=0):
    s = 0

    for a in range(*prange):
        for i in range(*hrange):
            s += 1 / (i - a)
    
    return s*g**2 / 16


def order3(dE2, hrange=(0,2), prange=(2,4), g=0):
    T_2 = 0
    for i in range(*hrange):
        for a in range(*prange):
            T_2 += -(0.5*g)**3 / (2*i - 2*a)**2

    return T_2

def RS_pert(g, ord3=False):
    E = np.zeros_like(g)

    dE0 = 2
    for i in range(len(g)):
        dE1 = -g[i]
        dE2 = order2(g = g[i])
        dE3 = order3(dE2, g=g[i])
        E[i] = dE0 + dE1 + dE2

        if ord3: E[i] += dE3

    return E

def RS_pert_compare(g, E_combs, labels=["FCI"], plot=True, filename=None):
    E_RS = RS_pert(g)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(g, E_RS, label=r"$E_{RS}$")
        for E, label in zip(E_combs, labels):
            ax.plot(g, E, label=label)

        ax.legend()
        plot_utils.save(filename)
        plt.show()

    return E_RS

if __name__ == "__main__":
    g = np.linspace(-1,1,10)
    E = RS_pert(g)
    print(E[0])