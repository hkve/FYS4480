import numpy as np
from scipy.special import comb
import plot_utils
import matplotlib.pyplot as plt

def H0(pair):
    p1, p2 = pair
    return 2*((p1-1) + (p2-1))


def V(pair_A, pair_B, g):
    k, l = pair_A
    i, j = pair_B

    return -0.5*g*((l==j)+(l==i)+(k==j)+(k==i))

def order2(pairs, d, g):
    E = 0
    for i in range(1, d):
        V0i = V(pairs[0], pairs[i], g)
        Vi0 = V(pairs[i], pairs[0], g)
        E0 = H0(pairs[0])
        Ei = H0(pairs[i])
        D0i = E0 - Ei

        E += (V0i*Vi0)/D0i
    
    return E


def order3(E1, pairs, d, g):
    E = 0

    for i in range(1, d):
        for j in range(1, d):
            V0i = V(pairs[0], pairs[i], g)
            Vj0 = V(pairs[j], pairs[0], g)
            Wij = V(pairs[i], pairs[j], g) - (i==j)*E1

            E0 = H0(pairs[0])
            Ei = H0(pairs[i])
            Ej = H0(pairs[j])

            D0i = E0 - Ei
            D0j = E0 - Ej

            dE = V0i*Wij*Vj0 / (D0i*D0j)

            E += dE

    return E

def order4(E1, E2, pairs, d, g):
    E = 0

    for i in range(1, d):
        for j in range(1, d):
            for k in range(1, d):
                V0i = V(pairs[0], pairs[i], g)
                Vk0 = V(pairs[k], pairs[0], g)
                
                Wij = V(pairs[i], pairs[j], g) - (i==j)*E1
                Wjk = V(pairs[j], pairs[k], g) - (j==k)*E1

                E0 = H0(pairs[0])
                Ei = H0(pairs[i])
                Ej = H0(pairs[j])
                Ek = H0(pairs[k])

                D0i = E0 - Ei
                D0j = E0 - Ej
                D0k = E0 - Ek

                dE = V0i*Wij*Wjk*Vk0 / (D0i*D0j*D0k) 

                E += dE

        E -= E2 * (V0i*V0i)/D0i**2

    return E

def solve(pairs, d, g, order=3):
    gs = pairs[0]

    E0 = H0(gs)
    E1 = V(gs, gs, g)
    E2 = order2(pairs, d, g)
    E3 = order3(E1, pairs, d, g)
    E4 = order4(E1, E2, pairs, d, g)

    E = [E0, E1, E2, E3, E4]    
    E_tot = 0
    for i in range(order+1):
        E_tot += E[i]

    return E_tot


def RS(g, order=3):
    p = 4
    d = int(comb(p,2))

    pairs = []
    for i in range(p):
        for j in range(i+1,p):
            pairs.append((i+1,j+1))

    E = np.zeros_like(g)
    for i in range(len(g)):
        E[i] = solve(pairs, d, g=g[i], order=order)

    return E

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

def run():
    g = np.linspace(-1,1,1000)
    FCI = CI(g)[:,0]

    # fig, ax = plt.subplots()

    E_rs_all = []
    for order in [1,2,3,4]:
        E_rs = RS(g, order=order)
        E_rs_all.append(E_rs)
        # ax.plot(g, E_rs, label=f"RS{order}")

    # ax.plot(g, FCI, label="FCI", c="k", ls="--", alpha=0.4)
    # ax.set(xlabel="g", ylabel="E(g)")
    # ax.legend()
    # plot_utils.save("RS_E")
    # plt.show()


    fig, ax = plt.subplots()
    for i, E_rs in enumerate(E_rs_all):
        order = i+1
        diff = np.abs(FCI-E_rs)
        ax.plot(g, diff, label=f"RS{order}")
    
    ax.set_yscale("log")
    ax.set(xlabel="g", ylabel=r"$|FCI-RS|$", ylim=(5e-5,1))
    ax.legend(ncol=2)
    plot_utils.save("RS_diff")
    plt.show()
# def RS_compare(g, E_combs, orders, labels, plot=True, filename=None):
#     E_RSs = []
#     for order in orders:
#         E_RS = RS(g, order=order)
#         E_RSs.append(E_RS)

#     if plot:
#         fig, ax = plt.subplots()
#         for E_comb, label in zip(E_combs, labels):
#             ax.plot(g, E_comb, label=label)
#         for E_RS, order in zip(E_RSs, orders):
#             label = "$E_{RS}^" + "{" + f"({order})" + "}$"
#             ax.plot(g, E_RS, label=label, ls=":")    

#         ax.set(xlabel="g", ylabel="E(g)")
#         ax.legend()
#         plot_utils.save(filename)
#         plt.show()

#     return E_RSs, orders

if __name__ == "__main__":
    run()