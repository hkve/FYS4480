import numpy as np
import matplotlib.pyplot as plt
import plot_utils

def H0(state):
    p, _ = state
    return (p-1)

def V(l, b, g, d, G):
    l_lvl, l_spin = l
    b_lvl, b_spin = b
    g_lvl, g_spin = g
    d_lvl, d_spin = d

    
    if l_spin == g_spin and b_spin == d_spin:
        if l_spin != b_spin and g_spin != d_spin:
            if l_lvl == b_lvl and g_lvl == d_lvl:
                return -0.5*G
    
    return 0

def density_matrix(C, Ns, Np):
    rho = np.zeros_like(C)

    for p in range(Ns):
        for q in range(Ns):
            s = 0
            for j in range(Np):
                s += C[j, p] * C[j, q]
            
            rho[p, q] = s

    return rho

def solve(Np, Ns, G=1, tol=1e-8, maxiters=1000):

    states = []
    for i in range(1,Np+1):
        states.append((i, 0))
        states.append((i, 1))

    C = np.eye(Ns,Ns)
    rho = density_matrix(C, Ns, Np)

    diff = 1
    iters = 0

    sp_E_old = np.zeros(Ns)    
    sp_E_new = np.zeros(Ns)
    while iters < maxiters and diff > tol:
        HFmat = np.zeros_like(rho)

        for p, state in enumerate(states):
            HFmat[p,p] = H0(state)

        for l in range(Ns):
            for g in range(Ns):

                elmSum = 0

                for b in range(Ns):
                    for d in range(Ns):
                        elmSum += rho[b,d]*V(states[l], states[b], states[g], states[d], G=G)
   
                HFmat[l,g] += elmSum

        sp_E_new, C = np.linalg.eigh(HFmat)
        rho = density_matrix(C.T, Ns, Np)
        diff = np.sum(np.abs(sp_E_new-sp_E_old))/Ns
        sp_E_old = sp_E_new
        iters += 1

    E_gs_1body = 0
    for a in range(Ns):
        E_gs_1body += rho[a,a] * H0(states[a])
    
    E_gs_2body = 0
    for a in range(Ns):
        for b in range(Ns):
            for g in range(Ns):
                for d in range(Ns):
                    E_gs_2body += rho[a,g]*rho[b,d]*V(states[a], states[b], states[g], states[d], G)
    
    E = E_gs_1body + 0.5*E_gs_2body
    return E

def HF_solve(g):
    Ns = 8
    Np = 4
    E_HF = np.zeros_like(g)

    for i in range(len(g)):
        E_HF[i] = solve(Np, Ns, G=g[i])

    return E_HF


def HF_compare(g, E_combs, labels=[r"$E_{FCI}$"], plot=True, filename=None):
    E_HF = HF_solve(g)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(g, E_HF, label=r"$E_{HF}$")
        for E, label in zip(E_combs, labels):
            ax.plot(g, E, label=label)

        ax.set(xlabel="g", ylabel="E(g)")
        ax.legend()
        plot_utils.save(filename)
        plt.show()

    return E_HF

if __name__ == "__main__":
    E_HF = HF_solve(np.linspace(-1,1,10))
    print(E_HF)