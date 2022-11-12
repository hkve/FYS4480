import numpy as np

def exact(eps1=-2, eps2=-1, lmbda=-0.5, diag = False):
    t1 = eps1+eps2
    t2 = np.sqrt((eps2 - eps1)**2 + 4*lmbda**2)

    E = (t1-t2, t1+t2)

    if diag:
        H = np.array([
            [eps1, lmbda],
            [lmbda, eps2]
        ])

        vals, vecs = np.linalg.eigh(H)

        return E, vals
    return E


if __name__ == "__main__":
    print(exact(lmbda=0.1, diag=True))