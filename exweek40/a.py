import numpy as np



def vecsvals(V, W, eps=2):
    n = 5
    H = np.zeros((n,n))
    np.fill_diagonal(H, np.array([-2*eps, -eps+3*W, 4*W, eps+3*W, 2*eps]))
    H[0,2] = H[2,0] = H[2,4] = H[4,2]  = np.sqrt(6)*V
    H[1,3] = H[3,1] = 3*V

    vals, vecs = np.linalg.eig(H)
    ids = np.argsort(vals)
    vals = vals[ids]
    vecs = vecs[:, ids]

    return vals, vecs

if __name__ == "__main__":
    # Pretty pure
    V = -1/3
    W = -1/4

    vals, vecs = vecsvals(V, W)

    print(vals)
    print(vecs)
    print("\n"*3)

    V = -4/3
    W = -1

    vals, vecs = vecsvals(V, W)

    print(vals)
    print(vecs)
