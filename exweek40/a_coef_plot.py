from a import vecsvals
import numpy as np
import matplotlib.pyplot as plt

def plot2coefs(V):
    Ws = np.linspace(-1,1, 100)

    s1 = np.zeros((2,len(Ws)))
    s2 = np.zeros((2,len(Ws)))

    for i, W in enumerate(Ws):
        vals, vecs = vecsvals(V, W)
        s1[0,i] = vecs[0,0]**2
        s1[1,i] = vecs[2,0]**2

        s2[0,i] = vecs[0,1]**2
        s2[1,i] = vecs[2,1]**2

    fig, ax = plt.subplots()

    c = ["r", "b"]
    ls = ["-", "--"]
    coef = [1,3]
    for i in range(2):
        ax.plot(Ws, s1[i],c=c[0], ls=ls[i], label=rf"$gs$ c{coef[i]}")
        ax.plot(Ws, s2[i],c=c[1], ls=ls[i], label=rf"$ex$ c{coef[i]}")

    ax.legend()
    plt.show()

plot2coefs(-0.3)