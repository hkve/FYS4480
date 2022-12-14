from scipy.special import binom

ns = [8, 32, 64, 128]
Ns = [4, 8, 16, 32]

for n in ns:
    for N in Ns:
        if n > N:
            comb = binom(n,N)
            print(f"{n = }, {N = }, {comb:.2e}")