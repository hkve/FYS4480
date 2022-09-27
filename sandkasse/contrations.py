from sympy import *
from sympy.physics.secondquant import *

i, j = symbols('i,j', below_fermi=True)
a, b = symbols('a,b', above_fermi=True)
p, q = symbols('p,q')
res = simplify(wicks(Fd(i)*F(a)*Fd(p)*F(q)*Fd(b)*F(j), keep_only_fully_contracted=True))

print(res)

