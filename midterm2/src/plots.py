import numpy as np
import matplotlib.pyplot as plt

import CI
import HartreeFock
import RayleighSchrodinger

def FCI_and_CID(g):
    E_FCI = CI.plot_eigvals(g, plot=True, filename="FCI", skip_4p4h=False)
    E_CID = CI.plot_eigvals(g, plot=True, filename="CID", skip_4p4h=True)

    Egs_FCI = E_FCI[:,0]
    Egs_CID = E_CID[:,0]

    CI.plot_diff(g, Egs_FCI, Egs_CID, filename="FCI_CID_diff")

def HF(g):
    E_FCI = CI.plot_eigvals(g, plot=False, skip_4p4h=False)
    Egs_FCI = E_FCI[:,0]
    
    E_HF = HartreeFock.HF_compare(g, [Egs_FCI], labels=[r"$E_{FCI}$"], plot=True, filename="HF")
    CI.plot_diff(g, Egs_FCI, E_HF, ylabel="$|E_{FCI}-E_{HF}|$", filename="FCI_HF_diff")


def FCI_and_RS(g, orders=[2,3,4]):
    E_FCI = CI.plot_eigvals(g, plot=False, skip_4p4h=False)
    Egs_FCI = E_FCI[:,0]

    E_RSs, orders = RayleighSchrodinger.RS_compare(g, [Egs_FCI], labels=[r"$E_{FCI}$"], orders=orders, plot=True, filename="RS" + "_".join([str(o) for o in orders]))
    if len(orders) == 1:
        order, = orders
        expo = "{" + f"({order})" + "}"
        ylabel = r"$|E_{FCI}-E_{RS}^" + rf"{expo}|$"
        E_RS, = E_RSs
        CI.plot_diff(g, Egs_FCI, E_RS, ylabel=ylabel, filename=f"FCI_RS{order}_diff")

def CI_AND_RS(g):
    E_CI = CI.plot_eigvals(g, plot=False, skip_4p4h=True)
    Egs_CI = E_CI[:,0]
    E_RS, _ = RayleighSchrodinger.RS_compare(g, [Egs_CI], labels=[r"$E_{CI}$"], orders=[2], plot=True, filename="RS2")
    E_RS, = E_RS
    CI.plot_diff(g, Egs_CI, E_RS, ylabel=r"$|E_{CI} - E_{RS}^{(2)}|$", filename=f"CI_RS2_diff")

if __name__ == "__main__":
    g = np.linspace(-1, 1, 100)
    FCI_and_CID(g)
    HF(g)
    FCI_and_RS(g, orders=[3])
    CI_AND_RS(g)
    FCI_and_RS(g, orders=[4])