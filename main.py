#!/usr/bin/env python
from pylab import *
from scipy.integrate import odeint
from scipy.optimize import brentq
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def SchroedingerFactory(psi, E, x, L, V_0):
    """
    Constructs the Schroedinger Equation callback function for
    use in the odeint library.
    """
    def V(x):
        if abs(x) <= L:
            return V_0 

        return 0

    def SchroedingerEquationCallback (psi, x):
        psi0 = psi[0] # psi - root function
        psi1 = psi[1] # psi - first derivative

        # Construct schroedinger equation with given derivatives
        psi2 = 2 * (V(x) - E)*psi0

        return psi1, psi2

    return SchroedingerEquationCallback

def SolveSchroedinger(psi, energy, x, L, V_0):
    psi = odeint(SchroedingerFactory(psi, energy, x, L, V_0), [1, 0], x)
    
    # odeint returns an array of [psi[x], psi'[x]]. So we are returning
    # the array of values representing psi[x] here
    return psi[:, 0]

def FindEigenstates(energies, edgeValues, x, psi, L, V_0):
    eigenstates = []
    signs = sign(edgeValues)

    # For every value far outside the well look for values = 0.
    # These are the physically allowed states.
    for i in range(len(edgeValues) - 1):
        # Check for sign change and use brentq method to get local zero
        if signs[i] + signs[i + 1] == 0:
            eigenstates.append(
                    brentq(
                        lambda energy: SolveSchroedinger(psi, energy, x, L, V_0)[-1],
                        energies[i],
                        energies[i+1])
                    )

    return eigenstates 


def main():
    V_0 = 20
    L = 0.5
    x = linspace(-2, 2, 1000)
    psi = np.zeros([1000, 2])
    energies = linspace(0, V_0, 100)
    
    edgeValues = []
    for energy in energies:
        psi0 = SolveSchroedinger(psi, energy, x, L, V_0) 
        edgeValues.append(psi0[-1]);

    psi = np.zeros([1000, 2])
    eigenstates = FindEigenstates(energies, edgeValues, x, psi, L, V_0)

    # figure()
    # plot(energies/V_0, edgeValues)
    # title('Values of the $\Psi(b)$ vs. Energy')
    # xlabel('Energy, $E/V_0$')
    # ylabel('$\Psi(x = b)$', rotation='horizontal')
    # for E in eigenstates[1:4]:
    #     plot(E/V_0, [0], 'go')
    #     annotate("E = %.2f"%E, xy = (E/V_0, 0), xytext=(E/V_0, 30))

    psi = np.zeros([1000, 2])
    for E in eigenstates[1:4]:
        psi0 = SolveSchroedinger(psi, E, x, L, V_0)
        plt.plot(x , psi0, lw=1)

    plt.axvline(x=-L);    
    plt.axvline(x=L);    

    plt.ylabel("psi(x)")
    plt.xlabel("x")
    plt.title("Wave function")

    plt.show()

if __name__ == "__main__":
    main()
