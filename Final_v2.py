import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scipy as scy
from scipy import sparse

# psi_in: initial wave function
# f_t: time dependence
# Difference between perfect bragg oscillations and difference when an actual device is applied
# LC basis --> LC circuit is modelled by transmon quibit. Inductor is Josephene

# E_L is Josephine energy

# Start with the full hamiltonian but then consider H_0 with pertubations

# Todo: What is phi_lc?
def phi_lc(E_C,E_L,nlevs): # Delta in picture notation
    """Flux (phase) operator in the LC basis."""
    return (8 * E_C / E_L) ** (0.25) * qt.position(nlevs)


def n_lc(E_C,E_L,nlevs): # Charging energy (potential energy of quadratic and higher order terms). Models capacitance
    """Charge operator in the LC basis."""
    return (E_L / (8 * E_C)) ** (0.25) * qt.momentum(nlevs)


def hamiltonian(E_C,E_L,nlevs=15): # Hamilitonian is combination of charge operator and the flux operator
    """Qubit Hamiltonian in the LC basis."""

    phi = phi_lc(E_C,E_L,nlevs)
    n = n_lc(E_C,E_L,nlevs)
    return 4 * E_C * n ** 2 + 0.5 * E_L * ( phi ** 2 - phi ** 4/ 12)


def n(E_C, E_L, nlevs, neff=None):
    """Charge operator in the qubit eigenbasis.

    Parameters
    ----------
        The number of qubit eigenstates if different from `self.nlev`.

    Returns
    -------
    :class:`qutip.Qobj`
        The charge operator.
    """
    if neff is None:
        neff = nlevs
    if neff < 1 or neff > nlevs:
        raise Exception('`neff` is out of bounds.')
    H1 = hamiltonian(E_C,E_L,nlevs)
    ens, evecs = H1.eigenstates()
    n_op = np.zeros((neff, neff), dtype=complex)
    for ind1 in range(neff):
        for ind2 in range(neff):
            n_op[ind1, ind2] = n_lc(E_C,E_L,nlevs).matrix_element(
                evecs[ind1].dag(), evecs[ind2])
    return qt.Qobj(n_op)


def ket_to_probs(ket_t,neff = 4):
    probs = np.zeros([neff,len(ket_t.states)])
    for tdx, psi in enumerate(ket_t.states):
        for ndx in range(neff):
            probs[ndx,tdx] = np.abs(qt.basis(4,ndx).overlap(psi))**2
    return probs


def H_coord(E_C,E_L,dx    = 0.02):
    x = np.arange(-np.pi, np.pi, dx)
    V = E_L *(0 + x **2 /2. - 1.*x**4/24.)
    V_mat = sparse.diags(V)
    V_mat.toarray()
    D2 = 8*E_C*sparse.diags([1, -2, 1], [-1, 0, 1], shape=(x.size, x.size)) / dx**2/2
    D2.toarray()*dx**2
    return qt.Qobj(-D2+V_mat)

# We now construct an evolution operator for time-independent Hamiltonian
EC = 0.2
EJ = 20. # generate your own EJ+ (1-2*np.random.random())
w01 = np.sqrt(8*EC*EJ)
print('a rough estimate for the transition frequency ',w01)

#the field with magnitude 1 in x-direction
H = hamiltonian(EC,EJ,nlevs  =10)
ens = H.eigenenergies()
print('The transition frequency is',ens[1]-ens[0],'GHz')
w_drive  = (ens[1]-ens[0])*2*np.pi

# The anharmonicity:
anhom = ens[2]-ens[1]-(ens[1]-ens[0])
print('anharmonicity: ', anhom)

S = H.eigenstates()
print('Eigenvalues, or eigenenergies, are in the first part of S\n')
print(S[0],'\n')
print('Eigenvectors, or eigenkets, are in the second part of S\n')
#print(S[1])

# The charge matrix element
M = n(EC,EJ,10,neff=4)

Heff = qt.Qobj(np.diag(ens[:4]))
print('For the harmonic oscillator the ratio is ',np.sqrt(2.))
print('For the transmon, this ratio is different ',M[1,2]/M[0,1])

def f_t(t,args):
    return np.sin(args['w']*t)

# Smooth turn on/off, gives better result for the gates
def f_t1(t,args):
    return np.sin(args['w']*t) * (np.sin(np.pi*t/args['tg']) ** 2)

t_gate = 77.
A_drive = 0.005
psi_in = qt.basis(4,0)
t_list = np.linspace(0,t_gate,1000)

H_0 = 2.*np.pi*Heff
H_full = [H_0,[2.*np.pi*A_drive*M,f_t]]
ket_t = qt.sesolve(H_full, psi_in,t_list, args={'w': w_drive} )

probs = ket_to_probs(ket_t)
plt.plot(t_list,probs[0,:])
plt.plot(t_list,probs[1,:])

plt.show()

probs = ket_to_probs(ket_t)
#plt.plot(t_list,probs[0,:])
#plt.plot(t_list,probs[1,:])
plt.plot(t_list,probs[2,:])
plt.show()

H_0 = 2.*np.pi*Heff
H_full = [H_0,[2.*np.pi*A_drive*M,f_t1]]
ket_t = qt.sesolve(H_full, psi_in,t_list, args={'w': w_drive, 'tg': t_gate} )

probs = ket_to_probs(ket_t)
plt.plot(t_list,probs[0,:])
plt.plot(t_list,probs[1,:])
plt.plot(t_list,probs[2,:])
plt.xlim(70,80)
plt.show()

probs = ket_to_probs(ket_t)
plt.plot(t_list,probs[2,:])
plt.show()

print(probs[:,-1])

#  We can also use the basis in the coordinate representation
#  It is less efficient than the LC basis for our problem, but more universal
H1 = H_coord(0.2, 20.,dx = 0.005)
ens = H1.eigenenergies()

