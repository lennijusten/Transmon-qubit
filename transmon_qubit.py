# Transmon qubit with weak AC Josephson effect
# Author: Lenni Justen and Dr. Maxim Vavilov

# See Koch, J. et. al, (2007). Charge-insensitive qubit design derived from the Cooper pair box.
# Physical Review A, 76(4), 042319. https://doi.org/10.1103/PhysRevA.76.042319

import qutip as qt
import scipy as scy
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from qutip import destroy

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

# Calculating anharmonicity
# Source https://qiskit.org/textbook/ch-quantum-hardware/transmon-physics.html
E_C = 300e6
E_J = 10 ** 3 * E_C
anharm = -E_C
w = 5e9

N_phis = 101
phis = np.linspace(-np.pi, np.pi, N_phis)
mid_idx = int((N_phis + 1) / 2)

# potential energies of the QHO & transmon
U_QHO = 0.5 * E_J * phis ** 2
U_QHO = U_QHO / w
U_transmon = (E_J - E_J * np.cos(phis))
U_transmon = U_transmon / w

N = 35
N_energies = 5
c = destroy(N)
H_QHO = w * c.dag() * c
E_QHO = H_QHO.eigenenergies()[0:N_energies]
H_transmon = w * c.dag() * c + (anharm / 2) * (c.dag() * c) * (c.dag() * c - 1)
E_transmon = H_transmon.eigenenergies()[0:2 * N_energies]

print(E_QHO[:4])
print(E_transmon[:8])

fig, axes = plt.subplots(1, 1, figsize=(6, 6))

axes.plot(phis, U_transmon, '-', color='orange', linewidth=3.0)
axes.plot(phis, U_QHO, '--', color='blue', linewidth=3.0)

E_corrected = []
for eidx in range(1, N_energies):
    delta_E_QHO = (E_QHO[eidx] - E_QHO[0]) / w
    delta_E_transmon = (E_transmon[2 * eidx] - E_transmon[0]) / w
    E_corrected.append(delta_E_transmon)

    QHO_lim_idx = min(np.where(U_QHO[int((N_phis + 1) / 2):N_phis] > delta_E_QHO)[0])
    trans_lim_idx = min(np.where(U_transmon[int((N_phis + 1) / 2):N_phis] > delta_E_transmon)[0])
    trans_label, = axes.plot([phis[mid_idx - trans_lim_idx - 1], phis[mid_idx + trans_lim_idx - 1]],
                             [delta_E_transmon, delta_E_transmon], '-', color='orange', linewidth=3.0)
    qho_label, = axes.plot([phis[mid_idx - QHO_lim_idx - 1], phis[mid_idx + QHO_lim_idx - 1]],
                           [delta_E_QHO, delta_E_QHO], '--', color='blue', linewidth=3.0)

axes.set_xlabel('Phase $\phi$', fontsize=18)
axes.set_ylabel('Energy Levels / $\hbar\omega$', fontsize=18)
axes.set_ylim(-0.2, 5)

qho_label.set_label('QHO Energies')
trans_label.set_label('Transmon Energies')
axes.legend(loc=2, fontsize=14)
plt.show()

A_harm = (E_transmon[2]-E_transmon[1])-(E_transmon[1]-E_transmon[0])

hbar = 1.054*10**-34
omega = 5e9
E_ratio = np.linspace(1**-5,0.5,100)
anhom = -2*np.pi*hbar*omega*(8*E_ratio)**(-1/2)

plt.plot(E_ratio, anhom)
plt.ylabel('Anharmonicity $\\alpha$')
plt.xlabel('$E_C/E_J$')
plt.show()

