import pickle
import numpy as np
from bayesmbar import FastMBAR
import openmm.unit as unit
import mdtraj
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

with open("./output/potential_energy.pkl", "rb") as f:
    us = pickle.load(f)

Ts = 1.0 / np.linspace(1.0 / 600, 1.0 / 300, 8)

num_conf = np.array([len(us[T]) for T in Ts])
us = np.concat([us[T] for T in Ts])

kbT = unit.MOLAR_GAS_CONSTANT_R * Ts * unit.kelvin
kbT = kbT.value_in_unit(unit.kilojoule_per_mole)

us = us[None, :] / kbT[:, None]

mbar = FastMBAR(us, num_conf, verbose=True, method="Newton")

logw = -us[-1] - mbar.log_prob_mix
logw -= np.max(logw)

w = np.exp(logw)
w /= np.sum(w)

with open("./output/weights.pkl", "wb") as f:
    pickle.dump(w, f)

prmtop = mdtraj.load_prmtop("./structure/alanine_dipeptide.prmtop")
phi_indices = [4, 6, 8, 14]
psi_indices = [6, 8, 14, 16]

phi_all = []
psi_all = []
for T in Ts:
    traj = mdtraj.load(f"./output/traj/T_{T:.2f}.dcd", top=prmtop)
    phi = mdtraj.compute_dihedrals(traj, [phi_indices])
    psi = mdtraj.compute_dihedrals(traj, [psi_indices])

    phi_all.append(phi.flatten())
    psi_all.append(psi.flatten())

phi = np.concatenate(phi_all)
psi = np.concatenate(psi_all)

fig = plt.figure()
plt.clf()
plt.hist2d(
    phi,
    psi,
    bins=30,
    cmap="viridis",
    norm=LogNorm(),
    range=[[-np.pi, np.pi], [-np.pi, np.pi]],
    weights=w,
    density=True,
)
plt.xlabel(r"$\phi$")
plt.ylabel(r"$\psi$")
plt.title(r"T = 300 K")
plt.gca().set_aspect("equal")
plt.colorbar()
plt.tight_layout()
plt.savefig("./output/phi_psi_reweight.png")