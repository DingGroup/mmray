import mdtraj
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sys import exit
from copy import deepcopy
from matplotlib.colors import LogNorm

#### load the exchange record
with open("./output/traj/TRE_record.pkl", "rb") as f:
    data = pickle.load(f)
exchange_record = np.array(data["record"])

#### rewinding the exchange record to get the replica record
replica_record = []
temperature_idx = np.arange(exchange_record.shape[1])
for i in range(exchange_record.shape[0]):
    temperature_idx = temperature_idx[exchange_record[i]]
    replica_record.append(deepcopy(temperature_idx))

replica_record = np.array(replica_record)

#### plot the replica record
fig = plt.figure(figsize=(20, 4))
plt.clf()
for j in range(replica_record.shape[1]):
    plt.plot(replica_record[:200, j])
plt.savefig("./output/record.png")

#### plot the phi-psi plot for each replica
Ts = 1.0 / np.linspace(1.0 / 600, 1.0 / 300, 8)
prmtop = mdtraj.load_prmtop("./structure/alanine_dipeptide.prmtop")
phi_indices = [4, 6, 8, 14]
psi_indices = [6, 8, 14, 16]

fig = plt.figure(figsize=(4.8 * 2, 3.6 * 4))
plt.clf()

for idx, T in enumerate(Ts):
    traj = mdtraj.load(f"./output/traj/T_{T:.2f}.dcd", top=prmtop)
    phi = mdtraj.compute_dihedrals(traj, [phi_indices])
    psi = mdtraj.compute_dihedrals(traj, [psi_indices])

    phi = phi.flatten()
    psi = psi.flatten()

    plt.subplot(4, 2, idx + 1)
    ax = plt.hist2d(
        phi,
        psi,
        bins=30,
        cmap="viridis",
        norm=LogNorm(vmin=1e-4, vmax=1),
        range=[[-np.pi, np.pi], [-np.pi, np.pi]],
        density = True
    )

    # plt.plot(phi, psi, '.', alpha = 0.5)
    plt.title(f"T = {T:.2f}")
    plt.xlabel(r"$\Phi$")
    plt.ylabel(r"$\Psi$")
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.colorbar()
    plt.gca().set_aspect("equal", adjustable="box")
    print(f"plotting replica {idx}")

plt.tight_layout()    
plt.savefig("./output/phi-psi.png")

