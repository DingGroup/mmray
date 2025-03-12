import numpy as np
import mdtraj
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pickle

prmtop = app.AmberPrmtopFile("./structure/alanine_dipeptide.prmtop")

system = prmtop.createSystem(
    nonbondedMethod=app.CutoffNonPeriodic,
    constraints=app.HBonds,
    implicitSolvent=app.OBC2,
)

integrator = mm.LangevinMiddleIntegrator(
    300 * unit.kelvin,
    1.0 / unit.picoseconds,
    2 * unit.femtoseconds,
)

platform = mm.Platform.getPlatformByName("Reference")

context = mm.Context(system, integrator, platform)


Ts = 1.0 / np.linspace(1.0 / 600, 1.0 / 300, 8)
us = {}
for T in Ts:
    us[T] = []
    traj = mdtraj.load(f"./output/traj/T_{T:.2f}.dcd", top="./structure/alanine_dipeptide.pdb")
    for xyz in traj.xyz:
        context.setPositions(xyz)
        state = context.getState(getEnergy=True)
        u = state.getPotentialEnergy()
        u = u.value_in_unit(unit.kilojoule_per_mole)
        us[T].append(u)
    us[T] = np.array(us[T])
    print(f"done with T={T:.2f}")

with open("./output/potential_energy.pkl", "wb") as f:
    pickle.dump(us, f)
