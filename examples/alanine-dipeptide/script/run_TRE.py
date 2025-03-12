import openmm as mm
import openmm.unit as unit
import openmm.app as app
import numpy as np
from sys import exit
import pickle
import ray
from mmray import TREActor, TRE
import os
import time
from copy import deepcopy

prmtop = app.AmberPrmtopFile("./structure/alanine_dipeptide.prmtop")
system = prmtop.createSystem(
    nonbondedMethod=app.CutoffNonPeriodic,
    constraints=app.HBonds,
    implicitSolvent=app.OBC2,
)
topology = prmtop.topology
pos = app.PDBFile("./structure/alanine_dipeptide.pdb").getPositions(asNumpy=True)
pos = pos.value_in_unit(unit.nanometers)


Ts = 1.0 / np.linspace(1.0 / 600, 1.0 / 300, 8)
actors = []

ray.init()

for T in Ts:
    topology = deepcopy(prmtop.topology)
    system = deepcopy(system)

    integrator = mm.LangevinMiddleIntegrator(
        T * unit.kelvin,
        1.0 / unit.picoseconds,
        2 * unit.femtoseconds,
    )
    platform_name = "CUDA"

    initial_position = deepcopy(pos)

    os.makedirs("./output/traj", exist_ok=True)
    reporters = {
        "DCD": {
            "file": f"./output/traj/T_{T:.2f}.dcd",
            "reportInterval": 1000,
        },
    }
    actor = TREActor.options(num_cpus=1,num_gpus=1).remote(
        topology, system, integrator, platform_name, initial_position, reporters
    )
    actors.append(actor)

for actor in actors:
    actor.minimize_energy.remote()

tre = TRE(actors)

print("running equilibration")
tre.run(num_steps=10_000, exchange_freq=0)

print("running temperature replica exchange")
start_time = time.time()
tre.run(num_steps=100_000_000, exchange_freq=1000)
end_time = time.time()


print(f"Time elapsed: {end_time - start_time:.2f} seconds")
print("exchagne_rate", tre.accept_rate)

with open("./output/traj/TRE_record.pkl", "wb") as f:
    pickle.dump(
        {"record": tre.record, "accept_rate": tre.accept_rate}, f
    )
    
ray.shutdown()