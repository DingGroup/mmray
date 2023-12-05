import openmm as mm
import openmm.app as app
import ray
import numpy as np
import openmm.unit as unit
from random import random
import warnings
import math


class TRE:
    """
    A class for running temperature replica exchange simulations.
    """

    def __init__(self, actors):
        """Construct TRE with a list of actors, each of which is a replica.
        The method is based on the following paper:
        "Replica-exchange molecular dynamics method for protein folding"
        DOI: https://doi.org/10.1016/S0009-2614(99)01123-9

        Parameters
        ----------
        actors: a list of TREActor

        """
        self.actors = actors
        self.num_replicas = len(self.actors)

        self._kb = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        self._kb = self._kb.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)

        self.T = ray.get([actor.get_t.remote() for actor in self.actors])
        self.kbT = self._kb * np.array(self.T)
        self.one_over_kbT = 1.0 / self.kbT

        self.record = [list(range(self.num_replicas))]
        self.accept_rate = {}
        self.num_attempts = {}
        for i in range(self.num_replicas - 1):
            self.accept_rate[(i, i + 1)] = 0.0
            self.num_attempts[(i, i + 1)] = 0

        self._num_exchange_calls = 0

    def exchange(self):
        uqv = ray.get([actor.get_uqv.remote() for actor in self.actors])
        record = list(range(self.num_replicas))

        ## if self._num_exchange_calls is even, attempt exchanges between 0 and 1, 2 and 3, ...
        ## if self._num_exchange_calls is odd, attempt exchanges between 1 and 2, 3 and 4, ...

        for i in range(self._num_exchange_calls % 2, self.num_replicas - 1, 2):
            j = i + 1
            delta = (uqv[i][0] - uqv[j][0]) * (
                self.one_over_kbT[i] - self.one_over_kbT[j]
            )

            accept_flag = 0
            if random() < np.exp(delta):
                accept_flag = 1

                ## exchange positions and velocities
                ## note that the velocities need to be rescaled
                self.actors[i].update_qv.remote(
                    uqv[j][1], uqv[j][2] * math.sqrt(self.T[i] / self.T[j])
                )
                self.actors[j].update_qv.remote(
                    uqv[i][1], uqv[i][2] * math.sqrt(self.T[j] / self.T[i])
                )
                record[i], record[j] = record[j], record[i]

            self.num_attempts[(i, j)] += 1
            self.accept_rate[(i, j)] = (
                self.accept_rate[(i, j)]
                * (self.num_attempts[(i, j)] - 1)
                / self.num_attempts[(i, j)]
                + accept_flag / self.num_attempts[(i, j)]
            )

        self._num_exchange_calls += 1
        self.record.append(record)

    def run(self, num_steps, exchange_freq):
        if exchange_freq <= 0:
            for actor in self.actors:
                actor.run_md.remote(num_steps)
        else:
            tot_steps = 0
            while tot_steps <= num_steps - exchange_freq:
                for actor in self.actors:
                    actor.run_md.remote(exchange_freq)
                tot_steps += exchange_freq
                self.exchange()


@ray.remote
class TREActor:
    """
    TREActor is a class for running molecular dynamics in a replica.
    """

    def __init__(
        self,
        topology,
        system,
        integrator,
        platform_name,
        initial_positions,
        reporters: dict = {},
    ):
        """
        Parameters
        ----------
        topology: openmm.app.Topology
            The topology of the system
        system: openmm.System
            The system
        integrator: openmm.Integrator
            The integrator
        platform_name: str
            The name of the platform to run the simulation on.
            See https://docs.openmm.org/latest/api-python/generated/simtk.openmm.app.Platform.html
            for details.
        initial_positions: np.ndarray
            The initial positions of the system
        reporters: dict

        """

        self.topology = topology

        self.system = system
        self.integrator = integrator
        self.platform_name = platform_name
        self.platform = mm.Platform.getPlatformByName(self.platform_name)

        self.initial_positions = initial_positions
        self.reporters = reporters

        self.simulation = app.Simulation(
            self.topology,
            self.system,
            self.integrator,
            self.platform,
        )

        if type(self.reporters) is not dict:
            raise ValueError("reporters must be a dictionary")

        for k, v in self.reporters.items():
            if k == "DCD":
                reporter = app.DCDReporter(**v)
                self.simulation.reporters.append(reporter)

        if self.initial_positions is not None:
            self.simulation.context.setPositions(self.initial_positions)

        if self.initial_positions is None:
            pos = self.simulation.context.getState(getPositions=True).getPositions()
            pos = np.array(pos.value_in_unit(unit.nanometer))
            if np.all(pos == 0):
                warnings.warn(
                    "Initial positions are all zero. This is probably not what you want. You can set the initial positions in the simulation object or pass them to the actor using the iniital_positions argument."
                )

    def run_md(self, num_steps):
        self.simulation.step(num_steps)

    def get_t(self):
        """Get the temperature of the simulation.

        Returns:
            float: The temperature of the simulation in Kelvin.
        """
        return self.simulation.integrator.getTemperature().value_in_unit(unit.kelvin)

    def get_uqv(self):
        """Get the potential energy (u), positions (q), and velocities (v) of the simulation."""
        state = self.simulation.context.getState(
            getEnergy=True, getPositions=True, getVelocities=True
        )
        u = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        q = state.getPositions()
        q = np.array(q.value_in_unit(unit.nanometer))
        v = state.getVelocities()
        v = np.array(v.value_in_unit(unit.nanometer / unit.picosecond))
        return [u, q, v]

    def update_qv(self, positions, velocities):
        self.simulation.context.setPositions(positions)
        self.simulation.context.setVelocities(velocities)

    def minimize_energy(self, tolerance=10, maxIterations=0):
        self.simulation.minimizeEnergy(tolerance=tolerance, maxIterations=maxIterations)
