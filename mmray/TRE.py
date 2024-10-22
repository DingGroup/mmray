import openmm as mm
import openmm.app as app
import ray
import numpy as np
import openmm.unit as unit
from random import random
import warnings


class TRE:
    """
    A class for running temperature replica exchange simulation
    """

    def __init__(self, actors):
        """Construct TRE with a list of actors, each of which is a replica.

        Parameters
        ----------
        actors: a list of TREActor

        """
        self.actors = actors
        self.num_replicas = len(self.actors)

        self._kb = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        self._kb = self._kb.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
        
        self.T = ray.get(
            [actor.get_temperature.remote() for actor in self.actors]
        )
        self.kbT = self._kb * np.array(self.T)
        self.one_over_kbT = 1.0 / self.kbT
        
        self.exchange_record = [list(range(self.num_replicas))]
        self.exchange_rate = 0
        self.num_exchange_attempts = 0

    def _exchange_position(self):
        energy_and_position = ray.get(
            [actor.get_energy_and_position.remote() for actor in self.actors]
        )

        record = list(range(self.num_replicas))

        for i in range(len(self.actors) - 1):
            j = i + 1
            delta = (energy_and_position[i][0] - energy_and_position[j][0]) * (
                self.one_over_kbT[i] - self.one_over_kbT[j]
            )

            flag = 0                        
            if random() < np.exp(delta):
                tmp = energy_and_position[i]
                energy_and_position[i] = energy_and_position[j]
                energy_and_position[j] = tmp

                self.actors[i].update_positions.remote(energy_and_position[i][1])

                if j == len(self.actors) - 1:
                    self.actors[j].update_positions.remote(energy_and_position[j][1])

                tmp = record[i]
                record[i] = record[j]
                record[j] = tmp

                flag = 1
            
            self.num_exchange_attempts += 1
            self.exchange_rate = self.exchange_rate * (self.num_exchange_attempts - 1) / self.num_exchange_attempts + flag / self.num_exchange_attempts

        self.exchange_record.append(record)

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
                self._exchange_position()


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

    def get_temperature(self):
        return self.simulation.integrator.getTemperature().value_in_unit(unit.kelvin)

    def get_energy_and_position(self):
        state = self.simulation.context.getState(
            getEnergy=True, getPositions=True
        )
        potential_energy = state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        pos = state.getPositions()
        pos = np.array(pos.value_in_unit(unit.nanometer))
        return potential_energy, pos

    def update_positions(self, positions):
        self.simulation.context.setPositions(positions)

    def minimize_energy(self, tolerance = 10, maxIterations = 0):
        self.simulation.minimizeEnergy(tolerance = tolerance, maxIterations = maxIterations)
        
