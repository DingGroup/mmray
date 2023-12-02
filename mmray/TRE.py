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
        self._kb = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        self._kb = self._kb.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)

    def _exchange_positions(self):
        positions = ray.get([actor.get_positions.remote() for actor in self.actors])

        potential_energies = ray.get(
            [actor.get_potential_energy.remote() for actor in self.actors]
        )
        temperatures = ray.get(
            [actor.get_temperature.remote() for actor in self.actors]
        )

        for i in range(len(self.actors) - 1):
            j = i + 1
            delta = (potential_energies[i] - potential_energies[j]) * (
                1.0 / (self._kb * temperatures[i]) - 1.0 / (self._kb * temperatures[j])
            )
            if random() < np.exp(delta):
                tmp = positions[i]
                positions[i] = positions[j]
                positions[j] = tmp

                tmp = potential_energies[i]
                potential_energies[i] = potential_energies[j]
                potential_energies[j] = tmp

                self.actors[i].update_positions.remote(positions[i])

                if j == len(self.actors) - 1:
                    self.actors[j].update_positions.remote(positions[j])

    def run(self, num_steps, exchange_freq):
        tot_steps = 0
        while tot_steps <= num_steps - exchange_freq:
            for actor in self.actors:
                actor.run_md.remote(exchange_freq)
            tot_steps += exchange_freq
            self._exchange_positions()


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
        reporters
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
            if k == 'DCD':
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

    def get_potential_energy(self):
        state = self.simulation.context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        return potential_energy

    def get_positions(self):
        pos = self.simulation.context.getState(getPositions=True).getPositions()
        pos = np.array(pos.value_in_unit(unit.nanometer))
        return pos

    def update_positions(self, positions):
        self.simulation.context.setPositions(positions)
