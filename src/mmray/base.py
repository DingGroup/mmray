from typing import List
import warnings
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
import ray


class BaseActor:
    """
    BaseActor is a class for running molecular dynamics in a replica.
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

        self.simulation = mm.app.Simulation(
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


    def run_md(self, n_steps):
        """
        Run a molecular dynamics simulation for n_steps steps.

        Parameters
        ----------
        n_steps: int
            The number of steps to run the simulation for
        """
        self.simulation.step(n_steps)

    def minimize_energy(self, tolerance=10, maxIterations=0):
        self.simulation.minimizeEnergy(tolerance=tolerance, maxIterations=maxIterations)

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

    def get_uq(self):
        """Get the potential energy (u) and positions (q) of the simulation."""
        state = self.simulation.context.getState(getEnergy=True, getPositions=True)
        u = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        q = state.getPositions()
        q = np.array(q.value_in_unit(unit.nanometer))
        return [u, q]

    def get_q(self):
        """Get the positions (q) of the simulation."""
        state = self.simulation.context.getState(getPositions=True)
        q = state.getPositions()
        q = np.array(q.value_in_unit(unit.nanometer))
        return q

    def get_u(self):
        return (
            self.simulation.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )

    def get_t(self):
        return self.simulation.integrator.getTemperature().value_in_unit(unit.kelvin)

    def update_qv(self, positions, velocities):
        self.simulation.context.setPositions(positions)
        self.simulation.context.setVelocities(velocities)

    def update_q(self, positions):
        self.simulation.context.setPositions(positions)

    def get_u_for_q(self, positions):
        old_q = self.get_q()
        self.update_q(positions)
        u = self.get_u()
        self.update_q(old_q)
        return u
    
