import numpy as np
import openmm.app as app
import openmm as omm
import openmm.unit as unit
import math
import ray

@ray.remote(num_cpus=1)
class TREActor:
    """
    TREActor is a class for running molecular dynamics in a replica.
    """

    def __init__(
        self,
        system,
        integrator,
        platform_name,
        temperature,
        topology,
        dcd_file_name,
        initial_positions=None,
    ):
        """Construct an instance of TREActor which holding all the information
        for running simulation in one replica.

        Parameters
        ----------
        system : System
            the System which will be simulated
        integrator : LangevinIntegrator
            the LangevinIntegrator which will be used to simulate the system
        platform_name : string
            the name of the Platform used for computing
        temperature : double
            the temperature used by the LangevinIntegrator
        initial_positions: vector< Vec3 > or ndarray of a size of [:, 3]
            the initial positions of particles
        topology : Topology
            the Topology of the System
        dcd_file_name : string
            the dcd file name where the trajectory is saved
        """

        self.system = system
        self.integrator = integrator
        self.platform_name = platform_name
        self.temperature = temperature
        self.initial_positions = initial_positions
        self.topology = topology
        self.dcd_file_name = dcd_file_name

        ## construct the platform and the context
        self.integrator.setTemperature(self.temperature)
        self.platform = omm.Platform.getPlatformByName(self.platform_name)
        self.context = omm.Context(self.system, self.integrator, self.platform)
        if self.initial_positions is not None:
            self.context.setPositions(self.initial_positions)

        ## open the dcd_file
        file_handle = open(dcd_file_name, "bw")
        self.dcd_file = omm.app.dcdfile.DCDFile(
            file_handle, self.topology, self.integrator.getStepSize()
        )

        ## record temperature and potential energy each time the actor
        ## saves the state
        self.temperature_record = []
        self.potential_energy_record = []

    def run_md(self, num_steps):
        self.integrator.step(num_steps)

    def get_potential_energy(self):
        state = self.context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        return potential_energy

    def update_temperature(self, temperature):
        state = self.context.getState(getVelocities=True)
        velocities = state.getVelocities()
        velocities = velocities * math.sqrt(temperature / self.temperature)
        self.context.setVelocities(velocities)
        self.temperature = temperature
        self.integrator.setTemperature(temperature)

    def get_temperature(self):
        return self.temperature

    def save_state(self):
        state = self.context.getState(getPositions=True)
        positions = state.getPositions()
        self.dcd_file.writeModel(positions)
        self.temperature_record.append(self.temperature)
        self.potential_energy_record.append(self.get_potential_energy())

    def get_temperature_record(self):
        return self.temperature_record

    def get_potential_energy_record(self):
        return self.potential_energy_record

    def get_checkpoint(self):
        state = self.context.getState(getPositions=True, getVelocities=True)
        positions = state.getPositions().values_in_unit(unit.nanometer)
        velocities = state.getVelocities().values_in_unit(
            unit.nanometer / unit.picoseconds
        )
        temperature = self.temperature

        return {
            "positions": positions,
            "velocities": velocities,
            "temperature": temperature,
        }

    def load_checkpoint(self, checkpoint):
        self.integrator.setTemperature(checkpoint["temperature"])
        self.context.setPositions(checkpoint["positions"])
        self.context.setVelocities(checkpoint["velocities"])

    def set_parameter(self, name, value):
        self.context.setParameter(name, value)


class TRE:
    """
    A class for running temperature replica exchange simulation
    """

    def __init__(self, actors):
        """Construct TRE with a list of actors, each of which is a replica

        Parameters
        ----------
        actors: a list of TREActor

        """
        self.actors = actors

    def _exchange_temperature(self):
        potential_energies = ray.get(
            [actor.get_potential_energy.remote() for actor in self.actors]
        )
        actor_temperatures = ray.get(
            [actor.get_temperature.remote() for actor in self.actors]
        )
        kbTs = [
            unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * (T * unit.kelvin)
            for T in actor_temperatures
        ]

        for idx_replica in range(len(self.actors) - 1):
            actor_sort_idx = np.argsort(actor_temperatures)

            i = actor_sort_idx[idx_replica]
            j = actor_sort_idx[idx_replica + 1]

            delta = (potential_energies[i] - potential_energies[j]) * (
                1.0 / kbTs[i] - 1.0 / kbTs[j]
            )

            if np.random.rand() < np.exp(delta):
                tmp = actor_temperatures[i]
                actor_temperatures[i] = actor_temperatures[j]
                actor_temperatures[j] = tmp

                self.actors[i].update_temperature.remote(actor_temperatures[i])
                self.actors[j].update_temperature.remote(actor_temperatures[j])

    def run(self, num_steps, exchange_freq, save_freq):
        if exchange_freq < save_freq and save_freq % exchange_freq != 0:
            raise ValueError(
                "save_freq has to be divisible by exchange_freq when it is larger than exchange_freq"
            )
        if exchange_freq > save_freq and exchange_freq % save_freq != 0:
            raise ValueError(
                "exchange_freq has to be divisible by save_freq when it is larger than save_freq"
            )

        if exchange_freq < save_freq:
            num_iterations = num_steps // exchange_freq
            for idx_iter in range(num_iterations):
                if (idx_iter + 1) % 100 == 0:
                    print("idx_iter: {}".format(idx_iter), flush=True)

                [actor.run_md.remote(exchange_freq) for actor in self.actors]
                if (idx_iter + 1) % (save_freq // exchange_freq) == 0:
                    [actor.save_state.remote() for actor in self.actors]
                self._exchange_temperature()

        else:
            num_iterations = num_steps // save_freq
            for idx_iter in range(num_iterations):
                if (idx_iter + 1) % 100 == 0:
                    print("idx_iter: {}".format(idx_iter), flush=True)

                [actor.run_md.remote(save_freq) for actor in self.actors]
                [actor.save_state.remote() for actor in self.actors]
                if (idx_iter + 1) % (exchange_freq // save_freq) == 0:
                    self._exchange_temperature()

    def get_checkpoints(self):
        checkpoints = ray.get([actor.get_checkpoint.remote() for actor in self.actors])
        return checkpoints

    def load_checkpoints(self, checkpoints):
        assert len(checkpoints) == len(self.actors)
        [
            self.actors[i].load_checkpoint.remote(checkpoints[i])
            for i in range(len(checkpoints))
        ]
