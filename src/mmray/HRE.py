from typing import List
import openmm as mm
import openmm.unit as unit
import ray
import numpy as np
from .base import BaseActor


class HRE:
    """
    HRE is a class for running Hamiltonian replica exchange.
    """

    def __init__(self, actors):
        """
        Parameters
        ----------
        actors: list
            A list of HREActor objects
        """

        self.actors = actors
        self.num_replicas = len(self.actors)

        self.record = [list(range(self.num_replicas))]
        self.accept_rate = {}
        self.num_attempts = {}
        for i in range(self.num_replicas - 1):
            self.accept_rate[(i, i + 1)] = 0.0
            self.num_attempts[(i, i + 1)] = 0

        self._num_exchange_calls = 0

        self.Ts = ray.get([actor.get_t.remote() for actor in self.actors])

        self._kb = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        self._kb = self._kb.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)

    def exchange(self):
        record = list(range(self.num_replicas))

        ## Hamiltonian replica exchange

        ## if self._num_exchange_calls is even, attempt exchanges between 0 and 1, 2 and 3, ...
        ## if self._num_exchange_calls is odd, attempt exchanges between 1 and 2, 3 and 4, ...
        for i in range(self._num_exchange_calls % 2, self.num_replicas - 1, 2):
            j = i + 1

            (uiqi, qi), (ujqj, qj) = ray.get(
                [self.actors[i].get_uq.remote(), self.actors[j].get_uq.remote()]
            )

            uiqj = ray.get(self.actors[i].get_u_for_q.remote(qj))
            ujqi = ray.get(self.actors[j].get_u_for_q.remote(qi))

            delta = (uiqj/self.Ts[i] + ujqi/self.Ts[j] - uiqi/self.Ts[i] - ujqj/self.Ts[j])/self._kb
            accept_flag = 0
            if np.random.rand() < np.exp(-delta):
                accept_flag = 1

                ## exchange positions
                self.actors[i].update_q.remote(qj)
                self.actors[j].update_q.remote(qi)
                
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
class HREActor(BaseActor):
    """
    HREActor is a class for running Hamiltonian replica exchange in one replica.
    """

    def __init__(
        self,
        topology: mm.app.Topology,
        system: mm.System,
        integrator: mm.Integrator,
        platform_name: str,
        initial_positions: np.ndarray,
        global_params: dict = {},
        reporters: List = [],
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
        global_params: dict
            The global parameters
        reporters: dict
            The reporters
        """

        super().__init__(
            topology,
            system,
            integrator,
            platform_name,
            initial_positions,            
            reporters
        )

        self.global_params = global_params

        for k, v in self.global_params.items():
            self.simulation.context.setParameter(k, v)