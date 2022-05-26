import time
from abc import ABCMeta, abstractmethod
from typing import Union, Optional

import gym

from agent import Trainable
from environments.lineFollowingEnvironment import LineFollowingEnv
from environments.lineFollowingEnvironment2 import LineFollowingEnv2
from settings import gv
import numpy as np
import sys


class PlaceCellAnalog2ActivationLayer:
    """
    Class to store the positions of a place cell layer.
    """
    positions: np.ndarray = None  # list of center positions
    distance_pc: np.ndarray  # can be cached without vector quantization

    def __init__(self, placecell_range: np.ndarray, num_cells_per_dim: np.ndarray):
        """
        Initialize by covering a n-dim space uniformly (per dimension).
        :param placecell_range: min max for each dimension
        :param num_cells_per_dim: resolution per dimension
        """
        # Average distance between centers covered by the pc per dimension
        self.distance_pc = np.abs((placecell_range[:, 1] - placecell_range[:, 0])) / (num_cells_per_dim - 1)
        numdims = placecell_range.shape[0]
        # for each dimension create a grid, then transform to have all coordinates
        pos = np.mgrid[tuple(slice(0., num_cells_per_dim[dim]) for dim in range(numdims))].T.reshape(-1,
                                                                                                     numdims)  # shape: (num_cells_per_dim^numdims, num_cells_per_dim)
        self.positions: np.ndarray = pos * self.distance_pc
        self.positions += placecell_range[:, 0]  # add offset to each element

        if sys.maxsize > 2 ** 32:  # check if 64bit
            self.vq_decay: np.float128 = np.float128(1.0)  # of vector quantization
        else:
            self.vq_decay: np.float64 = np.float64(1.0)  # of vector quantization

        self.sigmafactor: float = 1
        if gv.workerdata is not None:
            self.sigmafactor = gv.workerdata.get("receptivefieldsize", 1)

    def activation(self, observation: np.ndarray) -> np.ndarray:
        """
        calculate activation for each neuron with exponential kernel like in fremaux2013 (eq. 22)
        When vector quantization is activated this function call has side-effects as it moves the positions.
        :param observation:
        :return: activation levels
        """

        # changes every time, so cannot be cached
        # rezsigma = (len(dists) / np.sum(dists)
        k_averagedistancing = False
        if k_averagedistancing:
            rezsigma: np.ndarray = np.empty_like(self.positions)
            k = 4
            for i in range(len(rezsigma)):
                # distance to other place cells
                dist = np.linalg.norm(self.positions[i] - self.positions, axis=1)
                rezsigma[i] = np.average(dist[np.argpartition(dist, 1 + k)[1:1 + k]])  # ignore self
        else:
            rezsigma = self.sigmafactor / self.distance_pc  # calculation per dimension
        # use lp2 norm, weighted by dimensionality density
        dists: np.ndarray = np.linalg.norm((self.positions - observation) * rezsigma, axis=1)
        dists2: np.ndarray = dists ** 2  # todo why square the 2 norm?
        if gv.vq_learning_scale > 0:
            self.vector_quantization(observation, dists2)
        vec = np.exp(-dists2)  # use average distance
        return vec / np.sum(vec)  # normalize

    def vector_quantization(self, observation: np.ndarray, dist2: np.ndarray):
        """Move the neurons."""
        if sys.maxsize > 2 ** 32:  # check if 64bit
            self.vq_decay *= np.float128(1 - gv.vq_decay)  # exponentially decrease strength of vq
        else:
            self.vq_decay *= np.float64(1 - gv.vq_decay)  # exponentially decrease strength of vq
        changeamount = gv.vq_learning_scale * np.exp(-dist2 / self.vq_decay)
        self.positions += (observation - self.positions) * changeamount[:, np.newaxis]


Weightstorage: type = np.ndarray  # single dimensional or two dimensional with columns source target weight
# TypeAlias will be part in python 3.10
Action = Union[int, list[float]]


class Actor(Trainable, metaclass=ABCMeta):
    """Part of the agent that manages behaviour. Includes analog-2-spike and spike-2-analog. Descibred by
        Figure 4.6 in thesis."""

    def __init__(self, environment: Optional[gym.Env] = None):
        """

        :param environment: only used to derive output2action. todo overwrite the method instead
        """
        self.env: gym.Env = environment
        self.log_m: list[float] = []  # log neurotransmitter m every cycle
        self.weightlog: list[Weightstorage] = []  # log storing historic weight data
        # stats
        self.totalcycles = 0
        self.starttime = time.time_ns()

    def output2actions(self, output: np.ndarray):
        if isinstance(self.env, LineFollowingEnv) or isinstance(self.env, LineFollowingEnv2):
            action = [np.clip(0.5 - np.sum(output), 0., 1.)]  # analog value between 0 and 1
        else:
            # rate should only be a scalar value
            action: int = int(np.sign(np.sum(output)) == 1)  # 0 or 1
        return action

    @abstractmethod
    def cycle(self, time: float, observation_in: np.ndarray) -> Action:
        """
        Cycle must include mapping from. # Todo move upwards and put every custom logic in methods.
        :param time:
        :param observation_in:
        :return: optional neural activity
        """
        self.totalcycles += 1
        output = self.read_output(time)  # blocks until ready
        return self.output2actions(output)

    def give_reward(self, amount: float):
        self.log_m.append(amount)
        self.release_neurotransmitter(amount)

    @abstractmethod
    def release_neurotransmitter(self, amount: float):
        """insert reward into synapses"""

    @abstractmethod
    def get_weights(self) -> Weightstorage:
        """Get the parameters of the algorithm (e.g. connectome)"""

    def get_weight_log(self):
        """Get a list of the historic weights"""
        return self.weightlog

    @abstractmethod
    def read_output(self, time: float) -> list[float]:
        """returns a vector containing the sampled output signal per neuron
        :param time: Time when the output is read. Is only needed for filtered. (Spike to Analog)
        :rtype: list[float]"""
        pass

    def post_episode(self, episode):
        """
        store weights for loading in next episode and logging
       :return:
       """
        self.log_m.clear()

    def post_experiment(self):
        super().post_experiment()
        duration = (time.time_ns() - self.starttime) / 1_000_000
        print(f"{self.totalcycles} in {duration}ms={self.totalcycles / duration} cycles/ms")
