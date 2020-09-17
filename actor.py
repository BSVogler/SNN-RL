from typing import List, Tuple, Union

from lineFollowingEnvironment import LineFollowingEnv
from lineFollowingEnvironment2 import LineFollowingEnv2
from globalvalues import gv
import fremauxfilter
import numpy as np
import sys

class PlaceCellAnalog2ActivationLayer:
    positions: np.ndarray = None  # list of center positions
    distance_pc: np.ndarray  # can be cached without vector quantization

    def __init__(self, placecellrange: np.ndarray, num_cells_per_dim: np.ndarray):
        """

        :param placecellrange: min max for each dimension
        :param num_cells_per_dim:
        """
        # distance between centers covered by the pc per dimension
        self.distance_pc = np.abs((placecellrange[:, 1] - placecellrange[:, 0])) / (num_cells_per_dim - 1)
        numdims = placecellrange.shape[0]
        # for each dimension create a grid, then transform to have all coordinates
        pos = np.mgrid[tuple(slice(0., num_cells_per_dim[dim]) for dim in range(numdims))].T.reshape(-1,
                                                                                                     numdims)  # shape: (num_cells_per_dim^numdims, num_cells_per_dim)
        self.positions = pos * self.distance_pc
        self.positions += placecellrange[:, 0]  # add offset to each element

        if sys.maxsize > 2**32:#check if 64bit
            self.vq_decay = np.float128(1.0)  # of vector quantization
        else:
            self.vq_decay = np.float64(1.0)  # of vector quantization

        self.sigmafactor = 1
        if gv.workerdata is not None and "receptivefieldsize" in gv.workerdata:
            self.sigmafactor = gv.workerdata["receptivefieldsize"]

    def activation(self, observation: np.ndarray) -> np.ndarray:
        """
        calculate activation for each neuron with exponential kernel like in fremaux2013 (eq. 22)
        When vector quantization is activated this function call has side-effects as it moves the positions.
        :param observation:
        :return: activation levels
        """

        if gv.vq_learning_scale > 0:
            # changes every time, so cannot be cached
            # rezsigma = (len(dists) / np.sum(dists)
            k_averagedistancing = False
            if k_averagedistancing:
                rezsigma = np.empty_like(self.positions)
                k = 4
                for i in range(len(rezsigma)):
                    # distance to other place cells
                    dist = np.linalg.norm(self.positions[i] - self.positions, axis=1)
                    rezsigma[i] = np.average(dist[np.argpartition(dist, 1 + k)[1:1 + k]])  # ignore self
            else:
                rezsigma = self.sigmafactor / self.distance_pc
            dists: np.ndarray = np.linalg.norm((self.positions - observation) * rezsigma, axis=1)
            dists2: np.ndarray = dists ** 2
            input_activation = dists2  # use average distance
            self.vector_quantization(observation, dists2)
        else:
            # use lp2 norm, weighted by dimensionality density
            scaleddistance = (self.positions - observation) / self.distance_pc  # calculation per dimension
            input_activation = np.linalg.norm(scaleddistance, axis=1) ** 2
        vec = np.exp(-input_activation)
        return vec / np.sum(vec)  # normalize

    def vector_quantization(self, observation: np.ndarray, dist2: np.ndarray):
        if sys.maxsize > 2 ** 32:  # check if 64bit
            self.vq_decay *= np.float128(1 - gv.vq_decay)  # exponentially decrease strength of vq
        else:
            self.vq_decay *= np.float64(1 - gv.vq_decay)  # exponentially decrease strength of vq
        changeamount = gv.vq_learning_scale * np.exp(-dist2 / self.vq_decay)
        self.positions += (observation - self.positions) * changeamount[:, np.newaxis]


Weightstorage = np.array


class Actor:
    """Part of the agent that manages behaviour. Includes analog-2-spike and spike-2-analog."""

    def __init__(self):
        self.log_m: List[float] = []  # log every cycle

    def read_output(self, time: float) -> List[float]:
        """returns a vector containing the sampled output signal per neuron
         time: Time when the output is read. Is only needed for filtered. (Spike to Analog)"""

        # temporal average by filtering
        # for each neuron there is a signal
        spike_signals = self.connectome.get_outspikes()
        if gv.filter_outsignals:
            # plot spikes
            return fremauxfilter.filter(time, spike_signals)
        else:
            # just count spikes in cycle
            return list(map(lambda x: len(x), spike_signals))

    def cycle(self, time: float, observation_in: np.ndarray) -> Tuple[Union[int, List[float]], List]:
        pass

    def get_action(self, time: float) -> Tuple[Union[int, List[float]], List]:
        """
        Get the action this actor decided to do
        :param time: the time at which the output is read
        :return: Tuple[activation of action for gym, neural output activations]
        """
        outputs = self.read_output(time=time)
        left = outputs[0]
        right = outputs[1]

        if isinstance(self.env, LineFollowingEnv) or isinstance(self.env, LineFollowingEnv2):
            sensitivitiy = 1 / 400.0
            if gv.filter_outsignals:
                sensitivitiy *= 51.081  # empirical value from one episoded with least squares

            # in line following the action 0.5 means to go straight,
            action = [np.clip(0.5 - (left - right) * sensitivitiy, 0., 1.)]
        else:
            action = 0 if left >= right else 1
        return action, outputs

    def release_neurotransmitter(self, amount: float):
        """insert reward into nest synapses"""
        self.log_m.append(amount)

    def end_cycle(self, cycle_num):
        pass

    def end_episode(self, episode):
        """
        store weights for loading in next episode and logging
       :return:
       """
        self.log_m.clear()
