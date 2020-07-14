from typing import List, Tuple, Union, Dict

import nest

from lineFollowingEnvironment import LineFollowingEnv
from lineFollowingEnvironment2 import LineFollowingEnv2
from globalvalues import gv
import fremauxfilter
import numpy as np


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

        self.vq_decay = np.float128(1.0)  # of vector quantization

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
        self.vq_decay *= np.float128(1 - gv.vq_decay)  # exponentially decrease strength of vq
        changeamount = gv.vq_learning_scale * np.exp(-dist2 / self.vq_decay)
        self.positions += (observation - self.positions) * changeamount[:, np.newaxis]


Weightstorage = np.array


class Actor:
    """Part of the agent that manages behaviour. Includes analog-2-spike and spike-2-analog."""

    def __init__(self, placecell_range: Union[np.ndarray, None], num_neurons_per_dim: Union[np.ndarray, int], env,
                 neuron_labels: List[str] = None):
        """

        :param placecell_range: if not using place cell encoding pass None
        :param num_neurons_per_dim: for each dimension number of place fields
        :param env:
        :param neuron_labels:
        """
        self.env = env
        self.obsfactor = 1  # factors to scale observations, can be a numpy array of size of number of place cells
        # place cells
        num_place_cells: int = np.multiply.reduce(num_neurons_per_dim)
        self.placell_nneurons_per_dim = num_neurons_per_dim
        self.placecell_range: np.ndarray = placecell_range  # axis 0: dimension, axis 1: [from, to]
        if placecell_range is not None:
            labels = []
            for i in range(num_place_cells):
                pc_center = placecell_range[0, 0] + i * abs((placecell_range[0, 1] - placecell_range[0, 0])) / (
                        num_place_cells - 1)
                labels.append(f"pc_{pc_center}")
            self.placeCellA2SLayer: PlaceCellAnalog2ActivationLayer = PlaceCellAnalog2ActivationLayer(
                placecellrange=self.placecell_range,
                num_cells_per_dim=self.placell_nneurons_per_dim)

        from connectome import Connectome
        self.connectome = Connectome(actor=self, num_input=num_place_cells, neuron_labels=neuron_labels)
        # a matrix storing for each synapse (time, from, to) the history, +1 because 0 is initial
        self.weightlog: List[Weightstorage] = []
        self.only_positive_input = False  # if input is only positive

        # log
        self.log_m: List[float] = []  # log every cycle
        self.end_episode(-1)

    def set_input(self, observation: np.ndarray):
        """translate observation into rate code (Analog to Spike Rate)"""
        rate: np.ndarray
        # is using place cells?
        if self.placecell_range is not None:
            rate = self.placeCellA2SLayer.activation(
                observation=observation)
        else:
            # encode directly via rate
            if self.only_positive_input:
                rate: np.ndarray = observation
            else:
                # if it can also be negative split into two neurons
                rate = np.empty((observation.shape[0] * 2))
                for i, obs in enumerate(observation):
                    rate[i * 2] = -np.minimum(obs, 0)
                    rate[i * 2 + 1] = np.maximum(obs, 0)

        # logging
        # translate to scaled numpy array
        rate = rate * self.obsfactor
        rate = np.clip(rate, 0, gv.max_poisson_freq)
        self.connectome.set_inputactivation(rate)

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
        """has side effects, call only once per cycle"""
        # feed observations into brain
        self.set_input(observation_in)
        nest.Simulate(gv.cycle_length)  # run does not work because between cycles parameters are set
        return self.get_action(time=time)

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
        if len(self.connectome.conns_nest_in) >= 1:
            self.connectome.conns_nest_in.set({"n": -amount})
        # because of some strange bug where the first is a proxy node
        self.connectome.conns_nest_ex.set({"n": amount})
        self.log_m.append(amount)

    def end_cycle(self, cycle_num):
        self.connectome.end_cycle(cycle_num)

    def end_episode(self, episode):
        """
        store weights for loading in next episode and logging
       :return:
       """
        weights = self.connectome.get_weights()
        self.weightlog.append(weights)
        self.log_m.clear()
