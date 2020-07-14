from typing import List, Tuple, Union

from actor import Actor, PlaceCellAnalog2ActivationLayer
from globalvalues import gv
from connectome import Connectome
import numpy as np
import matplotlib.pyplot as plt

from lineFollowingEnvironment import LineFollowingEnv
from lineFollowingEnvironment2 import LineFollowingEnv2


class SymbolicActor(Actor):
    """Part of the agent that manages behaviour. Includes analog-2-spike and spike-2-analog."""

    def __init__(self, placecell_range: np.ndarray, num_neurons_per_dim: np.ndarray, env,
                 neuron_labels: List[str] = None):
        """

        :param num_neurons_per_dim: for each dimension numebr of place fields
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

        self.lastpc: int  # index
        self.placecellaction = np.random.uniform(low=-0.5, high=.5,
                                                    size=self.placeCellA2SLayer.positions.shape[0])  # default action is random
        # log
        self.lastactivation: np.array = np.array([])  # array, one hot encoded
        self.log_m: List[float] = []  # log every cycle
        self.weightlog: List[Actor.Weightstorage] = []
        self.end_episode(-1)
        self.positive_input = False  # if input is only positive

    def read_output(self, time: float) -> List[float]:
        output = self.placecellaction * self.lastactivation
        return output

    def cycle(self, time: float, observation_in: np.ndarray) -> Tuple[Union[int, List[float]], List]:
        """has side effects, call only once per cycle"""
        # get nearest place cell
        dists2: np.ndarray = np.linalg.norm(self.placeCellA2SLayer.positions - observation_in, axis=1) ** 2
        activations = self.placeCellA2SLayer.activation(observation_in)
        # lateral inhibition causes one hot encoding
        self.lastactivation = np.zeros_like(dists2)
        lastpc = np.argmax(activations)
        self.lastactivation[lastpc] = 1

        # vector quantiazion
        # plt.scatter(self.placecellpos[:, 0], self.placecellpos[:, 1], c=self.placecellpos[:, 3])
        # plt.show()

        #calculate output layer
        output = self.read_output(time)
        if isinstance(self.env, LineFollowingEnv) or isinstance(self.env, LineFollowingEnv2):
            action = [np.clip(0.5 - np.sum(output), 0., 1.)]
        else:
            # rate should only be a scalar value
            action: int = int(np.sign(np.sum(output)) == 1)  # 0 or 1
        return action, None

    def release_neurotransmitter(self, amount: float):
        """update last pc"""
        def g(weight: float) -> float:
            #eligibility trace after foderaro et al. 2010, not very beneficial
            return 0.2+np.abs(weight*5*np.exp(-np.abs(weight)/gv.w_max))

        self.placecellaction += np.sign(self.placecellaction*self.lastactivation) * amount #* g(self.placecellaction) proved not very beneficial
        self.placecellaction = np.clip(self.placecellaction, -gv.w_max, gv.w_max)

    def end_cycle(self, cycle_num):
        pass

    def end_episode(self, episode):
        """
        store weights for loading in next episode and logging
       :return:
       """
        self.weightlog.append(self.placecellaction.copy())
