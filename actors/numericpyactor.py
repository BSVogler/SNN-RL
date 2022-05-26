import time
from typing import Union, Optional

from actors.actor import Actor, PlaceCellAnalog2ActivationLayer, Weightstorage
from settings import gv
import numpy as np

from environments.lineFollowingEnvironment import LineFollowingEnv
from environments.lineFollowingEnvironment2 import LineFollowingEnv2


class NumericPyActor(Actor):
    """Part of the agent that manages behaviour. Includes analog-2-spike and spike-2-analog."""

    def __init__(self, placecell_range: np.ndarray, num_neurons_per_dim: np.ndarray, env,
                 neuron_labels: list[str] = None):
        """

        :param num_neurons_per_dim: for each dimension numebr of place fields
        :param env:
        :param neuron_labels:
        """
        super().__init__(env)
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
                placecell_range=placecell_range,
                num_cells_per_dim=num_neurons_per_dim)

        self.lastpc: int  # index
        # weights/actions which are associated to each place cell
        self.placecellaction = np.random.uniform(low=-0.5, high=.5,
                                                 size=self.placeCellA2SLayer.positions.shape[
                                                     0])  # default action is random
        # log
        self.lastactivation: np.array = np.array([])  # array, one hot encoded
        self.post_episode(-1)
        self.positive_input = False  # if input is only positive

    def get_weights(self) -> Weightstorage:
        return self.placecellaction.copy()

    def read_output(self, time: float) -> list[float]:
        return self.lastactivation * self.placecellaction

    def cycle(self, time: float, observation_in: np.ndarray) -> Union[int, list[float]]:
        """has side effects, call only once per ccoreloopycle"""
        # get nearest place cell
        self.totalcycles += 1
        activations = self.placeCellA2SLayer.activation(observation_in)
        # lateral inhibition causes one hot encoding
        self.lastactivation = np.zeros_like(activations)
        lastpc = np.argmax(activations)
        #print(f"numericmax: {lastpc}")
        self.lastactivation[lastpc] = 1

        # vector quantization
        # plt.scatter(self.placecellpos[:, 0], self.placecellpos[:, 1], c=self.placecellpos[:, 3])
        # plt.show()

        # calculate output layer
        output = self.read_output(time)
        return self.output2actions(output)

    def release_neurotransmitter(self, amount: float):
        """update last pc"""

        def g(weight: float) -> float:
            # eligibility trace after foderaro et al. 2010, not very beneficial
            return 0.2 + np.abs(weight * 5 * np.exp(-np.abs(weight) / gv.w_max))

        delta = np.sign(
            self.placecellaction * self.lastactivation) * amount
        self.placecellaction += delta  # * g(self.placecellaction) proved not very beneficial
        self.placecellaction = np.clip(self.placecellaction, -gv.w_max, gv.w_max)

    def post_episode(self, episode):
        """
        store weights for loading in next episode and logging
       :return:
       """
        super().post_episode(episode)
        self.weightlog.append(self.get_weights())
