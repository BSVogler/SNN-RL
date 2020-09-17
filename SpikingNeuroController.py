from typing import List, Union, Tuple
import numpy as np
from actor import Actor, PlaceCellAnalog2ActivationLayer, Weightstorage
from globalvalues import gv
try:
    import nest
except ImportError:
    print("Neural simulator Nest not found (import nest). Only able to run the simplified architecture.")


class SpikingNeurocontroller(Actor):
    def __init__(self, placecell_range: Union[np.ndarray, None], num_neurons_per_dim: Union[np.ndarray, int], env,
                 neuron_labels: List[str] = None):
        """

        :param placecell_range: if not using place cell encoding pass None
        :param num_neurons_per_dim: for each dimension number of place fields
        :param env:
        :param neuron_labels:
        """
        super().__init__()
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
        self.end_episode(-1)

    def _set_input(self, observation: np.ndarray):
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

    def cycle(self, time: float, observation_in: np.ndarray) -> Tuple[Union[int, List[float]], List]:
        """has side effects, call only once per cycle"""
        # feed observations into brain
        self._set_input(observation_in)
        nest.Simulate(gv.cycle_length)  # run does not work because between cycles parameters are set
        return self.get_action(time=time)

    def release_neurotransmitter(self, amount: float):
        super().release_neurotransmitter(amount)
        if len(self.connectome.conns_nest_in) >= 1:
            self.connectome.conns_nest_in.set({"n": -amount})
        # because of some strange bug where the first is a proxy node
        self.connectome.conns_nest_ex.set({"n": amount})

    def end_cycle(self, cycle_num):

        self.connectome.end_cycle(cycle_num)

    def end_episode(self, episode):
        super().end_episode(episode)
        weights = self.connectome.get_weights()
        self.weightlog.append(weights)
