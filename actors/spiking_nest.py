from typing import Union
import numpy as np

from agent import Trainable
from environments.lineFollowingEnvironment import LineFollowingEnv
from environments.lineFollowingEnvironment2 import LineFollowingEnv2

from actors import fremauxfilter
from actors.actor import Actor, PlaceCellAnalog2ActivationLayer, Weightstorage, Action
from settings import gv

try:
    import nest
except ImportError:
    print("Neural simulator Nest backend not found (import nest).")


class SpikingNest(Actor, Trainable):
    def __init__(self, placecell_range: Union[np.ndarray, None], num_neurons_per_dim: Union[np.ndarray, int], env,
                 neuron_labels: list[str] = None):
        """

        :param placecell_range: if not using place cell encoding pass None
        :param num_neurons_per_dim: for each dimension number of place fields
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
                placecell_range=self.placecell_range,
                num_cells_per_dim=self.placell_nneurons_per_dim)

        from actors.connectome import Connectome
        self.connectome: Connectome = Connectome(actor=self, num_input=num_place_cells, neuron_labels=neuron_labels)
        # a matrix storing for each synapse (time, from, to) the history, +1 because 0 is initial
        self.weightlog: list[Weightstorage] = []
        self.only_positive_input = False  # if input is only positive

        # log
        self.post_episode(-1)

    def get_weights(self) -> Weightstorage:
        return self.connectome.get_weights()

    def pre_episode(self):
        self.connectome.rebuild()

    def post_episode(self, episode):
        super().post_episode(episode)
        # todo move this to pre_episode?
        # only simulate if there is a connectome
        nest.Simulate(gv.cycle_length)
        weights = self.get_weights()
        self.weightlog.append(weights)

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

    def get_action(self, time: float) -> Action:
        """
        Get the action this actor decided to do
        :param time: the time at which the output is read
        :return: Tuple[activation of action for gym, neural output activations]
        """
        outputs: list[float] = self.read_output(time=time)
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
        return action

    def cycle(self, time: float, observation_in: np.ndarray) -> Action:
        """has side effects, call only once per cycle"""
        # feed observations into brain
        self.totalcycles += 1
        self._set_input(observation_in)
        nest.Simulate(gv.cycle_length)  # run does not work because between cycles parameters are set
        return self.get_action(time=time)

    def release_neurotransmitter(self, amount: float):
        if len(self.connectome.conns_nest_in) >= 1:
            self.connectome.conns_nest_in.set({"n": -amount})
        # because of some strange bug where the first is a proxy node
        self.connectome.conns_nest_ex.set({"n": amount})

    def post_cycle(self, cycle_num):
        self.connectome.post_cycle(cycle_num)

    def read_output(self, time: float) -> list[float]:
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
