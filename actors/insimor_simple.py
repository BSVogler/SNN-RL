import os
import sys
import ctypes
from typing import Optional

from numpy.ctypeslib import ndpointer
import numpy as np

from actors.actor import Actor, Action, Weightstorage, PlaceCellAnalog2ActivationLayer
from environments.lineFollowingEnvironment import LineFollowingEnv
from environments.lineFollowingEnvironment2 import LineFollowingEnv2


# define c types
# this is a facade to manage the insimor backend. it will continuously run in the background
def setup_insimor(num_place_cells) -> ctypes.CDLL:
    path = "../../libinsimou.dylib"  # go back because we are in the result path
    if os.path.isfile(path):
        try:
            insimor: ctypes.CDLL = ctypes.cdll.LoadLibrary(path)
        except OSError:
            print(f"Unable to load the system C library {path}")
            sys.exit()
    else:
        print(f"cannot find library {path}")
        sys.exit()
    # pointer to float to mark an array and the size
    insimor.setinput.argtypes = ctypes.POINTER(ctypes.c_double), ctypes.c_size_t
    insimor.setinput.restype = None
    insimor.setactivations.argtypes = ctypes.POINTER(ctypes.c_double), ctypes.c_size_t
    insimor.setactivations.restype = None
    insimor.give_reward.argtypes = [ctypes.c_double]
    insimor.give_reward.restype = None
    # todo replace with variant were the number of place cells is coming from insimor
    insimor.getWeights.restype = ndpointer(dtype=ctypes.c_double,
                                           shape=(num_place_cells,))  # use numpy array to access C array
    insimor.getAction.restype = ndpointer(dtype=ctypes.c_float,
                                          shape=(1,))  # use numpy array to access C array
    insimor.getOutputs.restype = ndpointer(dtype=ctypes.c_double,
                                           shape=(num_place_cells,))  # use numpy array to access C array
    insimor.printstats.restype = None
    insimor.stop.restype = None
    insimor.setWeights.argtypes = (ctypes.POINTER(ctypes.c_double),)
    insimor.setWeights.restype = None
    return insimor


class InsimorSimple(Actor):
    def __init__(self, placecell_range: np.ndarray, num_neurons_per_dim: np.ndarray, env,
                 neuron_labels: list[str] = None):
        super().__init__(env)
        self.env = env  # this is only needed to decide on the actions, todo
        # todo replace with C++ version in step 2
        num_place_cells: int = np.multiply.reduce(num_neurons_per_dim)
        if placecell_range is not None:
            labels = []
            for i in range(num_place_cells):
                pc_center = placecell_range[0, 0] + i * abs((placecell_range[0, 1] - placecell_range[0, 0])) / (
                        num_place_cells - 1)
                labels.append(f"pc_{pc_center}")
            self.placeCellA2SLayer: PlaceCellAnalog2ActivationLayer = PlaceCellAnalog2ActivationLayer(
                placecell_range=placecell_range,
                num_cells_per_dim=num_neurons_per_dim)
        self.insimor = setup_insimor(num_place_cells)
        # make deterministic, replace with c++ version when producing the same behavior
        DoubleArrayDin: type = ctypes.c_double * num_place_cells
        placecellactions = np.random.uniform(low=-0.5, high=.5,
                                             size=self.placeCellA2SLayer.positions.shape[
                                                 0])
        self.insimor.setWeights(DoubleArrayDin(*placecellactions.tolist()))
        self.insimor.start_sync()  # blocks until spawned

    def get_weights(self) -> Weightstorage:
        return self.insimor.getWeights()

    def read_output(self, time: float) -> list[float]:
        # this shoud return not the action but the output
        return self.insimor.getOutputs().tolist()

    def cycle(self, time: float, observation_in: np.ndarray) -> Action:
        self.totalcycles += 1

        # todo replace with C++ version in step 2
        activations = self.placeCellA2SLayer.activation(observation_in)

        # FloatArrayDin: type = ctypes.c_double * len(observation_in)  # Define a 4-length array of floats
        # parameter_array = FloatArrayDin(*observation_in.tolist())  # Define the actual array to pass to your C function
        # self.insimor.setinput(parameter_array, len(parameter_array))
        DoubleArrayDin: type = ctypes.c_double * len(activations)  # Define a 4-length array of floats
        parameter_array = DoubleArrayDin(*activations.tolist())  # Define the actual array to pass to your C function
        self.insimor.setactivations(parameter_array, len(parameter_array))

        # get from insimor
        output = self.read_output(time)
        return self.output2actions(output)

    def release_neurotransmitter(self, amount: float):
        self.insimor.give_reward(amount)

    def post_episode(self, episode):
        super().post_episode(episode)
        self.weightlog.append(self.get_weights())

    def post_experiment(self):
        super().post_experiment()
        self.insimor.stop()
        self.insimor.printstats()
