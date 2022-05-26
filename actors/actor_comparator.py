from typing import Optional, Type

import numpy as np

from actors.actor import Actor, Action, Weightstorage
from settings import gv


class Comparator(Actor):
    """A wrapper class to compare two actors behaviour."""

    def __init__(self, class1: Type, class2: Type, **kwargs):
        """class1 is used for returns"""
        super().__init__(environment=None)
        np.random.seed(gv.seed)
        self.instance1: class1 = class1(**kwargs)
        np.random.seed(gv.seed)
        self.instance2: class2 = class2(**kwargs)

    def get_weights(self) -> Weightstorage:
        cl1return = self.instance1.get_weights()
        cl2return = self.instance2.get_weights()
        for i in range(len(cl1return)):
            if cl1return[i] != cl2return[i]:
                #print(f"Difference in weight {i}.")
                break
        return cl1return

    def read_output(self, time: float) -> list[float]:
        cl1return = self.instance1.read_output(time)
        cl2return = self.instance2.read_output(time)
        for i in range(len(cl1return)):
            if cl1return[i] != cl2return[i]:
                #print("Difference in output.")
                pass
        return cl1return

    def cycle(self, time: float, observation_in: np.ndarray) -> tuple[Action, Optional[list]]:
        self.totalcycles += 1
        cl1return = self.instance1.cycle(time, observation_in)
        cl2return = self.instance2.cycle(time, observation_in)
        print("cycle")
        out1 = self.instance1.read_output(0)
        out2 = self.instance2.read_output(0)
        for i in range(len(out1)):
            if out1[i] != out2[i]:
                #print("Difference in output.")
                pass
        if cl1return != cl2return:
            #print("Difference in action.")
            pass
        return cl1return

    def release_neurotransmitter(self, amount: float):
        self.instance1.release_neurotransmitter(amount)

    def post_episode(self, episode):
        self.instance1.post_episode(episode)

    def post_experiment(self):
        self.instance1.post_experiment()
