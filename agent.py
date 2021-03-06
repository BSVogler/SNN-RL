from typing import List, Dict, Tuple
try:
    import nest
except ImportError:
    print("Neural simulator Nest not found (import nest). Only able to run the simplified architecture.")

from actor import Actor, Weightstorage
from critic import AbstractCritic
from globalvalues import gv


class Agent:
    def __init__(self, environment, actor: Actor, critic: AbstractCritic):
        self.actor = actor
        self.critic: AbstractCritic = critic
        self.environment = environment

    def get_action(self, time) -> List[float]:
        return self.actor.get_action(time)

    def end_cycle(self, cycle_num):
        self.actor.end_cycle(cycle_num)

    def end_episode(self, episode):
        self.critic.end_episode()
        self.actor.end_episode(episode)

    def prepare_episode(self):
        try:
            self.actor.connectome.rebuild()
        except AttributeError:
            pass

    def post_episode(self):
        try:
            #only simulate if there is a connectome
            self.actor.connectome
            nest.Simulate(gv.cycle_length)
        except AttributeError:
            pass


    def get_weights(self) -> Weightstorage:
            try:
                return self.actor.connectome.get_weights()
            except:
                return self.actor.placecellaction.copy()