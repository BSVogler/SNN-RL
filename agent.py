from actors.trainable import Trainable

try:
    import nest
except ImportError:
    print("Neural simulator Nest backend not found (import nest).")

from critic import AbstractCritic


class Agent(Trainable):
    """Here an agent implements the actor-critic RL pattern."""
    def __init__(self, environment, actor: 'Actor', critic: AbstractCritic):
        """Dependency injection constructor."""
        self.actor = actor
        self.critic: AbstractCritic = critic
        self.environment = environment

    def post_cycle(self, cycle_num):
        self.actor.post_cycle(cycle_num)

    def post_episode(self, episode):
        self.critic.post_episode(episode)
        self.actor.post_episode(episode)#

    def post_experiment(self):
        self.critic.post_experiment()
        self.actor.post_experiment()

