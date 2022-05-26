from typing import Protocol


class Trainable(Protocol):
    def post_cycle(self, cycle_num):
        """Code run after a cycle"""

    def pre_episode(self) -> None:
        """Code which is needed to run before the episode is started."""

    def post_episode(self, episode) -> None:
        """Code which is needed to run after the episode is done."""

    def post_experiment(self):
        """Code after trainign is done."""