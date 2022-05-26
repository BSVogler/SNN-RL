from abc import ABCMeta, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Union, Type, Any
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from actors.trainable import Trainable
from settings import gv

State = tuple[float, ...]
BucketIndex = tuple[int, ...]
Rewards = Union[float, tuple[float, ...]]


def hash_bucketed(bucket: BucketIndex) -> int:
    """get the number of the bucket"""
    return hash(bucket)


@dataclass
class RewardEntry:
    """Used for recording teh values to a percept."""
    state_index: BucketIndex
    rewards: list[Rewards]  # recordings of previous rewards


@dataclass
class BaselineEntry(RewardEntry):
    utility: float
    baseline: float


class AbstractCritic(Trainable, metaclass=ABCMeta):
    """
    Contains all the methods needed to implement a critic.
    """

    def __init__(self, obsranges: np.ndarray, recordingtype: Type = RewardEntry):
        self.tickentries: list[BaselineEntry.TicksEpisode] = []  # stores rewards in this trial
        self.entrytype = recordingtype
        self.recordings: dict[int, recordingtype] = dict()  # maps hash to rewards and utility
        self.bucketsperdim = gv.criticresolution  # good default value found by simulations on simple net
        if gv.workerdata is not None and "rescritic" in gv.workerdata:
            self.bucketsperdim = gv.workerdata["rescritic"]
        self.completedATrial = False
        self.state_limits = obsranges  # the ranges where the states are expected
        self.bucketing_steps: np.ndarray = (self.state_limits[:, 1] - self.state_limits[:, 0]) / self.bucketsperdim
        self.draw_limits: np.ndarray = self.state_limits.T  # the state limts which are drawn, first row is minimum, second is maximum
        self.knn_util = KNeighborsRegressor(n_neighbors=2, weights="distance")
        self.displayrange = None
        self.maxm: float = 6  # best default value found by simulations on simple net

    # @final #final is not supported in py37
    def tick(self, state: State, new_rewards: Rewards) -> (float, float):
        """
        input the states and rewards for a frame and get the utility back. Returns a single reward value based on teh new rewards. Later should return utility.
        :param state:
        :param new_rewards: the rewards for this state
        :returns reward signal and absolute utility (rating of this state)
        """
        errsignal, util = self._tick(state, new_rewards)

        if errsignal < 0:
            errsignal *= gv.factor_negative_util
        return errsignal, util

    @abstractmethod
    def _tick(self, state: State, new_rewards: Rewards) -> (float, float):
        """
        Intenral tick function. Needs to be overwritten by implementation.
        :param new_rewards:
        :return:
        """
        pass

    def bucketandhash(self, state: State) -> int:
        """get the hash numebr of the bucket"""
        return hash_bucketed(self.bucket_index(state))

    def bucket_index(self, state: State) -> BucketIndex:
        """get the index when put it into a bucket. Can be easily used to get the bucketed state."""
        # floordiv
        return tuple((v - self.state_limits[dim, 0]) // self.bucketing_steps[dim] for dim, v in enumerate(state))

    def bucket_index_floating(self, state: State) -> tuple[float, ...]:
        """get the index when put it into a bucket. Can be easily used to get the bucketed state."""
        # floordiv
        return tuple((v - self.state_limits[dim, 0]) / self.bucketing_steps[dim] for dim, v in enumerate(state))

    def query_bucket(self, state: State, bucket: int = 0, bucketed=None) -> Any:
        """return the bucket content for a already bucketed state"""
        if bucketed is None:
            bucketed = self.bucket_index(state)
        if bucket == 0:
            bucket = hash_bucketed(bucketed)

        if self.completedATrial < self.knn_util.n_neighbors:
            return self.entrytype(bucketed, [], .0, None)
        elif bucket in self.recordings:
            return self.recordings[bucket]
        else:  # inference
            return self.entrytype(bucketed, [], self.knn_util.predict([self.bucket_index_floating(state)])[0], None)

    def query(self, state: State) -> Any:
        """Get the utility for any unbucketed state"""
        if not self.completedATrial:
            raise AssertionError("Trial not ended. Call post_episode after collecting data.")

        return self.query_bucket(state=state, bucket=self.bucketandhash(state))

    # @final
    def post_episode(self, episode):
        """Called at the end of a trial"""
        """Overwrite to specify what should be performed when a trial ended."""
        # implements value iteration but assumes that current policy is optimal policy
        self.valueiterate(self.tickentries)

        # fit utility predictor
        self.completedATrial = len(self.recordings.values())
        states_index = []
        utility = []
        for rec in self.recordings.values():
            states_index.append(rec.state_index)
            utility.append(rec.utility)
        self.knn_util.fit(states_index, utility)

    def valueiterate(self, trajectory: list[namedtuple]):
        # todo trajectory is empty after first postepisode and will crash
        entry = self.recordings[trajectory[-1].bucket]
        if entry.utility is not None:
            reward = np.average(entry.rewards)
            entry.utility += gv.util_learn_rate * (reward - entry.utility)  # exp. moving average
        prev_util = entry.utility  # the very first will be None
        if prev_util is None:
            entry.utility = np.average(entry.rewards)
            prev_util = entry.utility
        # propagate backwards but skip the first
        for (bucket, rewards) in reversed(trajectory[:-1]):
            entry = self.recordings[bucket]
            entry_util = 0 if entry.utility is None else entry.utility
            td_error = gv.util_learn_rate * (np.average(rewards) + gv.util_discount_factor * prev_util - entry_util)
            entry.utility = entry_util + td_error
            prev_util = entry.utility

        trajectory.clear()
