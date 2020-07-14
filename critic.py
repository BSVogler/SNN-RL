import os
from abc import ABCMeta, abstractmethod

import numpy as np
from typing import List, Dict, NamedTuple, Any, Type, Tuple, Union
import matplotlib.pyplot as plt
from gym.spaces import Box
from sklearn.neighbors import KNeighborsRegressor
from dataclasses import dataclass

from globalvalues import gv

State = Tuple[float, ...]
BucketIndex = Tuple[int, ...]
Rewards = Union[float, Tuple[float, ...]]


def hash_bucketed(bucket: BucketIndex) -> int:
    """get the number of the bucket"""
    return hash(bucket)


@dataclass
class RewardEntry:
    """Used for recording teh values to a percept."""
    state_index: BucketIndex
    rewards: List[Rewards]  # recordings of previous rewards


class AbstractCritic(metaclass=ABCMeta):
    """
    Contains all the methods needed to implement a critic.
    """

    def __init__(self, obsranges: np.ndarray, recordingtype: Type = RewardEntry):
        self.tickentries: List[BaselineEntry.TicksEpisode] = []  # stores rewards in this trial
        self.entrytype = recordingtype
        self.recordings: Dict[int, recordingtype] = dict()  # maps hash to rewards and utility
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

    def bucket_index_floating(self, state: State) -> Tuple[float, ...]:
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
            raise AssertionError("Trial not ended. Call end_episode after collecting data.")

        return self.query_bucket(state=state, bucket=self.bucketandhash(state))

    # @final
    def end_episode(self):
        """Called at the end of a trial"""
        self._end_episode()

    def valueiterate(self, trajectory):
        entry = self.recordings[trajectory[-1].bucket]
        if entry.utility is not None:
            reward = np.average(entry.rewards)
            entry.utility += gv.util_learn_rate * (reward - entry.utility)  # exp. moving average
        prev_util = entry.utility  # the very first will be None
        if prev_util is None:
            entry.utility = np.average(entry.rewards)
            prev_util = entry.utility
        #propagate backwards but skip the first
        for (bucket, rewards) in reversed(trajectory[:-1]):
            entry = self.recordings[bucket]
            entry_util = 0 if entry.utility is None else entry.utility
            td_error = gv.util_learn_rate * (np.average(rewards) + gv.util_discount_factor * prev_util - entry_util)
            entry.utility = entry_util + td_error
            prev_util = entry.utility

        trajectory.clear()

    def _end_episode(self):
        """Overwrite to specifiy what should be performed when a trial ended."""
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


@dataclass
class BaselineEntry(RewardEntry):
    utility: float
    baseline: float


class DynamicBaseline(AbstractCritic):
    """Implements critic as in wunderlich et al. 2019 """
    TicksEpisode = NamedTuple("trialentry", [("bucket", int), ("rewards", Rewards)])

    def __init__(self, obsranges: np.ndarray, state_labels=None):
        super().__init__(obsranges, recordingtype=BaselineEntry)
        if state_labels is None:
            # todo should be generic "observation 1"
            state_labels = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Velocity At Tip"]
        self.state_labels = state_labels
        self.bias = 0.0  # bias value is added each tick
        self.last_r = None
        self.last_util = 0
        self.numbaselines = 0
        self.knn_baseline = KNeighborsRegressor(n_neighbors=2, weights="distance")

    learning_rate = 0.9  # for baseline

    def _end_episode(self):
        super()._end_episode()
        self.last_r = None
        self.last_util = None

        if gv.dynamic_baseline:
            #fit baseline predictor
            states = []
            baselines = []
            self.numbaselines = 0
            for rec in self.recordings.values():
                if rec.baseline is not None:
                    states.append(rec.state_index)
                    baselines.append(rec.baseline)
                    self.numbaselines += 1
            self.knn_baseline.fit(states, baselines)

    def _tick(self, state: State, new_rewards: Rewards) -> (float, float):
        """

        :param state:
        :param new_rewards:
        :return:
        """
        bucketed_state = self.bucket_index(state)
        bucket = hash_bucketed(bucketed_state)
        new_reward_reduced = np.average(new_rewards)  # reduce reward to a float
        self.tickentries.append(DynamicBaseline.TicksEpisode(bucket=bucket, rewards=new_rewards))

        # check if can retrieve utility
        entry = self.query_bucket(state, bucket, bucketed=bucketed_state)
        # save if new
        self.recordings[bucket] = entry
        # add new reward to recordings
        entry.rewards.append(new_rewards)

        # the utility of this state is the current reward plus future rewards
        util: float = new_reward_reduced + entry.utility
        # if it can compute delta_r/u
        if self.last_r is not None:
            # try returning the delta in utility for R-STDP, in the beginning return 0
            errsignal = util - self.last_util
            if gv.dynamic_baseline:
                if entry.baseline is None:
                    # init with r0
                    if self.numbaselines >= self.knn_baseline.n_neighbors:
                        entry.baseline = self.knn_baseline.predict([self.bucket_index_floating(state)])[0]
                    else:
                        entry.baseline = errsignal  # δu because we don't want to introduce a bias
                else:
                    # exp weighted update
                    entry.baseline += DynamicBaseline.learning_rate * errsignal
                # subtract dynamic baseline
                errsignal -= entry.baseline
            errsignal = np.clip(errsignal, -self.maxm, self.maxm) + self.bias
        else:
            errsignal = 0
        self.last_r = new_reward_reduced  # to compute delta r in next step
        self.last_util = util
        return errsignal, entry.utility

    def draw_rewards(self, xaxis=3, yaxis=2, show=True):
        """
        Draw recorded reward map
        """
        if not self.completedATrial:
            raise AssertionError("Trial not ended. Call end_trial after collecting data.")
        field = np.full((self.bucketsperdim, self.bucketsperdim), np.nan)
        # get reward values for each bucket
        for entry in self.recordings.values():
            # normalize then map to bucketsperdim
            coords = np.array(entry.state_index, dtype=np.int)
            # ignore recorded states outside observed range
            if 0 < coords[xaxis] < self.bucketsperdim and 0 < coords[yaxis] < self.bucketsperdim:
                field[coords[xaxis], coords[yaxis]] = np.average(entry.rewards)
        current_cmap = plt.cm.get_cmap("inferno")
        current_cmap.set_bad(color='green')
        heatmap = plt.imshow(field, cmap=current_cmap, interpolation='nearest')
        plt.title("Reward for States")
        steps = 8
        labels_x = [f"{x:.1f}" for x in
                    np.linspace(self.state_limits.T[0, xaxis], self.state_limits.T[1, xaxis], num=steps)]
        labels_y = [f"{y:.1f}" for y in
                    np.linspace(self.state_limits.T[0, yaxis], self.state_limits.T[1, yaxis], num=steps)]
        plt.xticks(np.linspace(0, self.bucketsperdim, num=steps), labels_x)
        plt.yticks(np.linspace(0, self.bucketsperdim, num=steps), labels_y)
        plt.xlabel(self.state_labels[xaxis])
        plt.ylabel(self.state_labels[yaxis])
        if show:
            plt.show()
        return heatmap

    def draw_utility(self, xaxis=3, yaxis=2, show=True):
        """
        Draw recorded reward map
        """
        if not self.completedATrial:
            raise AssertionError("Trial not ended. Call end_trial after collecting data.")
        field = np.full((self.bucketsperdim, self.bucketsperdim), np.nan)
        # get reward values for each bucket
        for entry in self.recordings.values():
            # normalize then map to bucketsperdim
            coords = np.array(entry.state_index, dtype=np.int)
            # ignore recorded states outside observed range
            if 0 < coords[xaxis] < self.bucketsperdim and 0 < coords[yaxis] < self.bucketsperdim:
                field[coords[xaxis], coords[yaxis]] = entry.utility
        current_cmap = plt.cm.get_cmap("inferno")
        current_cmap.set_bad(color='green')
        heatmap = plt.imshow(field, cmap=current_cmap, interpolation='nearest')
        plt.title("Value for Visited States")

        steps = 8
        labels_x = [f"{x:.1f}" for x in
                    np.linspace(self.state_limits.T[0, xaxis], self.state_limits.T[1, xaxis], num=steps)]
        labels_y = [f"{y:.1f}" for y in
                    np.linspace(self.state_limits.T[0, yaxis], self.state_limits.T[1, yaxis], num=steps)]
        plt.xticks(np.linspace(0, self.bucketsperdim, num=steps), labels_x)
        plt.yticks(np.linspace(0, self.bucketsperdim, num=steps), labels_y)
        plt.xlabel(self.state_labels[xaxis])
        plt.ylabel(self.state_labels[yaxis])
        if show:
            plt.show()
        return heatmap

    def draw_utility_inferred(self, xaxis=3, yaxis=2, show=True, legend=None):
        """
        Draw recorded reward map
        """
        # interpolate with querying when available, else interpolate image
        if self.state_limits.shape[0] == 2:
            # get reward values for each bucket, but limit resolution
            res = np.minimum(self.bucketsperdim, 100)
            field = np.empty((res, res))
            range_x = np.linspace(self.state_limits.T[0, xaxis], self.state_limits.T[1, xaxis], res)
            range_y = np.linspace(self.state_limits.T[0, yaxis], self.state_limits.T[1, yaxis], res)
            # loop over the states in the graph
            # only two dimension can be iterated and visualized. Therefore,
            # use a state to pick a prototype which can be visualized by modifieng only the otehr states
            bestrecording = next(iter(self.recordings.values()))
            # find best state in O(n)
            for state in self.recordings.values():
                if state.utility > bestrecording.utility:
                    bestrecording = state
            prototypestate = list(bestrecording.state_index)  # create a mutable copy
            for x, stateX in enumerate(range_x):
                for y, stateY in enumerate(range_y):
                    prototypestate[xaxis] = stateX
                    prototypestate[yaxis] = stateY
                    field[x, y] = self.query_bucket(state=tuple(prototypestate)).utility
        else:
            res = self.bucketsperdim
            interpollist = []
            for entry in self.recordings.values():
                # normalize then map to bucketsperdim
                coords = np.array(entry.state_index, dtype=np.int)
                # ignore recorded states outside observed range
                if 0 < coords[xaxis] < self.bucketsperdim and 0 < coords[yaxis] < self.bucketsperdim:
                    interpollist.append((coords[xaxis], coords[yaxis], entry.utility))
            interpollist = np.array(interpollist)
            from scipy.interpolate import griddata
            xi = np.arange(0, self.bucketsperdim)
            yi = np.arange(0, self.bucketsperdim)
            field = griddata(interpollist[:, 0:2], interpollist[:, 2], (xi[None, :], yi[:, None]), method='nearest').T
        current_cmap = plt.cm.get_cmap("inferno")
        current_cmap.set_bad(color='green')
        plt.imshow(field, cmap=current_cmap, interpolation='nearest')
        if legend is not None:
            plt.clim(legend.norm.vmin, legend.norm.vmax)

        plt.title("Interpolated State-Values for every State")

        steps = 8
        labels_x = [f"{x:.1f}" for x in
                    np.linspace(self.state_limits.T[0, xaxis], self.state_limits.T[1, xaxis], num=steps)]
        labels_y = [f"{y:.1f}" for y in
                    np.linspace(self.state_limits.T[0, yaxis], self.state_limits.T[1, yaxis], num=steps)]
        plt.xticks(np.linspace(0, res, num=steps), labels_x)
        plt.yticks(np.linspace(0, res, num=steps), labels_y)
        plt.xlabel(self.state_labels[xaxis])
        plt.ylabel(self.state_labels[yaxis])
        if show:
            plt.show()

    def draw(self, xaxis=3, yaxis=2):
        """Draws plots showing different state value function mappings.."""
        if len(list(self.recordings.values())[0].state_index) < 2:
            print("Cannot draw state maps with only one input dimension.")
            return
        fig, axis = plt.subplots(1, 4, figsize=(18, 4))
        fig.tight_layout()
        # fig.suptitle("State Maps")
        plt.subplot(141)
        self.draw_rewards(xaxis=xaxis, yaxis=yaxis, show=False)
        plt.subplot(142)
        legend = self.draw_utility(xaxis=xaxis, yaxis=yaxis, show=False)
        plt.subplot(143)
        self.draw_utility_inferred(xaxis, yaxis, show=False, legend=legend)
        ax3 = plt.subplot(144)
        ax3.axis('off')
        # might be improved with this: https://stackoverflow.com/a/38940369/2768715
        plt.colorbar(legend)
        # find free filename
        counter = 0
        filename = f"utility{counter}.pdf"
        while os.path.isfile(filename):
            counter += 1
            filename = f"utility{counter}.pdf"
        plt.savefig(filename)


AbstractCritic.register(DynamicBaseline)
