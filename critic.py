import copy
import os
from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np
from typing import NamedTuple, Any
import matplotlib.pyplot as plt
from gym.spaces import Box
from sklearn.neighbors import KNeighborsRegressor

from abstractcritic import AbstractCritic, Rewards, BaselineEntry, State, hash_bucketed
from settings import gv


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

    def post_episode(self, episode):
        super().post_episode(episode)
        self.last_r = None
        self.last_util = None

        if gv.dynamic_baseline:
            # fit baseline predictor
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
            raise AssertionError("Trial not ended. Call post_trial after collecting data.")
        field = np.full((self.bucketsperdim, self.bucketsperdim), np.nan)
        # get reward values for each bucket
        for entry in self.recordings.values():
            # normalize then map to bucketsperdim
            coords = np.array(entry.state_index, dtype=np.int)
            # ignore recorded states outside observed range
            if 0 < coords[xaxis] < self.bucketsperdim and 0 < coords[yaxis] < self.bucketsperdim:
                field[coords[xaxis], coords[yaxis]] = np.average(entry.rewards)
        current_cmap = copy.copy(plt.cm.get_cmap("inferno"))
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
            raise AssertionError("Trial not ended. Call post_trial after collecting data.")
        field = np.full((self.bucketsperdim, self.bucketsperdim), np.nan)
        # get reward values for each bucket
        for entry in self.recordings.values():
            # normalize then map to bucketsperdim
            coords = np.array(entry.state_index, dtype=np.int)
            # ignore recorded states outside observed range
            if 0 < coords[xaxis] < self.bucketsperdim and 0 < coords[yaxis] < self.bucketsperdim:
                field[coords[xaxis], coords[yaxis]] = entry.utility
        current_cmap = copy.copy(plt.cm.get_cmap("inferno"))
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
        current_cmap = copy.copy(plt.cm.get_cmap("inferno"))
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
