import math

import gym

#this fixes exp not beeing able to import because it is not in the pythonpath
import os,sys,inspect


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import exp
from actors.spiking_nest import SpikingNest
from agent import Agent
from settings import gv
import numpy as np

from critic import DynamicBaseline


def configure_training(expenv: exp.Experiment):
    """error encoded in place cells"""
    print("Preparing polebalancing experiment.")
    gv.manual_num_inhibitory = 0
    gv.structural_plasticity = False  # might be swiched to true later
    gv.random_reconnect = False
    gv.population_size = 1
    #the reward factor is closely connected to the output size of the ciritic
    # values are so small, so we scale it to boost learning
    gv.errsig_factor = 0.003 #0.03 unfiltered, 0.008 for utility
    gv.num_episodes = 1500
    env = gym.make('CartPole-v1')
    env.seed(gv.seed)
    gv.w0_min = 300.  # Minimum initial random value
    gv.w0_max = 800.  # Maximum initial random value

    placecell_range = [[]] * 4
    # for cartpole
    # actual ranges areonly have the size
    placecell_range[0] = [-2.4, 2.4]  # Cart pos
    placecell_range[1] = [-3, 3]  # Cart Velocity   #usually upper limit is aroudn 1.9
    theta_threshold_radians = 12 * 2 * math.pi / 360
    placecell_range[2] = [-theta_threshold_radians, theta_threshold_radians]  # is in rad
    placecell_range[3] = [-4, 4]  # angular velocity

    placecell_range = np.array(placecell_range)

    if "num_cells" in gv.workerdata:
        num_neurons_per_dim = [np.array([3, 3, 5, 5]),
                               np.array([5, 5, 7, 7]),
                               np.array([7, 7, 15, 15])][int(gv.workerdata["num_cells"])]
    else:
        num_neurons_per_dim = np.array([5, 5, 7, 7])
    neuron_labels = ["Cart Pos. +", "Cart Pos. -", "Cart Vel. +", "Cart Vel. -", "Pole Angle +",
                     "Pole Angle -", "Pole Vel. +",
                     "Pole Vel. -"]
    expenv.env = env
    expenv.env.seed(gv.seed)
    critic = DynamicBaseline(obsranges=placecell_range)
    expenv.agent = Agent(expenv.env,
                         actor=SpikingNest(placecell_range=placecell_range, num_neurons_per_dim=num_neurons_per_dim, env=expenv.env),
                         critic=critic)
    expenv.agent.placell_nneurons_per_dim = num_neurons_per_dim
    expenv.penalty = -50.
    expenv.agent.actor.obsfactor = 400  # clamped if too big
    #expenv.agent.actor.connectome.load("5ebd31ed74bd5dfd5c40804e")

if __name__ == "__main__":
    args = exp.parseargs()
    exp.trainingrun(configure_training, num_processes=args.processes, gridsearchpath=args.gridsearch)
