#this fixes exp not beeing able to import because it is not in the pythonpath
import os,sys,inspect


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import exp
from SpikingNeuroController import SpikingNeurocontroller
from agent import Agent
from exp import Experiment
from globalvalues import gv
import numpy as np

from critic import DynamicBaseline
from lineFollowingEnvironment2 import LineFollowingEnv2


def configure_training(expenv: Experiment):
    """error encoded in place cells"""
    print("Preparing placenet experiment.")
    gv.manual_num_inhibitory = 0
    gv.structural_plasticity = False  # might be swiched to true later
    gv.random_reconnect = False
    gv.population_size = 1
    gv.num_episodes = 300
    gv.criticresolution = 40
    gv.dynamic_baseline = False
    gv.vq_learning_scale = 0
    #the reward factor is closely connected to the output size of the ciritic
    # values are so small, so we scale it to boost learning
    gv.errsig_factor = 0.002 #0.03 unfiltered, 0.008 for utility
    gv.w0_min = 500.  # Minimum initial random value
    gv.w0_max = 800.  # Maximum initial random value
    #relative input with place cell encoding currently does not work
    relative_input = False #externalize computations of difference
    lfenv = LineFollowingEnv2(absolute_observation=not relative_input)
    if relative_input:
        placecell_range = np.array([[-lfenv.track_width * 1 / 3.0, lfenv.track_width * 1 / 3.0]])
    else:
        placecell_range = np.array([[-lfenv.track_width * 1 / 3.0, lfenv.track_width * 1 / 3.0],
                                    [-lfenv.track_width * 1 / 3.0, lfenv.track_width * 1 / 3.0]])
    num_neurons_per_dim = np.full(len(placecell_range), fill_value=2)

    expenv.env = lfenv
    expenv.env.seed(gv.seed)
    critic = DynamicBaseline(obsranges=placecell_range)
    expenv.agent = Agent(expenv.env,
                         actor=SpikingNeurocontroller(placecell_range=placecell_range, num_neurons_per_dim=num_neurons_per_dim, env=expenv.env),
                         critic=critic)
    expenv.agent.placell_nneurons_per_dim = num_neurons_per_dim
    expenv.penalty = 0.
    expenv.agent.actor.obsfactor = 400  # clamped if too big

if __name__ == "__main__":
    args = exp.parseargs()
    single, res = exp.trainingrun(configure_training, num_processes=args.processes, gridsearchpath=args.gridsearch)
    if single is not None:
        single.drawspikes()
