
#this fixes exp not beeing able to import because it is not in the pythonpath
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import exp
from agent import Agent
from exp import Experiment, trainingrun
from globalvalues import gv
import numpy as np

from critic import DynamicBaseline
from lineFollowingEnvironment2 import LineFollowingEnv2
from symbolicactor import SymbolicActor


def configure_training(expenv: Experiment):
    """error encoded in place cells"""
    print("Preparing line following with symbolic computation experiment.")
    gv.manual_num_inhibitory = 0
    gv.structural_plasticity = False  # might be switched to true later
    gv.random_reconnect = False
    gv.population_size = 1
    gv.errsig_factor = 0.001  # 0.03 unfiltered, 0.008 for utility
    gv.num_episodes = 1500
    gv.criticresolution = 40
    gv.dynamic_baseline = False
    #relative input with palce cells currently does not work
    relative_input = False #externalize computations of difference
    lfenv = LineFollowingEnv2(absolute_observation=not relative_input)
    if relative_input:
        placecell_range = np.array([[-lfenv.track_width * 1 / 3.0, lfenv.track_width * 1 / 3.0]])
        state_labels = ["Error from optimum"]
    else:
        placecell_range = np.array([[-lfenv.track_width * 1 / 3.0, lfenv.track_width * 1 / 3.0],
                                    [-lfenv.track_width * 1 / 3.0, lfenv.track_width * 1 / 3.0]])
        state_labels = ["Current Position", "Target Position"]
    num_neurons_per_dim = np.full(len(placecell_range), fill_value=2)

    expenv.env = lfenv
    expenv.env.seed(gv.seed)
    gv.w_max = 2

    expenv.env.seed(gv.seed)
    critic = DynamicBaseline(obsranges=placecell_range, state_labels=state_labels)
    expenv.agent = Agent(expenv.env,
                         actor=SymbolicActor(placecell_range=placecell_range, num_neurons_per_dim=num_neurons_per_dim,
                                             env=expenv.env),
                         critic=critic)
    expenv.agent.placell_nneurons_per_dim = num_neurons_per_dim
    expenv.penalty = 0.
    expenv.agent.actor.obsfactor = 400  # clamped if too big


if __name__ == "__main__":
    args = exp.parseargs()
    single, res = exp.trainingrun(configure_training, num_processes=args.processes, gridsearchpath=args.gridsearch)
    if single is not None:
        single.agent.critic.draw(xaxis=1, yaxis=0)
