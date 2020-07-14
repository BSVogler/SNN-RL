#this fixes exp not beeing able to import because it is not in the pythonpath
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import exp
from agent import Agent
from actor import Actor
from globalvalues import gv
import numpy as np

from critic import DynamicBaseline
from lineFollowingEnvironment2 import LineFollowingEnv2


def configure_training(expenv):
    # manual minimal net with relative input encoding as depictured in figure d)
    gv.num_hidden_neurons = 0
    gv.manual_num_inhibitory = 0
    gv.manualwiring = [
        (0, 3, True),
        (1, 2, True),
    ]
    gv.structural_plasticity = False
    gv.w0_min = 500.  # Minimum initial random value
    gv.w0_max = 800.  # Maximum initial random value
    gv.num_episodes = 40
    gv.errsig_factor = 0.003
    lfenv = LineFollowingEnv2(absolute_observation=False)

    expenv.env = lfenv
    expenv.env.seed(gv.seed)
    # the error can not get bigger than a third of the width
    env_range = np.array([[-lfenv.track_width * 1 / 3.0, lfenv.track_width * 1 / 3.0]])
    critic = DynamicBaseline(obsranges=env_range)
    expenv.agent = Agent(expenv.env,
                         actor=Actor(placecell_range=None, num_neurons_per_dim=2, env=expenv.env, neuron_labels=["neg", "pos"]),
                         critic=critic)
    expenv.agent.actor.only_positive_input = False
    expenv.agent.actor.obsfactor = np.array([400 for x in range(2)])  # clamped anyway


if __name__ == "__main__":
    args = exp.parseargs()
    exp.trainingrun(configure_training, num_processes=args.processes, gridsearchpath=args.gridsearch)
