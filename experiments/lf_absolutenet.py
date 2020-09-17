#this fixes exp not beeing able to import because it is not in the pythonpath
import os,sys,inspect


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from SpikingNeuroController import SpikingNeurocontroller
from critic import DynamicBaseline
from lineFollowingEnvironment2 import LineFollowingEnv2
import exp
from globalvalues import gv
import numpy as np
from agent import Agent

def configure_training(expenv):
    # manual minimal net with absolute input encoding as depictured in figure c)
    gv.num_hidden_neurons = 4
    gv.manual_num_inhibitory = 2
    gv.manualwiring = [
        (3, 6, True),
        (3, 9, True),
        (4, 7, True),
        (4, 8, True),
        (7, 9, False),
        (6, 8, False)
    ]
    gv.w0_min = 800.  # Minimum initial random value
    gv.w0_max = 800.  # Maximum initial random value
    gv.num_episodes = 200
    gv.structural_plasticity = False
    lfenv = LineFollowingEnv2(absolute_observation=True)
    lfenv.seed(gv.seed)
    placecell_range = np.array([[-lfenv.track_width * 1 / 3.0, lfenv.track_width * 1 / 3.0],
                                [-lfenv.track_width * 1 / 3.0, lfenv.track_width * 1 / 3.0]])
    critic = DynamicBaseline(obsranges=placecell_range)
    expenv.agent = Agent(expenv.env,
                         actor=SpikingNeurocontroller(placecell_range=None, num_neurons_per_dim=1, neuron_labels=["current Position", "next Position"], env=expenv.env),
                         critic=critic)
    expenv.penalty = -2.
    expenv.agent.actor.only_positive_input = True
    expenv.agent.actor.obsfactor = np.array([800 for x in range(2)])  # clamped anyway


if __name__ == "__main__":
    args = exp.parseargs()
    exp.trainingrun(configure_training, num_processes=args.processes, gridsearchpath=args.gridsearch)
