#this fixes exp not beeing able to import because it is not in the pythonpath
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import exp
from actor import Actor
from globalvalues import gv
import numpy as np

def configure_training(expenv):
    # fully connected net with relative input encoding
    gv.num_hidden_neurons = 2
    gv.manual_num_inhibitory = 0
    gv.w0_min = 200.  # Minimum initial random value
    gv.w0_max = 800.  # Maximum initial random value
    gv.num_episodes = 10
    gv.errsig_factor = 0.03
    gv.render = False
    gv.structural_plasticity = False
    gv.random_reconnect = True
    expenv.createLineFollowing(Actor(num_inputneurons=2, neuron_labels=["error pos", "error neg"]))
    expenv.agent.actor.obsfactor = np.array([800 for x in range(2)])  # clamped anyway

if __name__ == "__main__":
    args = exp.parseargs()
    exp.trainingrun(configure_training, num_processes=args.processes, gridsearchpath=args.gridsearch)
