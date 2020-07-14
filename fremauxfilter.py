from typing import List
import numpy as np
from globalvalues import gv


def kernel(t: np.ndarray) -> np.ndarray:
    """
    kernel as defined in fremaux2013
    :param t: first axis: neuron number, second: list of spike times
    :return:
    """
    tau = 40
    ypsilon = 10
    for i, spiketime in enumerate(t):
        #invalidmask = (0 > spiketime) | (spiketime > gv.cycle_length)
        #clip negative spike times
        spiketime = np.clip(spiketime, a_min=0, a_max=None)
        #spiketime[invalidmask] = 0
        t[i] = (np.exp(-spiketime / tau) - np.exp(-spiketime / ypsilon)) / (tau - ypsilon)
    return t


def filter(time, spikesignals: List[List[float]]) -> np.ndarray:
    """Converts vector of list of spike times to vector of list of float at time t."""
    spikesignals_np = np.array([np.array(xi) for xi in spikesignals], dtype=object)
    #per neuron thre can only be one spike at each timestep therefore use 1 for f(tau)
    integrands = kernel(time - spikesignals_np)
    # calc. discrete integral by summing the integrands for each neuron
    ouputs: np.ndarray = np.zeros(len(spikesignals))
    for nidx, integrand in enumerate(integrands):
        ouputs[nidx] = np.sum(integrand)
    return ouputs

    #todo allow filter for every t, e.g. when t is None
