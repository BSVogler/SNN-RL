import os
import sys
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np

from actors import fremauxfilter
from actors.actor import Weightstorage
from settings import gv
import datetime
from matplotlib.colors import LightSource


# a module of helperfunctions to plot various statistics

def voltage(measurements, persp="3d"):
    """

    :param measurements: nest.GetStatus(multimeter)[0]["events"]
    :param persp:
    :return:
    """
    # draw voltage curves
    fig = plt.figure(num=0, figsize=(6.5, 3.5), dpi=150)
    if persp == "3d":
        ax = fig.add_subplot(111, projection='3d')
        ax.set_proj_type('ortho')
        for n in range(gv.voltageRecordings):
            voltage = measurements["V_m"][n::len(gv.voltageRecordings)]  # skip 6, start with offset
            times = measurements["times"][n::len(gv.voltageRecordings)]
            ax.plot(xs=times % 100, ys=times / 100, zs=voltage, label=f"$V_m$ {list(gv.neuronLabelMap.values())[n]}")
        plt.legend()
        plt.show()
    else:
        ax = fig.add_subplot(111)
        for neuron in gv.voltageRecordings:
            voltage = measurements["V_m"][
                      neuron::len(gv.voltageRecordings)]  # pick every numMeasurements, start with offset n
            times = measurements["times"][neuron::len(gv.voltageRecordings)]
            ax.plot(times, voltage, label=f"$V_m$ {neuron}: {gv.neuronLabelMap[neuron]}")
        plt.legend()
        plt.ylabel("voltage [mV]")
        plt.xlabel("time [ms]")
        plt.title("Voltage over time")
        plt.show()


def spikes(spikes_nest: Dict, outsignal: List[List[float]], output_ids: List[int] = []):
    """
    Draws only recorded spikes.
    :param outsignal: 
    :param spikes_nest:
    :param output_ids: pass ids of output channel for different color and further analysis
    :return:
    """
    last_spiketime: int = spikes_nest["times"][-1]
    lastvalid = np.arange(0, last_spiketime, gv.cycle_length)[-1]  # lazy hack
    # create list of signals from nest format
    spiketimes = {}  # dicts are only filled for non-outspikes
    colors = []
    colorcounter = 0
    onlyout = False
    import matplotlib.colors as mcolors
    for i, spiketime in enumerate(spikes_nest["times"]):
        if spiketime > lastvalid:
            continue
        neurid = spikes_nest['senders'][i]
        # if the first time this neuron appeared create new list
        if neurid not in spiketimes:
            if not onlyout or neurid in output_ids:
                spiketimes[neurid] = []
                # sort into categories
                if neurid in output_ids:
                    colors.append(list(mcolors.TABLEAU_COLORS)[colorcounter % 10])
                    colorcounter += 1
                else:
                    colors.append("black")
        if not onlyout or neurid in output_ids:
            spiketimes[neurid].append(spiketime)

    # put labels on it
    labels = []
    if len(gv.neuronLabelMap) > 0:
        for key in list(spiketimes.keys()):
            label = gv.neuronLabelMap[key] if key in gv.neuronLabelMap else ""
            labels.append(label)

    # now draw
    height = max(2 + len(spiketimes) / 3, 15)
    fig = plt.figure(figsize=(14, height))

    plt.subplot(411)
    plt.eventplot(spiketimes.values(), linewidths=0.8, colors=colors)
    plt.yticks(range(len(spiketimes.keys())), labels)
    # include upper limit
    xticks = np.arange(0, last_spiketime, gv.cycle_length)
    if len(xticks) < 500:
        # only draw xticks if not noo much
        plt.xticks(xticks)
    plt.grid()
    plt.margins(x=0.03)  # kinda misaligned because it does not start spiking at 0 and last cycle time
    plt.ylabel("Neuron")
    plt.xlabel("time [ms]")
    plt.title('Spike Events per Neuron')

    if len(output_ids) > 0:
        # plot filtered activity
        plt.subplot(412)
        outsignal_dict: Dict[str, List[float]] = dict()
        for i, population in enumerate(outsignal):
            outsignal_dict[f"population{i}"] = population
        filtersig = filtered_signal(last_spiketime, outsignal_dict)
        if len(xticks) < 500:
            plt.xticks(xticks)
        plt.margins(x=0.02)
        plt.xlabel("time [ms]")
        plt.ylabel("Activity")

        # plot sampled activity
        plt.subplot(413)
        read_out_activity(last_spiketime, outsignal_dict, filtered_signal=filtersig)
        if len(xticks) < 500:
            plt.xticks(xticks)
        plt.margins(x=0.02)
        plt.xlabel("time [ms]")
        plt.ylabel("Activity")

        # plot ??
        plt.subplot(414)
        insignal: Dict[str, List[float]] = dict()
        insignals = list(spiketimes.values())
        for i, population in enumerate(insignals):
            insignal[f"population{i}"] = population
        read_out_activity(last_spiketime, insignal)
        plt.margins(x=0.02)
        if len(xticks) < 500:
            plt.xticks(xticks)
        plt.xlabel("time [ms]")
    plt.show()


def filtered_signal(last_spiketime: int, spikes_per_id: Dict[str, List[float]]):
    """Draw a plot with fremaux fitlering"""
    signal = np.empty((len(spikes_per_id), int(last_spiketime)))  # for every ms

    outspikes_list = list(spikes_per_id.values())
    for t in range(0, int(last_spiketime)):
        signal[:, t] = fremauxfilter.filter(t, outspikes_list)

    # draw
    plt.title('Continuous, Filtered Out-Neurons Activity')
    plt.plot(signal.T)
    plt.ylabel("Activity")
    ax = plt.gca()
    ax.xaxis.grid(True)
    plt.legend(spikes_per_id.keys())
    return signal


def read_out_activity(last_spiketime: int, spikes_per_id: Dict[str, List[float]], filtered_signal=None):
    """draws a plot showing the activity of the out neurons when read."""
    # data for activity diagram
    activity_in_cycle = np.zeros((len(spikes_per_id),  # yaxis
                                  int(last_spiketime // gv.cycle_length) + 1))  # time axis
    nidx = 0
    # for each neuron
    for spiketimes in spikes_per_id.values():
        if len(spiketimes) == 0:
            continue
        for spike in spiketimes:
            cycle = int(spike // gv.cycle_length) + 1  # offset of one bc 0-50 is cycle 1
            # might be in no cycle
            if cycle < len(activity_in_cycle[nidx]):
                activity_in_cycle[nidx][cycle] += 1
        nidx += 1
    # xcoordinates
    sample_times = range(0, int(activity_in_cycle.shape[1] * gv.cycle_length), int(gv.cycle_length))

    # filtered signal only at sample time
    if filtered_signal is not None:
        filtered_signal_sampled = np.empty_like(activity_in_cycle)
        for i, t in enumerate(range(0, filtered_signal.shape[1], int(gv.cycle_length))):
            filtered_signal_sampled[:, i] = filtered_signal[:, t]

        # use least squares to find a scaling that almost fits everywhere
        tominimize = lambda scaling: activity_in_cycle.flatten() - scaling * filtered_signal_sampled.flatten()
        import scipy
        scaling = scipy.optimize.leastsq(tominimize, x0=10)[0][0]
        # normalize
        # scaling = np.max(activity_in_cycle) / np.max(filtered_signal_sampled)
        filtered_signal_sampled *= scaling
        plt.plot(sample_times, filtered_signal_sampled.T, drawstyle='steps-post',
                 label="Filtered Signal Sampled (normalized)")

        plt.title(f"Sampled Activity Out-Neurons (Scaling ({scaling:.2f})")
    else:
        plt.title(f"Sampled Activity Out-Neurons")
    # todo use neuron output id
    labels = None if len(spikes_per_id) > 5 else "Number of Spikes per Cycle"
    # draw
    plt.plot(sample_times, activity_in_cycle.T, drawstyle='steps-post', label=labels)
    plt.ylabel("Activity")
    ax = plt.gca()
    ax.xaxis.grid(True)
    plt.legend()


def return_graph(rewards, show=True, ax=None):
    """A plot showing the average reward over time."""
    plt.plot(rewards, label="Returns")

    def running_mean(x, N: int):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    try:
        plt.plot(running_mean(rewards, min(int(len(rewards) / 4), 40)), label="moving average")
    except ValueError:
        pass
    print(f"final avg. reward of last 100 {np.average(rewards[-100:])}/195")
    plt.legend()
    plt.ylabel("Return")
    plt.xlabel("Trial Number")
    plt.title('Return per trial')
    if ax is not None:
        ax.axhline(y=np.average(rewards), xmin=0.0, xmax=1.0, linestyle='--', dashes=(0.5, 5.))
    if show:
        plt.show()


def weight_changes(syn_w: np.ndarray, connections: np.ndarray = None, show=True):
    """
    Draw changes in weights
    :param syn_w: time dimension, then list of weight
    :param connections: should match connetions
    :param show:
    :return:
    """
    plt.title('Weight with reward factor ' + str(gv.errsig_factor))
    # syn_w = syn_w.reshape(syn_w.shape[0], syn_w.shape[1]*syn_w.shape[2])

    weights = syn_w[..., 2]  # all times, all weights, only weights

    plt.plot(range(len(weights)), weights)
    if connections is not None and len(connections) <= 11:
        labels = [f"{x[0]}->{x[1]}" for x in connections]
        plt.gca().legend(labels)
    plt.ylabel("weight")
    plt.xlabel("episode")
    if show:
        plt.show()


def error_signal(utility, fig=None, persp="heat"):
    plt.title('Used error signals')
    max_cycles = np.max(np.argmin(utility, axis=1)) + 1
    valuesToShow = utility[:, :max_cycles].T  # switch cycles and episodes
    # plt.plot(valuesToShow)

    if persp == "3d":
        cycle_axis = np.linspace(0, valuesToShow.shape[0], valuesToShow.shape[0])
        episode_axis = np.linspace(0, valuesToShow.shape[1], valuesToShow.shape[1]).T

        sx = cycle_axis.size
        sy = episode_axis.size

        cycle_axis = np.tile(cycle_axis, (sy, 1))
        episode_axis = np.tile(episode_axis, (sx, 1)).T
        if fig is None:
            newfig = plt.figure()
            ax = newfig.add_subplot(1, 1, 1, projection='3d')
        else:
            ax = fig.add_subplot(3, 1, 1, projection='3d')
        light = LightSource(315, 45)
        cm = plt.cm.get_cmap("inferno")
        azimuth = 45
        altitude = 60
        ax.view_init(altitude, azimuth)
        if gv.num_episodes > 1:
            illuminated_surface = light.shade(valuesToShow, cmap=cm)
            ax.plot_surface(cycle_axis, episode_axis, valuesToShow, rstride=1, cstride=1, linewidth=0,
                            antialiased=False, facecolors=illuminated_surface, label="utilities")
        else:
            ax.plot_surface(cycle_axis, episode_axis, valuesToShow, rstride=1, cstride=1, linewidth=0,
                            antialiased=False, label="utilities")
    else:
        current_cmap = plt.cm.get_cmap("inferno")
        current_cmap.set_bad(color='green')
        legend = plt.imshow(valuesToShow, cmap=current_cmap, interpolation='nearest')
        plt.colorbar(legend)
    plt.ylabel("cycle")
    plt.xlabel("episode")
    # plt.title("Utility per cycle") #title is covered in condensed view
    # plt.legend()
    if fig is None:
        plt.show()


def utilities_over_time(utils):
    plt.plot(utils.T)
    plt.title("Utilities over time")
    plt.xlabel("cycles")
    plt.ylabel("Utility")
    plt.show()


def report(utility, weights: Union[List[Weightstorage], np.ndarray], returnpereps, connections: np.ndarray,
           filename=None,
           env=None):
    """
    Draw a report consisting of many parts
    :param env:
    :param filename:
    :param utility:
    :param weights:
    :param returnpereps:
    :param connections:
    :return:
    """
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Report " + str(datetime.datetime.now()))
    # plt.subplot(321)
    # connectome(connections)
    plt.subplot(322)
    try:
        error_signal(utility, fig=fig)
    except:
        print("Rendering of error signal history failed.")

    if env is not None and not gv.headless:
        plt.subplot(323)
        try:
            plt.imshow(env.render(mode='rgb_array'))
        except:
            print("Rendering of env failed.")

    plt.subplot(324)
    try:
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        weight_changes(weights, connections, show=False)
    except:
        print("Rendering of weight changes failed.")

    plt.subplot(325)
    plt.text(0.0, 0.5, str(gv.workerdata), fontsize=12, wrap=True)

    ax = plt.subplot(326)
    returnpereps = np.array(returnpereps)
    try:
        nr = np.isnan(returnpereps)
        returnpereps[nr] = 0
        return_graph(returnpereps, show=False, ax=ax)
    except:
        e = sys.exc_info()[0]
        print("Rendering of return history failed:" + str(e))

    if filename is None:
        # find free filename
        counter = 0
        filename = f"report{counter}.pdf"
        while os.path.isfile(filename):
            counter += 1
            filename = f"report{counter}.pdf"
    plt.savefig(filename)
