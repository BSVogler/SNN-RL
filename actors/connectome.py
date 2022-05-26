import collections
import pickle

import dumpload

try:
    import nest
except ImportError:
    print("Neural simulator Nest backend not found (import nest).")

import numpy as np
from nest.lib.hl_api_types import NodeCollection, SynapseCollection

import draw
from actors.actor import Weightstorage
from settings import gv

Vertex = tuple[int, int]


class Connectome:
    def __init__(self, actor, num_input=8, num_output=2, neuron_labels: list[str] = []):
        """
        Create and connect a nest network
        :param initial:
        :param num_input:
        :param num_output:
        :return:
        """
        self.actor = actor
        self.connectome_dump: dict = {}
        # note that the order does not reflect the internal nest order
        self.neur_ids_in: list[int] = []
        self.neur_ids_detect: list[list[int]] = []
        self.neur_ids_parrot = []
        self.neur_ids_out: list[int] = []
        self.neur_ids_core: np.ndarray  # the neurons which are used for wiring (except spike generators)
        self.neur_ids_ex = []  # includes input
        self.neur_ids_hidden_in = []  # ids of the free (not in population) in hidden layer
        self.neur_ids_hidden_ex = []  # ids of the free (not in population) in hidden layer

        # store connected pairs (source, target) for reconnecting, order different from nest
        self.conns: np.ndarray
        self.conns_in: np.ndarray
        self.conns_ex: np.ndarray

        # redundance in nest format/indexing
        self.conns_nest: SynapseCollection = []  # contains all stdp synapses
        self.conns_nest_ex: SynapseCollection = []
        self.conns_nest_in: SynapseCollection = []
        self.populations_nest: list[NodeCollection] = []
        self.neurons_nest_ex: NodeCollection = []
        self.neurons_nest_in: NodeCollection = []
        self.neurons_nest: NodeCollection = []
        self.neurons_input: NodeCollection = []

        self.cycles_to_reconnect = 2  # counts the cycles till a reconnect (structural plasticity) happens
        self.num_input = num_input
        self.num_output = num_output
        self.last_num_spikes = []  # record the number of spikes in this trial to only get new spikes for each out population
        self.multimeter = None
        self.synapsecontingent = 1

        self.spike_recorder = None
        self.spike_recorders_populations: list[NodeCollection] = []
        # log
        self.lastweights: Weightstorage = []  # only stores last entries for restoring in next episode, order like nest connections

        self.neuron_labels: list[str] = neuron_labels

        # after initalizing fields construct
        nest.set_verbosity("M_WARNING")
        # create neuromodulator synapse
        vt = nest.Create('volume_transmitter')
        gv.define_stdp(vt)
        self.construct()
        print("Training " + str(len(self.conns_nest)) + " weights")

    def create_input_layer(self, initial):
        """input layer in nest"""
        # todo use populations of size > 1
        self.neurons_input = nest.Create("poisson_generator", self.num_input)
        self.neur_ids_in = self.neurons_input.tolist()
        # introduce parrot neuron to fix limitation of devices with STDP synapses
        self.parrots = nest.Create("parrot_neuron", self.num_input)
        self.neur_ids_parrot = self.parrots.tolist()
        if initial:
            self.neur_ids_ex.extend(self.neur_ids_parrot)
            # add each parrot as a population
            for i in range(self.num_input):
                self.populations_nest.append(self.parrots[i])
            # connect without adding to front-end
        nest.Connect(self.neurons_input, self.parrots, "one_to_one")

        self.spike_recorder = nest.Create('spike_recorder')
        nest.Connect(self.parrots, self.spike_recorder, 'all_to_all')

    def init_labels(self):
        """
        set labels for matching neuron id to label
        :return:
        """
        if self.neuron_labels is not None and self.num_input < 3000:
            neuronlist = self.neurons_input.tolist()
            parrotlist = self.parrots.tolist()
            for i in range(self.num_input):
                gv.neuronLabelMap[neuronlist[i]] = self.neuron_labels[i]
                # copy labels
                gv.neuronLabelMap[parrotlist[i]] = gv.neuronLabelMap[neuronlist[i]]

        gv.neuronLabelMap[self.neur_ids_out[0]] = "Left"
        gv.neuronLabelMap[self.neur_ids_out[1]] = "Right"

    def connect_front_end(self):
        """Connect middleware to fron front-end related connections"""
        # temporary for faster creation
        self.conns_ex = []
        self.conns_in = []
        self.conns = []
        if gv.manualwiring is not None:
            for conn in gv.manualwiring:
                for from_neuron in self.populations_nest[conn[0]].tolist():
                    for to_neuron in self.populations_nest[conn[1]].tolist():
                        self.conns.append((from_neuron, to_neuron))
                        if conn[2]:
                            self.conns_ex.append((from_neuron, to_neuron))
                        else:
                            self.conns_in.append((from_neuron, to_neuron))
        else:
            # connect input to everything
            self.add_connection(self.neur_ids_parrot, self.neur_ids_out, 'excitatory')
            self.add_connection(self.neur_ids_parrot, self.neur_ids_hidden_ex, 'excitatory')
            # connect hidden with out
            self.add_connection(self.neur_ids_hidden_ex, self.neur_ids_out, 'excitatory')

        self.conns_ex = np.array(self.conns_ex)
        self.conns_in = np.array(self.conns_in)
        self.conns = np.array(self.conns)

    def construct(self):
        self.populations_nest.clear()
        self.create_input_layer(True)

        if gv.manualwiring is not None:
            num_inhibitory = gv.manual_num_inhibitory
        else:
            if gv.num_hidden_neurons > self.num_output + 1:
                num_inhibitory = int(np.ceil((self.num_output + gv.num_hidden_neurons) * gv.fraction_hidden_inhibitory))
            else:
                num_inhibitory = 0
        # create populations
        # self.neurons_nest_ex = nest.NodeCollection([])
        # hidden population
        if gv.num_hidden_neurons - num_inhibitory < 0:
            raise AssertionError("Invalid number of excitatory and inhibitory neurons")
        # reset last detected spikes
        self.spike_recorders_populations = []
        num_hidden_ex = gv.num_hidden_neurons - num_inhibitory
        self.neurons_nest_ex = None
        if num_hidden_ex > 0:
            self.neur_ids_hidden_ex = self.create_population(num_hidden_ex,
                                                             initial=True,
                                                             recurrent=True)  # todo add fraction of inhibitory
        # create output populations
        self.neur_ids_out = []
        self.last_num_spikes = []
        outpopulations = []
        for i in range(gv.num_output_populations):
            popnest, poplist = self.create_population(gv.population_size, initial=True, recurrent=False)
            self.neur_ids_out.extend(poplist)
            outpopulations.append(popnest)

            # self.neur_ids_in.extend(self.neur_ids_layer_in)
            self.neur_ids_ex.extend(self.neur_ids_out)

        # connect out populations lateral inhibition, not in front-end because it is not a STDP synapse
        if gv.lateral_inhibition > 0:
            # todo replace with autapses': False, all to all
            for i in outpopulations:
                for j in outpopulations:
                    if i != j:
                        nest.Connect(i, j, syn_spec={'weight': -gv.lateral_inhibition})  # static synapses

        self.connect_front_end()
        self.init_labels()

        # using a distribution is not possible with stdp_dopamine_synapse
        # {"distribution": "uniform", "low": -weight, "high": weight}

        # setup measurement devices
        self.multimeter = nest.Create("multimeter")
        self.multimeter.set({"record_from": ["V_m"]})  # removed in nest3: "withtime": True,

        nest.Connect(self.neurons_nest_ex, self.spike_recorder)
        nest.Connect(self.multimeter, self.neurons_nest_ex)
        gv.voltageRecordings = set(self.neurons_nest_ex.tolist())

        # connection in front-end already established: now connect neurons in nest backend
        nest.Connect(self.conns_ex[:, 0],
                     self.conns_ex[:, 1],
                     syn_spec={'synapse_model': 'stdp_dopamine_synapse_ex'},
                     conn_spec="one_to_one")
        if len(self.conns_in) > 0:
            nest.Connect(self.conns_in[:, 0],
                         self.conns_in[:, 1],
                         syn_spec={'synapse_model': 'stdp_dopamine_synapse_in'},
                         conn_spec="one_to_one")

        # update references to nest
        self.update_connections_nest()

        self.connectome_dump = dumpload.Dump()
        # remove volume transmissor from dump, slow as it is unmodifable
        withoutifrst = list(self.connectome_dump["nodes"])
        withoutifrst.pop(0)
        self.connectome_dump["nodes"] = tuple(withoutifrst)
        # random initial weights
        # randomize weights for inhibitory
        if gv.num_hidden_neurons > 1:  # can only have inhibitory if there are more than one excitatory
            rand_weight = gv.pyrngs[0].uniform(gv.w0_min, gv.w0_max, size=len(self.conns_nest_in))
            self.conns_nest_in.set({"weight": -rand_weight})

        rand_weight = gv.pyrngs[0].uniform(gv.w0_min, gv.w0_max, size=len(self.conns_nest_ex))
        self.conns_nest_ex.set({"weight": rand_weight})

        self.neur_ids_core = np.unique(
            [x[0] for x in self.conns] + [x[1] for x in self.conns])  # the connected neurons of interest

    def rebuild(self):
        """in intial phase create connections in front-end (neur_ids), always creates nest connections (back-end).
        Keeps the weights. """
        nest.set_verbosity("M_ERROR")
        nest.ResetKernel()
        # create neuromodulator synapse
        vt = nest.Create('volume_transmitter')
        gv.define_stdp(vt)

        # loading of dump
        dumpload.Load(self.connectome_dump)
        # update node collections reference, suggested improvement in https://github.com/nest/nest-simulator/issues/1821
        self.neurons_input = NodeCollection(self.neur_ids_in)
        self.spike_recorders_populations = [NodeCollection(ids) for ids in self.neur_ids_detect]

        # todo maybe not needed
        self.update_connections_nest()
        # use weights from last episode, offset one because of initial weights in 0
        self.restorelastweights()
        if gv.structural_plasticity and gv.random_reconnect:
            self.random_reconnect()

        nest.set_verbosity("M_WARNING")

    def set_inputactivation(self, rate: np.ndarray):
        """Set the activation levels of the input neurons."""
        self.neurons_input.set({"rate": rate})
        time = nest.GetKernelStatus("time")
        self.neurons_input.set({"origin": time})
        self.neurons_input.set({"stop": gv.cycle_length})

    def get_weights(self) -> Weightstorage:
        """

        :return: a dict where a tuple of source target returns the weight
        """
        data = self.conns_nest.get(keys={"weight", "source", "target"})
        # save in history
        self.lastweights = np.array((data["source"], data["target"], data["weight"])).T
        return self.lastweights

    def restorelastweights(self):
        """Sets the last set weights to the nest back-end"""
        self.set_weights(self.lastweights)

    def load(self, mongoid: str):
        from models import trainingrun
        weights = trainingrun.Episode.objects(id=mongoid).first().weights
        # unpickle
        wlist = pickle.loads(weights)
        self.set_weights(wlist)

    def set_weights(self, weights: Weightstorage, pick=False):
        """
        Assume that the back-end contains the connections where this was extracted.
        :param weights: source, target, weights
        :param pick: if true will parse by using source and target information. slower
        :return:
        """
        if pick:
            for (source, target, weight) in weights:
                nest.GetConnections(source=NodeCollection([int(source)]),
                                    target=NodeCollection([int(target)]),
                                    synapse_model='stdp_dopamine_synapse_ex').set({"weight": weight})
        else:
            self.conns_nest.set({"weight": weights[:, 2]})

    def create_population(self, num_neurons: int, initial: bool, recurrent=False) -> tuple[NodeCollection, list[int]]:
        """Creates nest neurons and automatically connects"""

        # back-end
        neurons_nest = nest.Create(gv.neuronmodel, num_neurons)
        if self.neurons_nest_ex is None:
            self.neurons_nest_ex = neurons_nest
        else:
            self.neurons_nest_ex += neurons_nest
        self.neurons_nest = neurons_nest
        neurons_list = neurons_nest.tolist()
        self.populations_nest.append(neurons_nest)
        # add a detector
        detector = nest.Create('spike_recorder')
        self.spike_recorders_populations.append(detector)
        self.neur_ids_detect.append(detector.tolist())
        self.last_num_spikes.append(0)
        nest.Connect(neurons_nest, detector)

        if initial and recurrent:
            for neur_a in neurons_list:
                for neur_b in neurons_list:
                    if neur_a != neur_b:  # no autapses
                        self.add_connection(neur_a, neur_b, True)

        return neurons_nest, neurons_list

    def add_connection(self, from_nid, to_nid, synapse_type: bool):
        """
        Connects lists or single neurons in an all to all fashion to the front-end.
        :param from_nid: global neuron id
        :param to_nid: global neuron id
        :param synapse_type: true excitatory, false inhibitory
        :return:
        """
        # resolve tuples
        if isinstance(from_nid, collections.Sequence):
            for f in from_nid:
                if isinstance(to_nid, collections.Sequence):
                    for t in to_nid:
                        if f != t:
                            self.add_connection(f, t, synapse_type)
                else:
                    if f != to_nid:
                        self.add_connection(f, to_nid, synapse_type)
            return
        elif isinstance(to_nid, collections.Sequence):
            for t in to_nid:
                if from_nid != t:
                    self.add_connection(from_nid, t)
            return

        self.conns.append((from_nid, to_nid))
        if synapse_type:
            self.conns_ex.append((from_nid, to_nid))
        else:
            self.conns_in.append((from_nid, to_nid))

    def update_connections_nest(self):
        """
        Before calling this method, connect eveything in the front and back-end. Updates the references to the nest back-end based on the front-end connectome
        """
        # todo should include inhibtiory, todo skip first becaue of some bug when concating synapse collection
        self.conns_nest: SynapseCollection = nest.GetConnections(synapse_model='stdp_dopamine_synapse_ex')
        self.conns_nest_ex: SynapseCollection = nest.GetConnections(synapse_model='stdp_dopamine_synapse_ex')
        if len(self.neurons_nest_in) > 0:
            self.conns_nest_in: SynapseCollection = nest.GetConnections(source=self.neurons_nest_in,
                                                                        target=self.neurons_nest_in,
                                                                        synapse_model='stdp_dopamine_synapse_in')

    def remove_weak_conns(self, connlist, model):
        """
        Checks every synapse and removes weak ones in connlis.
        :param connlist:
        :param model:
        :return:
        """

        if len(connlist) == 0:
            return False

        def checkIfMatch(a: Vertex, b: Vertex):
            return a[0] == b[0] and a[1] == b[1]

        removed = False
        tobe_removed = []
        for _, conn in enumerate(connlist):
            # get directly form backend because we are editing the back-end, which will outdate teh connection
            nestconn = nest.GetConnections(source=nest.NodeCollection([conn[0]]), target=nest.NodeCollection([conn[1]]))
            w = nestconn.get({"weight"})["weight"]
            if abs(w) < gv.strp_min:
                print(f"Disconnecting {conn}")
                nest.Disconnect(nest.NodeCollection([conn[0]]),
                                nest.NodeCollection([conn[1]]),
                                syn_spec={'synapse_model': model})
                tobe_removed.append(conn)
                removed = True

                # delete from conns
                connpair: Vertex = (conn[0], conn[1])
                if model == "stdp_dopamine_synapse_ex":
                    for i, n in enumerate(self.conns_ex):
                        if checkIfMatch(n, connpair):
                            del self.conns_ex[i]
                            break
                else:
                    for i, n in enumerate(self.conns_in):
                        if checkIfMatch(n, connpair):
                            del self.conns_in[i]
                            break
                for i, n in enumerate(self.conns):
                    if checkIfMatch(n, connpair):
                        del self.conns[i]
                        break
        if removed:
            self.synapsecontingent += len(tobe_removed)
            for delete in tobe_removed:
                if delete in connlist:
                    # might be already deleted when using conns_ex or conns_in
                    connlist.remove(delete)
            self.update_connections_nest()

        return removed

    def update_structural_plasticity(self):
        """Cyclic update to check if connectiosn should be removed."""
        if self.cycles_to_reconnect <= 0:
            removed = False
            removed |= self.remove_weak_conns(self.conns_ex, "stdp_dopamine_synapse_ex")
            removed |= self.remove_weak_conns(self.conns_in, "stdp_dopamine_synapse_in")
            if removed:
                self.update_connections_nest()
            self.cycles_to_reconnect = 2
        else:
            self.cycles_to_reconnect -= 1

    def get_outspikes(self, recent=True) -> list[list[float]]:
        """Get the aggregated spike times for a population
        :recent: if false returns signal over whole trial"""
        spikes = [[]] * len(self.spike_recorders_populations)
        for pop_i, detector in enumerate(self.spike_recorders_populations):
            if recent:
                # filter spikes since last cycle
                spikes[pop_i] = detector.get({"events"})["events"]["times"][self.last_num_spikes[pop_i]:]
            else:
                spikes[pop_i] = detector.get({"events"})["events"]["times"]
        return spikes

    def post_cycle(self, cyclenum):
        for i, pop in enumerate(self.spike_recorders_populations):
            self.last_num_spikes[i] = len(pop.get({"events"})["events"]["times"])

    def random_reconnect(self):
        """Adds connection from neurons which recently fired to a random target. Can result in no change."""
        # todo use self.getrecentfiring()

        # calculate neurons which fired in the last cycle
        spikesenders: list[float] = self.spike_recorder.get({"events"})["events"]["senders"]
        # filter spikes since last cycle
        spikesenders: list[float] = spikesenders[self.last_num_spikes:]
        neurons_fired_cycle = np.unique(spikesenders)

        source = np.random.choice(neurons_fired_cycle, 1)[0]
        type = "excitatory" if source in self.neur_ids_ex else "inhibitory"
        # get synapses where there is zero weight
        noconn_from_source = set(np.where(self.actor.lastweightsmatrix[source, :] == 0)[0])
        candidates = set(self.neur_ids_core) & noconn_from_source
        if len(candidates) > 1 and self.synapsecontingent > 0:
            self.synapsecontingent -= 1
            # no self connection
            candidates.remove(source)
            target = np.random.choice(list(candidates), 1)[0]
            print(f"random connect of {source}->{target}")
            # add to front-end
            self.add_connection(source, target, type)
            nest.set_verbosity("M_ERROR")
            nest.Connect(nest.NodeCollection([source]), nest.NodeCollection([target]),
                         syn_spec={
                             'synapse_model': 'stdp_dopamine_synapse_in' if type == "inhibitory" else 'stdp_dopamine_synapse_ex'})
            nest.set_verbosity("M_WARNING")
            synapse = nest.GetConnections(source=nest.NodeCollection([source]), target=nest.NodeCollection([target]))
            if type == "inhibitory":
                synapse.set({"weight": -gv.w0_min})
            else:
                synapse.set({"weight": gv.w0_min})
            # indices have changes so update everything
            self.update_connections_nest()

    def drawspikes(self):
        draw.spikes(nest.GetStatus(self.spike_recorder)[0]["events"],
                    outsignal=self.get_outspikes(recent=False),
                    output_ids=self.neur_ids_out)
