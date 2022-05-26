import copy
import nest
from typing import Dict, List


def Dump(selections: List = ("nodes", "synapses")) -> Dict:
    """Returns a dictionary containing the current net in memory for serialization.

    Only a subset of the data can be serialized.


    Parameters
    ----------
    selections : List[strings], optional
        Only obtain a subset of the net parameters. Options are "nodes" and/or "synapses"


    Returns
    -------
    Dict:
        Object containing the specified serialization sorted by dict key likes "nodes" or "synapses"


    See Also
    ---------

    Load : Function to construct the net from the data.

    """

    nest.ll_api.sli_func("({}) M_WARNING message".format("Dumping is currently only supported for a single process."))
    dumpdata = dict()
    numnetwork = nest.GetKernelStatus("network_size")

    # nothing added
    if numnetwork < 2:
        return dumpdata

    nodes = nest.NodeCollection(range(1, numnetwork + 1))
    if "nodes" in selections:
        dumpdata["nodes"] = nest.GetStatus(nodes)

    syn_ids = nest.GetConnections(source=nodes)
    if "synapses" in selections:
        dumpdata["synapses"] = nest.GetStatus(syn_ids)
    return dumpdata


def Load(data: Dict) -> Dict:
    """
    Loads a dictionary obtained by the dump method.

    Repeated loading will add to the network size. To overwrite clear with a call to resetnetwork.
    To directly overwrite the binary memory state a lower level access needs to be developed in the future.

    Parameters
    ----------
    data: Dictionary
        the data to be loaded.

    Returns
    -------
    Dict:
        the created nest obejcts and all synapses

    See Also
    ---------

    Dump : Function to obtain a structured memory dump.
    """
    created = dict()
    nest.ll_api.sli_func("({}) M_WARNING message".format("Loading is currently only supported for a single process."))
    if "nodes" in data:
        dictmissbefore = nest.GetKernelStatus({"dict_miss_is_error"})[0]
        nest.SetKernelStatus({"dict_miss_is_error": False})
        verbose = nest.get_verbosity()
        nest.set_verbosity("M_ERROR")
        newnodes = nest.NodeCollection()
        try:
            for d in data["nodes"]:
                newnodes += nest.Create(d["model"], d)
        finally:
            nest.SetKernelStatus({"dict_miss_is_error": dictmissbefore})
            # restore verbosity level
            nest.set_verbosity(verbose)

        created["nodes"] = newnodes

    if "synapses" in data:
        try:
            for conn in data["synapses"]:
                source, target = conn["source"], conn["target"]
                # remove unused, copy to prevent side effects
                specs = copy.copy(conn)
                specs.pop("port")
                specs.pop("receptor")
                specs.pop("synapse_id")
                specs.pop("target_thread")
                specs.pop("source")
                specs.pop("target")
                nest.Connect([source], [target], syn_spec=specs)
        except Exception as e:
            print("Error during synapse loading:" + str(e))

        #we only want the newly created connections, but nest.Connect does not return a SynapseCollection object
        created["synapses"] = nest.GetConnections()

    return created
