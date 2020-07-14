import nest
import nest.raster_plot
import numpy as np
import matplotlib.pyplot as plt
import draw
import globalvalues as gv

gv.configure_nest()
nest.ResetKernel()
type = "iaf_psc_alpha"
inputs = nest.Create("poisson_generator", 1)
# introduce parrot neuron to fix limitation of devices with STDP synapses
parrots = nest.Create("parrot_neuron", 1)
nest.Connect(inputs, parrots)
spike_detector = nest.Create('spike_detector')
nest.Connect(parrots, spike_detector, 'all_to_all')
out = nest.Create(type,1)

vt = nest.Create('volume_transmitter')
gv.define_stdp(vt)
nest.Connect(parrots, out, syn_spec={'model': 'stdp_dopamine_synapse_ex'})

nest.SetStatus(inputs, {"start":0.,
                        "rate":1000.,
                        "stop":gv.cycle_length})
nest.Simulate(gv.cycle_length)
#now stop spiking and give reward
conn = nest.GetConnections(source=parrots, target=out)
weights = []
weights.append(np.array(nest.GetStatus(conn, keys="weight")))
nest.SetStatus(conn, {"n": 100.})
weights.append(np.array(nest.GetStatus(conn, keys="weight")))
nest.Simulate(1.)
weights.append(np.array(nest.GetStatus(conn, keys="weight")))
nest.Simulate(1.)
weights.append(np.array(nest.GetStatus(conn, keys="weight")))
nest.Simulate(1.)
weights.append(np.array(nest.GetStatus(conn, keys="weight")))

spikes = nest.GetStatus(spike_detector)[0]["events"]
draw.spikes(spikes)

#nest.raster_plot.from_device(spike_detector, hist=True, hist_binwidth=40.,
#                             title='Repeated stimulation by Poisson generator')
#nest.raster_plot.show()

plt.plot(weights)
plt.show()