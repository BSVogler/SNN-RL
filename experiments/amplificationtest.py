#%%
# import nest
import numpy as np
import nest.raster_plot

import matplotlib.pylab as plt


# this experiments shows the transfer function of a rate coded
if __name__ == "__main__":
    # %%
    inoutmap = {}
    #nest.SetKernelStatus({'resolution': 0.01})
    nest.set_verbosity("M_ERROR")
    nsteps = 460
    for i in range(nsteps):
        nest.ResetKernel()
        #alternatively use a poisson generator
        #inp = nest.Create("poisson_generator", {"origin":0.0, "start":0.0,"stop":100.0, "rate": float(i*6)}})
        inp = nest.Create("spike_generator", {"allow_offgrid_times": True,"spike_times": np.linspace(0.1,100.1,int(i*0.4))})
        outp = nest.Create("iaf_psc_alpha") #linear with aeif_psc_alpha
        sd = nest.Create('spike_detector')
        sd2 = nest.Create('spike_detector')
        nest.Connect(inp, outp, syn_spec={"weight": 1000.0})
        nest.Connect(inp, sd)
        nest.Connect(outp, sd2)
        spike_det = nest.Create("spike_detector")
        nest.Connect(inp, spike_det)
        nest.Connect(outp, spike_det)

        nest.Simulate(100.0)

        count_in = len(nest.GetStatus(sd)[0]['events']['times'])
        count_out = len(nest.GetStatus(sd2)[0]['events']['times'])
        inoutmap[count_in] = count_out
#%%
    # scount = spike_det.get("n_events")
    # if scount>0:
    #     nest.raster_plot.from_device(spike_det, hist=False)
    #     plt.show()
#%%
    x,y = zip(*inoutmap.items())
    plt.plot(x,y,marker='.', label="rate code transfer function")
    plt.title("Rate Code Transfer Function")
    plt.xlabel("Number of Presynaptic Spikes")
    plt.ylabel("Number of Elicited Spikes")
    plt.grid()
    plt.show()
    #draw.spikes(nest.GetStatus(sd)[0]["events"], outsignal=nest.GetStatus(sd2)[0]["events"], output_ids=["in", "out"])