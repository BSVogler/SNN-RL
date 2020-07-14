#%%
# import nest
import nest.voltage_trace

from globalvalues import gv
import matplotlib.pylab as plt


# this experimetns shows the linearity of the firing rate
if __name__ == "__main__":
    # %%
    inoutmap = {}
    for i in range(40):
        inp = nest.Create(gv.neuronmodel)
        outp = nest.Create(gv.neuronmodel)
        nest.Connect(inp, outp, syn_spec={"weight": 5000.0})
        voltmeter = nest.Create("voltmeter")
        nest.Connect(voltmeter, inp)
        nest.Connect(voltmeter, outp)
        sd = nest.Create('spike_detector')
        nest.Connect(inp, sd)
        sd2 = nest.Create('spike_detector')
        nest.Connect(outp, sd2)
        inp.I_e = 40.0*i
        nest.Simulate(100.0)
        #nest.voltage_trace.from_device(voltmeter)
        #plt.show()
        count_in = len(nest.GetStatus(sd)[0]['events']['times'])
        count_out = len(nest.GetStatus(sd2)[0]['events']['times'])
        inoutmap[count_in] = count_out
        nest.ResetKernel()
        #print(f"in:{count_in}")
        #print(f"out:{count_out}")
        #print(f"amplification {count_out/count_in:.2}")
#%%
    x,y = zip(*inoutmap.items())
    plt.plot(x,y)
    plt.title("Rate Code Transfer Function")
    plt.xlabel("Number of Presynaptic Spikes")
    plt.ylabel("Number of Elicited Spikes")
    plt.grid()
    plt.show()
    #draw.spikes(nest.GetStatus(sd)[0]["events"], outsignal=nest.GetStatus(sd2)[0]["events"], output_ids=["in", "out"])