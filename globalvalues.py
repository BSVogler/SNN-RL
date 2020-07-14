import nest
import numpy as np
import typing


class Settings:
    workerdata = None # per training instance
    demo = False  # when set to true will only display and not learn or train. there are more setting to disabel training
    neuronLabelMap: typing.Dict[int, str] = {}  # map neuron ids to strings
    seed = 42
    cycle_length = 40.0  # ms
    voltageRecordings = 0  # a set of ids recording the voltages
    delay = 1.0  # delay between EPSP/IPSP and action potential
    filter_outsignals = False  # filter signal as in fremaux 2013
    neuronmodel = "iaf_psc_alpha"  # wunderlich use iaf_psc_exp
    num_hidden_neurons = 0  # only free neurons
    num_output_populations = 2  # number of ouput populations
    population_size = 5
    lateral_inhibition = 300 #if zero it is disabled, weight
    criticresolution = 300
    dynamic_baseline = True

    fraction_hidden_inhibitory = 0.2  # can be overwritten by using a manual wiring
    manualwiring: typing.List[typing.Tuple[int, int, bool]] = None  # which population is connected to which, last param is excitatory or inhibitory

    manual_num_inhibitory = 0  # overwrites automatic creation of inhibitory
    errsig_factor = 0.052  # https://github.com/clamesc/Training-Neural-Networks-for-Event-Based-End-to-End-Robot-Control/blob/56dc686cbc660e8c462c2c2b6f1310f83ee70ea9/Controller/R-STDP/parameters.py#L29
    factor_negative_util = 1  # factor for negative utilities, Dabney2020
    tau_n = 200.  # Time constant of reward signal
    tau_c = 1000.  # Time constant of eligibility trace
    w0_min = 100.  # Minimum initial random value
    w0_max = 500.  # Maximum initial random value
    w_max = 3000.0  # https://github.com/clamesc/Training-Neural-Networks-for-Event-Based-End-to-End-Robot-Control/blob/56dc686cbc660e8c462c2c2b6f1310f83ee70ea9/Controller/R-STDP/parameters.py#L24

    max_poisson_freq = 500.
    max_util_integral = float("inf")  # 0.7
    util_discount_factor = 0.8
    util_learn_rate = 1.0  # usually 0.9, to disable td learning set it to 1

    vq_learning_scale = 1e-3  # learn rate of vector quantization (scaling), 0 disables it
    vq_decay = 1e-4 # (decaying speed)

    num_episodes = 6000
    max_cycles = 600  # 300 ~ episode in lf2
    num_plots = 0  # number of spiking plots
    render = False  # render the environment
    headless = False  # prevents every rendering, even in pyplot, prevents crash when importing from gym.envs.classic_control import rendering
    save_to_db = False
    allow_early_termination = False

    structural_plasticity = True  # allow removal or adding of synapses
    random_reconnect = False  # randomly adds new synapses
    strp_min = w0_min / 4  # minimum value where a synapse is removed
    @staticmethod
    def define_stdp(vt):
        # the time constant of the depressing window of STDP is a parameter of the post-synaptic neuron.
        rstdp_syn_spec_exitory = {'Wmax': gv.w_max,
                                  'Wmin': 0.0,
                                  'delay': gv.delay,
                                  "weight": gv.w0_min,  # will be overwritten with randomized value later
                                  "A_plus": 1.0,
                                  "A_minus": 1.0,
                                  "tau_n": gv.tau_n,
                                  "tau_c": gv.tau_c,
                                  'vt': vt.tolist()[0]}

        rstdp_syn_spec_inhibitory = rstdp_syn_spec_exitory.copy()
        rstdp_syn_spec_inhibitory["Wmin"] = -gv.w_max
        rstdp_syn_spec_inhibitory['Wmax'] = 0.0

        # alternative to set defaults is to create a new model from a copy
        nest.CopyModel('stdp_dopamine_synapse', 'stdp_dopamine_synapse_ex', rstdp_syn_spec_exitory)
        # nest.SetDefaults("stdp_dopamine_synapse_ex", rstdp_syn_spec_exitory)
        nest.CopyModel('stdp_dopamine_synapse', 'stdp_dopamine_synapse_in', rstdp_syn_spec_inhibitory)
        # nest.SetDefaults("stdp_dopamine_synapse_in", rstdp_syn_spec_inhibitory)

    @staticmethod
    def init():
        # run for every process
        # same results every run
        nest.SetKernelStatus({"grng_seed": gv.seed})
        # numpy seed
        np.random.seed(gv.seed)
        nest.EnableStructuralPlasticity()
        numproc = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
        gv.pyrngs = [np.random.RandomState(s) for s in range(gv.seed, gv.seed + numproc)]


gv = Settings()
