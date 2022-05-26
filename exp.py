import asyncio
import copy
import multiprocessing
import os
import argparse
import pickle
import sys
from multiprocessing import Process
from threading import Thread
from typing import Tuple, List, Dict, Callable, Optional, Union, Awaitable

import mongoengine
import numpy as np
import atexit
import datetime

import websockets
from gym.envs.classic_control import CartPoleEnv
import draw
import models.trainingrun as tg
from agent import Agent
from environments.lineFollowingEnvironment import LineFollowingEnv
from environments.lineFollowingEnvironment2 import LineFollowingEnv2
from mongoengine import Document, FileField, ListField, StringField, IntField, DateTimeField, FloatField, \
    ReferenceField, disconnect, DictField
import json
from settings import gv
from webserv import initiate_server

ws_uri = "ws://localhost:4041"


class Experiment(Document):
    """
    Creates and manages a traiing environment with a SNN RL algorithm.
    """
    training = ReferenceField(tg.Trainingrun)
    parameterdump = StringField()
    time_start = DateTimeField()
    time_end = DateTimeField()
    time_elapsed = FloatField()  # in s
    diagrams = ListField(FileField())
    cycle_i = IntField(default=0)
    totalCycleCounter = IntField(default=0)
    episode = IntField(default=0)
    return_per_episode_sum = ListField(FloatField())  # per episode
    log_Δw = ListField(FloatField())  # per episode
    log_m = ListField(FloatField())  # per episode
    epslength = ListField(IntField())
    episodedata = ListField(ReferenceField(tg.Episode))  # stores references to all episodes
    workerdata = DictField()

    def __init__(self, *args, **values):
        super().__init__(*args, **values)
        gv.init()
        self.printedbias = False
        self.env = None
        self.penalty = -8  # penalty for ending
        self.errsig_contingent = [0]

        self.return_per_episode_sum = []
        self.totalCycleCounter = -1  # will be increased at the beginning of the cycle

        self.log_Δw = []
        self.log_m = []
        self.rewards: List = []  # reward of last episode
        self.errsigs = None
        self.utils = None
        self.agent: Agent = None

        self.lastweights: np.array = 0  # initialized with 0 so that it can be used in computation

        self.epslength = []  # stores the number of cycles for each episode

    def cycle(self, observation_in: np.array) -> np.array:
        """Calculates brain one frame, applies action and simulates environment for a frame
        : observation_in: the last observation
        :return float values
        """
        if gv.render:
            self.env.render()
        self.totalCycleCounter += 1
        # feed observations into brain
        action = self.agent.actor.cycle(time=gv.cycle_length * self.cycle_i,
                                                         observation_in=observation_in)
        # simulate environment
        observation, reward, done, info = self.env.step(action)
        reward_internal = reward
        # distance from ideal position
        # if isinstance(self.env.env, CartPoleEnv):
        #    reward_internal = 50 * np.math.cos(observation[2])
        if not self.printedbias:
            print("Bias: " + str(reward + self.penalty))
            self.printedbias = True

        try:  # try because of env.env
            if done and not (isinstance(self.env.env, CartPoleEnv) and self.cycle_i >= 200):
                # add a penalty for cartpole when failed
                reward_internal += self.penalty
        except:
            pass

        err_signal, util = self.agent.critic.tick(state=observation, new_rewards=[reward_internal])
        # store unedited
        if not gv.demo:
            self.errsigs[self.episode, self.cycle_i] = err_signal
            # self.utils[self.episode, self.totalCycleCounter] = util
            self.rewards.append(reward)

        # clamp utility
        if gv.max_util_integral != float("inf"):
            if abs(self.errsig_contingent[-1] + err_signal) >= gv.max_util_integral:
                err_signal = 0
            self.errsig_contingent.append(self.errsig_contingent[-1] + err_signal)

        # gv.outactivity["utility"].append(utility)

        if gv.structural_plasticity:
            self.agent.actor.connectome.update_structural_plasticity()

        # Set reward signal for left and right network
        self.agent.actor.give_reward(err_signal * gv.errsig_factor)

        self.agent.post_cycle(self.cycle_i)
        return done, observation

    def simulate_episode(self) -> bool:
        """Simulate one episode
        :return: True if everything went okay. False if training needs to be canceled
        """
        if self.episode > 0:
            self.agent.pre_episode()

        observation = self.env.reset()
        self.rewards.clear()
        for self.cycle_i in range(gv.max_cycles):
            # if failed, break early
            done, observation = self.cycle(observation_in=observation)
            if done:
                break
        # extra simulation time to apply changes in last cycle before resetting
        self.epslength.append(self.cycle_i)

        return self.post_episode()

    def post_episode(self) -> bool:
        """
        :return: True if everything went okay. False if training needs to be canceled
        """
        if not gv.demo:
            self.agent.post_episode(self.episode)
        eps: tg.Episode = tg.Episode()
        eps.rewards = self.rewards
        if gv.save_to_db:
            eps.episode = self.episode
            if len(self.agent.actor.log_m) > 0:
                eps.neuromodulator = self.agent.actor.log_m
                self.log_m.append(np.average(eps.neuromodulator))

        # extract the last weights
        # try:
        #    weights = np.array(list(self.agent.actor.get_weights().values()))
        # except:
        weights = self.agent.actor.get_weights()
        # check if no weight changed -> Early termination
        Δw: float = np.sum(weights - self.lastweights)
        self.log_Δw.append(Δw)
        if gv.allow_early_termination and self.episode > 50 and -0.00001 < Δw < 0.00001:
            self.early_termination(eps, weights)
            return False
        self.lastweights = weights

        self.return_per_episode_sum.append(np.sum(self.rewards))
        if gv.save_to_db:
            # save at the end of the training
            if self.episode > 0 and self.episode % (gv.num_episodes - 1) == 0:
                self.save_episode(eps, weights)
            self.save()
        return True

    def early_termination(self, eps, weights):
        print("\nEarly termination because Δw=0.")
        # todo log a message in the db
        if gv.save_to_db:
            # eps.activation = list(np.average(np.array(self.agent.actor.log_activation), axis=0))
            eps.neuromodulator = self.agent.actor.log_m
            self.save_episode(eps, weights)
            try:
                self.agent.actor.connectome.drawspikes()
            except AttributeError:
                pass
            self.save()

    def save_episode(self, eps, weights):
        eps.weights_human = weights.tolist()
        eps.weights = pickle.dumps(weights)
        eps.save()
        self.episodedata.append(eps.id)

    async def train(self, ws=None, lastsend: Awaitable=None):
        """Trains the agent for given numbers"""
        # extend on existing recordings
        self.errsigs = np.full((self.episode + gv.num_episodes, gv.max_cycles), np.nan)
        for episode_training in range(gv.num_episodes):
            # episode_training=0
            # while self.totalCycleCounter < gv.max_cycles:
            episode_training += 1

            # simulate
            if not self.simulate_episode():
                break
            # "CartPole-v0 defines solving as getting average return of 195.0 over 100 consecutive trials."
            last100return = np.average(self.return_per_episode_sum[self.episode - 100:self.episode + 1])

            # time/performance evaluation
            tpe = (datetime.datetime.utcnow() - self.time_start) / episode_training
            # tpc = (datetime.datetime.utcnow() - self.time_start) / self.totalCycleCounter
            # eta = tpc * (gv.max_cycles - self.totalCycleCounter)
            eta: datetime.timedelta = tpe * (gv.num_episodes - episode_training)
            overwrite = "\r" if self.episode > 0 else ""
            weights = self.agent.actor.get_weights()
            if ws:
                await lastsend #compute while the result can be there later
                lastsend = ws.send(pickle.dumps(weights))
            # when running in same process
            #import webserv
            #webserv.Web_socket_handler.push_update(weights)

            # +1 because episodes are zero-indexed
            sys.stdout.write(
                f"{overwrite}{(self.episode + 1) * 100 / gv.num_episodes:3.2f}% (Episode: {self.episode}, Cycle: {self.totalCycleCounter}) ETA {eta}. 𝔼[r]={last100return:.1f}  |w\u20D7|={np.sum(weights)}...")
            sys.stdout.flush()

            # plots
            if gv.num_plots > 0 and gv.num_episodes > gv.num_plots and self.episode % (
                    gv.num_episodes // gv.num_plots) == 0:
                # draw.voltage(self.agent.actor.connectome.multimeter, persp="2d")
                try:
                    self.agent.actor.connectome.drawspikes()
                except AttributeError:
                    pass
            self.episode += 1

        if lastsend is not None:
            await lastsend

        print(f"Cycles: {self.totalCycleCounter}")

    def drawreport(self):
        # self.agent.critic.draw(xaxis=0, yaxis=1)
        filename = f"{self.id}.png" if self.id is not None else None
        try:
            connectome = self.agent.actor.connectome.conns
        except:
            connectome = None
        draw.report(utility=self.errsigs,
                    weights=np.array(self.agent.actor.weightlog),
                    returnpereps=self.return_per_episode_sum,
                    connections=connectome,
                    filename=filename,
                    env=self.env)

    def presetup(self):
        print("Process w/ worker id " + str(multiprocessing.current_process()))
        dbconnect()

        self.time_start = datetime.datetime.utcnow()
        if gv.save_to_db:
            self.save()  # safe first to get id

        # pre-training
        def dump(obj):
            f = ""
            for attr in dir(obj):
                if attr != "__dict__":
                    f += "obj.%s = %r" % (attr, getattr(obj, attr)) + "\n"
            return f

        self.parameterdump = dump(gv)
        # dump(f, self)
        # dump(f, self.agent.critic)

        # register instance
        self.training.instances.append(str(self.id))
        if gv.save_to_db:
            self.training.save()

    def post_train(self):
        """When training is done (experiment is over)"""
        # stats
        self.time_end = datetime.datetime.utcnow()
        self.time_elapsed = (self.time_end - self.time_start).total_seconds()
        if gv.save_to_db:
            self.save()
        self.drawreport()

        self.env.close()
        self.agent.post_experiment()
        # if not gv.render:
        #    self.show()

    async def run(self, workerdata: Dict = None) -> List[float]:
        """
        Create and trains the network.
        :param configurator:
        :param workerdata:
        :return: the results of the training
        """
        self.training = workerdata.pop("training")
        self.presetup()

        self.workerdata = workerdata
        gv.workerdata = workerdata  # not nice to add it as a global variable

        # create experiment
        configurator: Callable
        if "configurator" in workerdata and workerdata["configurator"] is not None:
            configurator = workerdata.pop("configurator")
        else:
            from experiments import lf_placecells
            configurator = lf_placecells.configure_training
        configurator(self)

        # parse some gridsearch parameters to overwrite configurator
        if workerdata:
            for (key, value) in self.workerdata.items():
                if hasattr(gv, key):
                    setattr(gv, key, value)
                elif key == "vq_lr_int":
                    gv.vq_learning_scale = list([0, 10 ** -4, 10 ** -3, 10 ** -2])[int(value)]
                elif key == "vq_decay_int":
                    gv.vq_decay = list([0, 10 ** -4, 10 ** -3, 10 ** -2])[int(value)]
                else:
                    print("unknown gridsearch hyperparameter " + key)

        # training for pole
        try:
            async with websockets.connect(ws_uri) as websocket:
                lastsend = websocket.send("sender")
                await self.train(websocket, lastsend)
        except ConnectionRefusedError:
            await self.train()
        except ConnectionError:
            await self.train()
        except OSError:
            await self.train()
        self.post_train()

        return self.return_per_episode_sum

    def show_current(self):
        """Shows one episode and stops training for this episode."""
        global gv
        gv_old = copy.deepcopy(gv)
        gv.errsig_factor = 0.
        gv.structural_plasticity = False
        gv.render = True
        gv.demo = True
        self.agent.pre_episode()
        self.simulate_episode()
        gv = gv_old


async def runworker(dataperworker: Optional[Dict]) -> List[float]:
    """
    Set up a worker (process) and run an experiment.
    :param dataperworker:
    :return:
    """
    # redundant copy of method because the gridsearch returns validation errors
    # there was a crash when db was disabled with a gridsearch pool
    # this cannot be a local function bedause it will cause a crash"
    return await Experiment().run(dataperworker)


def gridsearch(num_processes: int, training, configurator: Callable) -> List:
    """perform a gridsearcg on the giving trainingdata """
    pool = multiprocessing.Pool(num_processes)
    withoutgivenvalues = filter(lambda v: "from" in v, training.gridsearch.values())
    # todo insert ranges in gridsearch
    withgivenvalues = filter(lambda v: "range" in v, training.gridsearch.values())
    parameters: List[slice] = [slice(rangedetails["from"], rangedetails["to"], complex(rangedetails["steps"])) for
                               rangedetails in withoutgivenvalues]
    rangesgridsearch: np.array = np.mgrid[parameters].reshape(len(parameters), -1).T
    # put in an array containg the parameters per worker
    workload: List[Dict] = []
    paramnameslist = list(training.gridsearch.keys())
    for workerdata in rangesgridsearch:
        # each gets a training reference
        obj = {"training": training}
        if configurator is not None:
            obj["configurator"] = configurator
        for paramidx, param in enumerate(workerdata):
            obj[paramnameslist[paramidx]] = param
        workload.append(obj)
    result = pool.map(func=runworker, iterable=workload)
    pool.close()
    pool.join()

    if len(parameters) == 2:
        numcolums = list(training.gridsearch.values())[0]["steps"]
        resultnp = np.array(result).reshape((-1, numcolums))
        table = "\\begin{center}\\begin{tabular}{ | l | l | l | l | l |}\\hline\n"
        table += "num.~cells & $\\lambda =0$ (no vq) & $\\lambda =0.0001$ & $\\lambda =0.001$ & $\\lambda =0.01$"
        for i, resultitem in enumerate(result):
            if i % numcolums == 0:
                table += "\\\\ \\hline\n"
                table += str(int(i / numcolums))  # row name

            table += f" & {np.average(resultitem):.0f}"
        table += "\\\\ \\hline\n"
        table += "\\end{tabular}\\end{center}"
        print(table)
    return result


dirname = ""


def exit_handler():
    os.chdir("../../")
    global dirname
    if len(os.listdir(dirname)) == 0:
        os.rmdir(dirname)


def createexpdatadir():
    """create new directory for test results and switches to it"""
    counter = 0
    dirbase = "experimentdata/gsrewardsignal"
    global dirname
    dirname = dirbase + str(counter)
    while os.path.isdir(dirname):
        counter += 1
        dirname = dirbase + str(counter)
    os.makedirs(dirname)
    os.chdir(dirname)
    print(f"saving to {dirname}\n")

    atexit.register(exit_handler)


def dbconnect():
    mongoengine.connect(
        db='snntrainings',
        username='',
        password='',
        port=45920,
        authentication_source='admin',
        host=''
    )


async def trainingrun(configurator: Callable = None, num_processes: int = 1, gridsearchpath: str = None) -> Tuple[
    Union[None, Experiment], List]:
    """
    Creates an experiemnt and runs it.
    :param configurator:
    :param num_processes:
    :param gridsearchpath:
    :return: if a single experiment returns this. None if gridsearch.
    """
    training = tg.Trainingrun()

    training.time_start = datetime.datetime.utcnow()
    if gv.save_to_db:
        dbconnect()
        training.save()
        print(f"💾DB Trainingrun: ObjectId(\"{training.id}\")")
        disconnect()

    # start websocket server
    # try connecting to webserv, if it fails spawn a new webserver
    loop = asyncio.new_event_loop()
    import websockets
    async def hello():
        try:
            async with websockets.connect(ws_uri) as websocket:
                await websocket.send("Hello world!")
                await websocket.recv()
            global wssession
            wssession = websockets.connect(ws_uri)
        except Exception:
            print("Spawn new websocket server")
            p = Process(target=initiate_server, args=())
            p.daemon = False  # keep process alive
            p.start()
            print("Killing websocket server")

            # when the main process is killed it kills the child process
            # using subprocess should prevent that https://izziswift.com/how-to-start-a-background-process-in-python/
            # p.join()

    #asyncio.get_event_loop().run_until_complete(hello())

    # t1 = Thread(target=initiate_server, args=(loop,))
    # t1.start()

    # if gridsearch
    singleexp = None
    if gridsearchpath is not None:
        with open(gridsearchpath, "r") as file:
            training.gridsearch = json.loads(file.read())
        createexpdatadir()
        result = gridsearch(num_processes, training, configurator)
    else:
        # if not a gridsearch
        createexpdatadir()
        datasingleworker = {"training": training} if configurator is None else {"training": training,
                                                                                "configurator": configurator}
        singleexp = Experiment()
        result = await singleexp.run(datasingleworker)

    training.time_end = datetime.datetime.utcnow()
    training.time_elapsed = (training.time_end - training.time_start).total_seconds()
    if gv.save_to_db:
        dbconnect()
        training.save()
        disconnect()

    print(f"{training.time_elapsed / 60:10.1f} min")

    return singleexp, result

    #     plt.plot(gv.outactivity["out1"], label="ouput 0")
    #     plt.plot(gv.outactivity["out2"], label="ouput 1")
    #     #plt.plot(np.array(gv.outactivity["action"])*30, label="action")
    #     plt.plot(gv.outactivity["in1"], label="input 1")
    #     plt.plot(gv.outactivity["in2"], label="input 2")
    # #    plt.plot(np.array(exp.outactivity[2])*80, label="utility")
    # #    plt.plot(exp.utilitycontingent, label="used utility")
    #     for xc in exp.epslength:
    #         plt.axvline(x=xc, color='k')
    #     plt.title("Experiment")
    #     plt.xlabel("cycle")
    #     plt.legend()
    #     plt.show()

    # for exp in exps:
    #      exp.join()


def parseargs():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--processes', type=int, default=multiprocessing.cpu_count(),
                        help='The number of cores. Currently only supporting multi-cores in grid search.')
    parser.add_argument('-g', '--gridsearch', type=str, default=None, help='json  specifing grid search parameter')
    parser.add_argument('--headless', action='store_true', help='Do not render.')
    args = parser.parse_args()
    gv.headless = args.headless
    if gv.headless:
        gv.render = False
    return args


if __name__ == "__main__":
    args = parseargs()
    trainingrun(num_processes=args.processes, gridsearchpath=args.gridsearch)
