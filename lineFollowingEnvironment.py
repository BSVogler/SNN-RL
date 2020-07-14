from typing import List, Any

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

from globalvalues import gv


class LineFollowingEnv(gym.Env):
    """
    Description:
        A one dimensional lien following task.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(2)
        Num	Observation                 Min         Max
        0	Current Position            0            width
        1	Next step line position      0           width

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, absolute_observation=False):
        self.absolute_observation = absolute_observation
        self.tracklength = 100
        self.width = 3.0
        self.track = np.sin(np.linspace(0, 2 * np.pi, self.tracklength)) + self.width / 2
        self.trackPiece: List[Any] = [None] * self.tracklength
        self.bound = []
        self.botpos = [0,0]

        self.action_space = spaces.Box(np.array([0.0]), np.array([1]))
        self.observation_space = spaces.Box(np.array([0.0, 0.0]), np.array([self.width, self.width]), dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.previouspath = []
        self.steps_beyond_done = None
        self.reset()

    def update_state(self):
        self.state = (self.botpos[0], self.track[(self.botpos[1] + 1) % self.tracklength])
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.previouspath[-1].append(self.botpos[0])
        self.botpos[0] += action[0] - 0.5
        self.update_state()
        reward = self.width / 4 - abs(self.botpos[0] - self.track[self.botpos[1]])  # surviving is positive
        self.botpos[1] += 1
        self.botpos[1] %= self.tracklength
        done = reward <= 0

        if done:
            #clamp so that every failing state is equal regarding reward
            reward = 0
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            if self.steps_beyond_done is not None:
                self.steps_beyond_done += 1
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0

        # relative
        if self.absolute_observation:
            return np.array(self.state), reward, done, {}
        else:
            return np.array([self.state[0] - self.state[1]]), reward, done, {}

    def reset(self):
        """Reset the enviroment. Returns initial reward"""
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.previouspath.append([])
        self.steps_beyond_done = None
        self.botpos = [self.width * gv.pyrngs[0].uniform(0.7, 0.74), 0]  # x float, y index # middle is self.width / 2
        self.update_state()
        initalreward = self.width / 4 - abs(self.botpos[0] - self.track[self.botpos[1]])  # surviving is positive
        if self.absolute_observation:
            return np.array(self.state), initalreward
        else:
            return np.array([self.state[0] - self.state[1]]), initalreward

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400
        stepHeight = screen_height / float(self.tracklength)
        unitLength = screen_width / self.width / 2
        offsety = 10
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            cartwidth = 4.0
            cartheight = 4.0
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            # track
            self.trackTrans = rendering.Transform()
            for i in range(0, self.tracklength):
                self.trackPiece[i] = rendering.Line((self.track[i] * unitLength, i * stepHeight),
                                                    (self.track[(i + 1) % self.tracklength] * unitLength,
                                                     (i + 1) * stepHeight))
                # self.trackPiece[i] = rendering.Line((0, 30), (10, 40))
                self.trackPiece[i].add_attr(self.trackTrans)
                self.trackPiece[i].set_color(.5, .5, .8)
                self.viewer.add_geom(self.trackPiece[i])
            # bound left
            for i in range(0, self.tracklength):
                newPiece = rendering.Line(((self.track[i] + self.width / 2) * unitLength, i * stepHeight),
                                          ((self.track[(i + 1) % self.tracklength] + self.width / 4) * unitLength,
                                           (i + 1) * stepHeight))
                newPiece.add_attr(self.trackTrans)
                newPiece.set_color(.5, .0, .0)
                self.bound.append(newPiece)
                self.viewer.add_geom(newPiece)
            # bound right
            for i in range(0, self.tracklength):
                newPiece = rendering.Line(((self.track[i] - self.width / 2) * unitLength, i * stepHeight),
                                          ((self.track[(i + 1) % self.tracklength] - self.width / 4) * unitLength,
                                           (i + 1) * stepHeight))
                newPiece.add_attr(self.trackTrans)
                newPiece.set_color(.5, .0, .0)
                self.bound.append(newPiece)
                self.viewer.add_geom(newPiece)

        if self.state is None: return None

        for i, trial in enumerate(self.previouspath[-20:]):
            grayvalue = 1 - float(i) / len(self.previouspath[-20:])
            color = (grayvalue, grayvalue, grayvalue)
            for i in range(1, len(trial)):
                self.viewer.draw_line((trial[i - 1] * unitLength, (i - 1) * stepHeight + offsety),
                                      (trial[i] * unitLength, i * stepHeight + offsety),
                                      color=color)

        cartx = self.botpos[0] * unitLength  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, self.botpos[1] * stepHeight + offsety)
        self.trackTrans.set_translation(0, 0 + offsety)
        # self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
