import math
from typing import List, Any

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

from settings import gv


class LineFollowingEnv2(gym.Env):
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
        'video.frames_per_second': int(1000 / gv.cycle_length)
    }

    segments = 12

    def __init__(self, absolute_observation=False):
        self.absolute_observation = absolute_observation
        self.tracklength = 600
        self.track_width = 3.0

        self.track: List[float] = [0.0] * self.tracklength
        self.trackPiece: List[Any] = [None] * self.tracklength
        self.bound = []
        self.botpos = [0, 0.0]  # x, y

        self.action_space = spaces.Box(np.array([0.0]), np.array([1]))
        self.observation_space = spaces.Box(np.array([0.0, 0.0]), np.array([self.track_width, self.track_width]),
                                            dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.previouspath = []
        self.steps_beyond_done = None
        self.grid_from = math.floor(-self.track_width / 2)
        self.grid_to = math.ceil(self.track_width / 2)
        self.grid_res = 0
        self.grid = []
        self.reset()

    def enable_grid(self):
        self.grid_res = math.ceil(abs(self.grid_from - self.grid_to))  # count of lines

    def update_state(self):
        # current pos, next track pos
        self.state = (self.botpos[1], self.track[(self.botpos[0] + 1) % self.tracklength])
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: list[float]):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.previouspath[-1].append(self.botpos[1])
        self.botpos[1] += action[0] - 0.5
        self.update_state()
        reward = self.track_width - abs(self.botpos[1] - self.track[self.botpos[0]])  # surviving is positive
        self.botpos[0] += 1
        self.botpos[0] %= self.tracklength
        done = reward <= 0

        # stop if not visible any more
        if abs(self.botpos[1]) >= self.track_width / 2:
            done = True

        if done:
            # clamp so that every failing state is equal regarding reward
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
        # create new path
        # track is between 0 and width
        seglength = int(self.tracklength // LineFollowingEnv2.segments)
        randomvar = self.track_width * 2 / 3.0
        for seg in range(LineFollowingEnv2.segments):
            random = np.random.random_sample() * randomvar - randomvar / 2  # centered at 0.

            for i in range(seg * seglength, (seg + 1) * seglength):
                self.track[i] = random

        self.previouspath.append([])
        self.steps_beyond_done = None
        self.botpos = [0, gv.pyrngs[0].uniform(0,
                                               self.track_width / 2) - self.track_width / 4]  # x index, y float # middle is self.width / 2
        self.update_state()
        if self.absolute_observation:
            return np.array(self.state)
        else:
            return np.array([self.state[0] - self.state[1]])

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400
        unit_height = screen_height / self.track_width
        unit_length = screen_width / float(self.tracklength)
        centerY = screen_height / 2
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # grid
            if self.grid_res > 0:
                gridstep = (self.grid_to - self.grid_from) / self.grid_res * unit_height
                self.grid.append(rendering.Line((0, centerY),
                                                (screen_width, centerY)))
                self.grid[0].set_color(.0, .9, .0)
                self.viewer.add_geom(self.grid[0])
                for y in range(self.grid_res):
                    ypos = y * gridstep + self.grid_from * unit_height
                    self.grid.append(rendering.Line((0, ypos + centerY),
                                                    (screen_width, ypos + centerY)))
                    self.grid[y + 1].set_color(.8, .2, .2)
                    self.viewer.add_geom(self.grid[y + 1])

            # add cart
            cartwidth = 4.0
            cartheight = 4.0
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            # bound left
            # for i in range(0, self.tracklength):
            #     newPiece = rendering.Line(((self.track[i] + self.width / 2) * unitLength, i * unit_height),
            #                               ((self.track[(i + 1) % self.tracklength] + self.width / 4) * unitLength,
            #                                (i + 1) * unit_height))
            #     newPiece.add_attr(self.trackTrans)
            #     newPiece.set_color(.5, .0, .0)
            #     self.bound.append(newPiece)
            #     self.viewer.add_geom(newPiece)
            # # bound right
            # for i in range(0, self.tracklength):
            #     newPiece = rendering.Line(((self.track[i] - self.width / 2) * unitLength, i * unit_height),
            #                               ((self.track[(i + 1) % self.tracklength] - self.width / 4) * unitLength,
            #                                (i + 1) * unit_height))
            #     newPiece.add_attr(self.trackTrans)
            #     newPiece.set_color(.5, .0, .0)
            #     self.bound.append(newPiece)
            #     self.viewer.add_geom(newPiece)

        if self.state is None:
            return None

        # track
        color = (.5, .5, .8)
        for i in range(0, self.tracklength):
            self.viewer.draw_line((i * unit_length,
                                   +self.track[i] * unit_height + centerY),
                                  ((i + 1) * unit_length,
                                   +self.track[(i + 1) % self.tracklength] * unit_height + centerY),
                                  color=color)

        # trail
        for i, trial in enumerate(self.previouspath[-20:]):
            grayvalue = 1 - float(i) / len(self.previouspath[-20:])
            color = (grayvalue, grayvalue, grayvalue)
            for i in range(1, len(trial)):
                self.viewer.draw_line(((i - 1) * unit_length, trial[i - 1] * unit_height + centerY),
                                      (i * unit_length, trial[i] * unit_height + centerY),
                                      color=color)

        cartx = self.botpos[0] * unit_length  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, self.botpos[1] * unit_height + centerY)
        # self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
