import math
import unittest
import numpy as np
from gym.spaces import Box

from critic import DynamicBaseline


class MyTestCase(unittest.TestCase):

    def test_tick(self):
        # test two dimensional states
        util = DynamicBaseline(obsranges=np.array([[0.0, 0.0], [3.0, 3.0]]))
        errsig, utilValue = util.tick((0.0, 3.0), (1.0, 1.0))
        self.assertAlmostEqual(utilValue, 1.0)
        errsig, a = util.tick((0.0, 2.0), (0.0, 0.0))
        errsig, b = util.tick((0.0, 2.0), (3.0, 3.0))
        self.assertAlmostEqual(a, 0)
        self.assertAlmostEqual(b, 3)  # should return last value

    def test_drawrewards(self):
        util = DynamicBaseline(obsranges=np.array([[0.0, 3.4], [0.0, 5.0], [2., 3.]]))
        util.tick((0.0, 3.0, 2.),  (1.0, 4.0))
        util.tick((2.0, 0.3, 3.),  (2.0, 3.0))
        util.tick((1.0, 2.0, 2.),  (1.0, 2.))
        util.tick((-4.0, 2.0, 2.), (3.4, 1.0))
        util.tick((1.3, 1.2, 2.),  (-2.0, 2.))
        util.tick((-4.0, 2.0, 2.), (3.4, 1.0))
        util.tick((3.32, 1.2, 2.), (-2.0, 2.))
        util.tick((-4.0, 1.52, 3.),(2.24, 2.))
        util.tick((2.34, 1.4, 2.), (-1.4, 2.))
        util.post_episode()
        util.draw_rewards(xaxis=0, yaxis=1)

    def test_draw(self):
        util = DynamicBaseline(obsranges=np.array([[0.0, 2.0], [0.0, 1.0]]))
        for x in range(0, 40):
            util.tick((x / 30.0, math.sin(x / 20) * 1.0), (2 * x))
        util.post_episode()
        util.draw(xaxis=0, yaxis=1)


if __name__ == '__main__':
    unittest.main()
