"""
Environment for rrt_2D
@author: huiming zhou
"""


class Env:
    def __init__(self):
        self.x_range = (-10, 60)
        self.y_range = (-10, 60)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()

    @staticmethod
    def obs_boundary():
        obs_boundary = [
            [-10, -10, .1, 70],
            [-10, 60, 70, .1],
            [-10, -10, 70, .0],
            [60, -10, .1, 70],

        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        return [[20, -10, .1, 40],
                [40, 20, .1, 40]]

    @staticmethod
    def obs_circle():
        return []