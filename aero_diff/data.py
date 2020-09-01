import math
import torch


class Panel:
    """
    Contains information related to a panel.
    """

    def __init__(self, point_a, point_b):

        self.point_a, self.point_b = point_a, point_b
        self.point_c = (point_a + point_b) / 2
        self.length = torch.dist(self.point_a, self.point_b)

        # orientation of the panel (angle between x-axis and panel's normal)
        diff = point_b - point_a
        if diff[0] > 0:
            self.beta = math.pi + torch.acos(- diff[1] / self.length)
        else:
            self.beta = torch.acos(diff[1] / self.length)

        self.sigma = 0.0  # source strength
        self.vt = 0.0  # tangential velocity
        self.cp = 0.0  # pressure coefficient


class Mesh:

    def __init__(self, points_a, points_b):
        self.points_a, self.points_b = points_a, points_b
        self.points_c = (points_a + points_b) / 2

        self.lens = torch.sqrt((points_b[:, 0] - points_a[:, 0]) ** 2 +
                               (points_b[:, 1] - points_a[:, 1]) ** 2)

        diffs = points_b - points_a

        self.betas = torch.where(diffs[:, 0] > 0,
                                 math.pi + torch.acos(- diffs[:, 1] / self.lens),
                                 torch.acos(diffs[:, 1] / self.lens))

        self.sigma = 0.0  # source strength
        self.vt = 0.0  # tangential velocity
        self.cp = 0.0  # pressure coefficient
