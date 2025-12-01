from src.model.model import Model
import numpy as np


class LorenzST7d(Model):
    def __init__(self):
        super().__init__()
        self.degree_state = 7
        self.degree_input = 0
        self.degree_disturbance = 7
        self.disturbance_type = "uniform"
        self.d_para = {'lower_bound': np.array([-1] * 7),
                       'upper_bound': np.array([1] * 7)}
    @staticmethod
    def fx(x, d, u=None):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x3 = x[:, 3]
        x4 = x[:, 4]
        x5 = x[:, 5]
        x6 = x[:, 6]

        d0 = d[:, 0]
        d1 = d[:, 1]
        d2 = d[:, 2]
        d3 = d[:, 3]
        d4 = d[:, 4]
        d5 = d[:, 5]
        d6 = d[:, 6]

        x_next = [None] * 7
        step = 0.01

        x_next[0] = x0 + step * ((x1 - x5) * x6 - x0 + d0)
        x_next[1] = x1 + step * ((x2 - x6) * x0 - x1 + d1)
        x_next[2] = x2 + step * ((x3 - x0) * x1 - x2 + d2)
        x_next[3] = x3 + step * ((x4 - x1) * x2 - x3 + d3)
        x_next[4] = x4 + step * ((x5 - x2) * x3 - x4 + d4)
        x_next[5] = x5 + step * ((x6 - x3) * x4 - x5 + d5)
        x_next[6] = x6 + step * ((x0 - x4) * x5 - x6 + d6)

        return x_next


