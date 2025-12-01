from src.model.model import Model
import numpy as np


class StabilizationSafeST(Model):
    def __init__(self):
        super().__init__()
        self.degree_state = 3
        self.degree_disturbance = 3
        self.disturbance_type = "uniform"
        self.d_para = {'lower_bound': np.array([1.0, 1.0, 2.0]),
                       'upper_bound': np.array([2.0, 2.0, 3.0])}

    @staticmethod
    def fx(x, d, u=None):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        d0 = d[:, 0]
        d1 = d[:, 1]
        d2 = d[:, 2]

        x_next = [None] * 3

        step = 0.01
        x_next[0] = x0 + step * (-x0 + x1 - x2 - x0 * d0)
        x_next[1] = x1 + step * (-x0 * (x2 + 1) - x1 - x1 * d1)
        x_next[2] = x2 + step * (0.76524*x0 - 4.7037 * x2 - x2 * d2)

        return x_next




