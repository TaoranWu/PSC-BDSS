from src.model.model import Model
import numpy as np


class N2SafeST(Model):
    def __init__(self):
        super().__init__()
        self.degree_state = 2
        self.degree_disturbance = 1
        self.disturbance_type = "truncated_normal"
        self.d_para = {'lower_bound': np.array([-0.7]),
                       'upper_bound': np.array([0.7]),
                       'mean': 0.0,
                       'variance': 0.1**2}

    @staticmethod
    def fx(x, d, u=None):
        x0 = x[:, 0]
        x1 = x[:, 1]
        d0 = d[:, 0]

        x_next = [None] * 2

        step = 0.01
        x_next[0] = x0 + step * (x1 - x0 * (d0+0.5))
        x_next[1] = x1 + step * (-(1-x0**2) * x0 - x1)

        return x_next
