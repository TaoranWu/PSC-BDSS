from src.model.model import Model
import numpy as np


class Arch3SafeST(Model):
    def __init__(self):
        super().__init__()
        self.degree_state = 2
        self.degree_disturbance = 2
        self.disturbance_type = "uniform"
        self.d_para = {'lower_bound': np.array([-0.5, -0.5]),
                       'upper_bound': np.array([0.5, 0.5])}

    @staticmethod
    def fx(x, d, u=None):
        x0 = x[:, 0]
        x1 = x[:, 1]
        d0 = d[:, 0]
        d1 = d[:, 1]

        x_next = [None] * 2

        step = 0.01
        x_next[0] = x0 + step * (x0 - x0**3 + x1 - x0*x1**2 + d0)
        x_next[1] = x1 + step * (-x0 + x1 - x0**2*x1 - x1**3 + d1)

        return x_next




