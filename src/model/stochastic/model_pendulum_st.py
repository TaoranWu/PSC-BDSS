from src.model.model import Model
import numpy as np


class PendulumST(Model):
    def __init__(self):
        super().__init__()
        self.degree_state = 2
        self.degree_disturbance = 1
        self.disturbance_type = "uniform"
        self.d_para = {'lower_bound': np.array([0.9]),
                       'upper_bound': np.array([1.1])}


    @staticmethod
    def fx(x, d, u=None):
        x0 = x[:, 0]
        x1 = x[:, 1]
        d0 = d[:, 0]

        x_next = [None] * 2

        step = 0.1
        x_next[0] = x0 + step * x1
        x_next[1] = x1 + step * (-2*x1/d0+0.81*np.sin(x0)*np.cos(x0)-np.sin(x0))

        return x_next
