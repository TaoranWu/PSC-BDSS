from src.model.model import Model
import numpy as np


class SankST(Model):
    def __init__(self):
        super().__init__()
        self.degree_state = 4
        self.degree_disturbance = 1
        self.disturbance_type = "uniform"
        self.d_para = {'lower_bound': np.array([-1]),
                       'upper_bound': np.array([1])}

    @staticmethod
    def fx(x, d, u=None):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x3 = x[:, 3]
        d0 = d[:, 0]

        x_next = [None] * 4

        step = 0.01
        x_next[0] = x0 + step * (-x0+x1**3-3*x2*x3+d0)
        x_next[1] = x1 + step * (-x0 - x1**3)
        x_next[2] = x2 + step * (x0*x3 - x2)
        x_next[3] = x3 + step * (x0*x2 - x3**3)

        return x_next
