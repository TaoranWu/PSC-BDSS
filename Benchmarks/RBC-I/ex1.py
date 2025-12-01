from src.model import *
from src.RBC1 import RBC1
from src.utils.sample_data import *

if __name__ == '__main__':
    model = N2SafeST()
    prob = RBC1(model)

    alpha1, alpha2 = 0.05, 0.05
    delta = 1e-6
    x_lb = np.array([-0.8, -0.8])
    x_ub = np.array([0.8, 0.8])
    safe_set = Ellipsoid([0, 0], [1/0.64, 1/0.64])
    prob.set_options(degree=1, x_lb=x_lb, x_ub=x_ub,
                     alpha1=alpha1, alpha2=alpha2,
                     delta=delta,
                     lamda=0.01,
                     C=-1, coe_b=10,
                     batch_size=50,
                     random_seed=0)

    h1 = prob.solve(safe_set)





