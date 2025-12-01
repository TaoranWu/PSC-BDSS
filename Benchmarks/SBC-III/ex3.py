from src.model.stochastic import *
from src.SBC3 import SBC3
from src.utils.sample_data import *

if __name__ == '__main__':
    model = StabilizationSafeST()
    prob = SBC3(model)

    alpha1 = 0.01
    delta1, delta2 = 1e-6, 0.999
    l = 0.2
    print("1- alpha1 / (l*delta2) = ", 1 - alpha1 / (l * delta2))
    random_seed = 0
    x_lb = np.array([-1, -1, -1])
    x_ub = np.array([1, 1, 1])
    safe_set = Interval([-1, -1, -1], [1, 1, 1])
    prob.set_options(degree=1, x_lb=x_lb, x_ub=x_ub,
                     alpha1=alpha1, delta1=delta1, U_a=1.1, l=l,
                     delta2=delta2,
                     tau=0.01,
                     batch_size=100,
                     random_seed=random_seed)

    obj_x = sample_obj_data(1e3, safe_set, method="random")

    h1, lamda = prob.solve(obj_x, safe_set)
    print(lamda)
