from src.model import *
from src.RBC2 import RBC2
from src.utils.sample_data import *

if __name__ == '__main__':
    model = LorenzST7d()
    prob = RBC2(model)

    alpha1, delta1 = 0.01, 1e-6
    alpha2, delta2 = 0.05, 0.999
    l = 0.2
    print(alpha1)
    print("1- alpha1 / (l*delta2) = ", 1 - alpha1 / (l * delta2))

    x_lb = np.array([-1] * 7)
    x_ub = np.array([1] * 7)
    safe_set = Interval([-1] * 7, [1] * 7)
    prob.set_options(degree=1, x_lb=x_lb, x_ub=x_ub,
                     alpha1=alpha1, delta1=delta1,
                     alpha2=alpha2, delta2=delta2,
                     lamda=0.01,
                     l=l,
                     C=-1, coe_b=10,
                     batch_size=10,
                     random_seed=0)

    h1 = prob.solve(safe_set)





