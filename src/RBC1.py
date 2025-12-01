from __future__ import annotations
from .template import Templates
from .solver_xi import SolverXi
from .utils import sample_data, sample_disturbance
from .plot_manager import PlotManager
from .utils import sample_data
from tqdm import tqdm
import numpy as np
import math
import time
import warnings


class RBC1:
    def __init__(self, model: callable):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        self.model = model

        self.templates = None
        self.solver = None
        self.verbose = None
        self.alpha1 = 0
        self.delta = 0
        self.alpha2 = 0
        self.plot_manager = None
        self.traj = None
        self.coe = None
        self.lamda = None
        self.N = 0
        self.C = 1
        self.batch_size = None

    def set_options(self, **kwargs):
        self.verbose = kwargs.get("verbose", 0)

        templ_type = kwargs.get("template_type", "handelman")
        degree = kwargs.get("degree", 6)
        x_lb = kwargs.get("x_lb", None)
        x_ub = kwargs.get("x_ub", None)

        self.templates = Templates(degree_systems=self.model.degree_state, temp_type=templ_type,
                                   degree=degree, x_lb=x_lb, x_ub=x_ub, verbose=self.verbose)

        num_vars = self.templates.num_vars + 1
        # print(f"num_vars: {num_vars}")

        coe_b = kwargs.get("coe_b", 10)
        assert coe_b > 2
        self.solver = SolverXi(num_vars=num_vars, coe_lb=0, coe_ub=coe_b, verbose=self.verbose)

        alpha1 = kwargs.get("alpha1", None)
        alpha2 = kwargs.get("alpha2", None)
        delta = kwargs.get("delta", None)
        N = kwargs.get("N", None)
        if alpha1 is not None and alpha2 is not None and delta is not None and N is not None:
            assert N >= (2 / (alpha1 * alpha2) * (math.log(1 / delta) + num_vars))
        elif alpha1 is not None and alpha2 is not None and delta is not None and N is None:
            N = 2 * (math.log(1 / delta) + num_vars) / (alpha1 * alpha2)
            N = math.ceil(N)
            print("N: ", N)
        elif alpha1 is None and alpha2 is None and delta is not None and N is not None:
            alpha = 2 / N * (math.log(1 / delta) + num_vars)
            print("alpha: ", alpha)
        else:
            raise ValueError("At least two of alpha, delta, and N must be entered!")
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.delta = delta
        self.N = N

        self.C = kwargs.get("C", -100)
        self.lamda = kwargs.get("lamda", 0.01)

        self.batch_size = kwargs.get("batch_size", None)

        random_seed = kwargs.get("random_seed", False)
        if random_seed is not False:
            np.random.seed(random_seed)

        plot_dim = kwargs.get("plot_dim", [[0, 1]])
        plot_project_values = kwargs.get("plot_project_values", {})
        self.plot_manager = PlotManager(dim=plot_dim, project_values=plot_project_values, grid=False,
                                        v_filled=True,
                                        save=True, prob_name=self.model.__class__.__name__, save_file='jpg')

    def solve(self, safe_set):
        start_time = time.time()

        x_data = sample_data.sample_data(self.N, safe_set, "random")
        d = sample_disturbance.sample_disturbance(self.N, distribution=self.model.disturbance_type,
                                                  d_para=self.model.d_para)

        h_x = self.templates.calc_values(x_data)

        h_fx = []
        constant = []

        fx = np.array(self.model.fx(x_data, d, None)).T
        is_in_safe_set = sample_data.get_is_in_safe_set(fx, safe_set)

        h_fx = self.templates.calc_values(fx)

        h_unsafe = np.zeros((h_fx.shape[1],))
        h_fx[~is_in_safe_set] = h_unsafe

        constant = (1 - is_in_safe_set.astype(int)) * self.C

        constraint = self.lamda * h_x - h_fx        # <=
        constraint = np.insert(constraint, 0, -1, axis=1)
        self.solver.add_constraint(constraint, constant)

        solution = self.solver.solve()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The time required for solution is {elapsed_time} seconds.")

        self.coe = solution

        return self.coe

    def get_h_value(self, x_data):
        h_x_mono = self.templates.calc_values(x_data)
        h_x_value = np.dot(h_x_mono, self.coe)
        return h_x_value

    def sim_traj(self, init_state, sim_time):
        if self.model.degree_state != 2:
            raise NotImplementedError()
        traj_all = []
        for i in range(len(init_state)):
            traj = [init_state[i]]
            x = np.array(init_state[i]).reshape(1, -1)
            for j in range(sim_time[i]):
                d = np.random.uniform(low=self.model.d_lb, high=self.model.d_ub, size=(x.shape[0], self.model.d_lb.shape[0]))
                x = self.model.fx(x, d)
                x = np.array(x)
                x = x.reshape(2)
                traj.append(x)
                x = x.reshape(1, -1)

                print(x)
                print(self.get_h_value(x))
            traj_all.append(traj)
        self.traj = traj_all
