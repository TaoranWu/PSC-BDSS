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


class RBC2:
    def __init__(self, model: callable):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        self.model = model

        self.templates = None
        self.solver = None
        self.verbose = None
        self.alpha1 = 0
        self.delta1 = 0
        self.alpha2 = 0
        self.delta2 = 0
        self.plot_manager = None
        self.traj = None
        self.coe = None
        self.lamda = None
        self.N = 0
        self.M = 0
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

        coe_b = kwargs.get("coe_b", 100)
        assert coe_b > 2
        self.solver = SolverXi(num_vars=num_vars, coe_lb=0, coe_ub=coe_b, verbose=self.verbose)

        alpha1 = kwargs.get("alpha1", None)
        delta1 = kwargs.get("delta1", None)
        N = kwargs.get("N", None)
        if alpha1 is not None and delta1 is not None and N is not None:
            assert alpha1 >= (2 / N * (math.log(1 / delta1) + num_vars))
        elif alpha1 is not None and delta1 is not None and N is None:
            N = 2 * (math.log(1 / delta1) + num_vars) / alpha1
            N = math.ceil(N)
            print("N: ", N)
        else:
            raise ValueError("At least two of alpha, beta, and N must be entered!")
        self.alpha1 = alpha1
        self.delta1 = delta1
        self.N = N

        alpha2 = kwargs.get("alpha2", None)
        delta2 = kwargs.get("delta2", None)
        l = kwargs.get("l", None)
        M = kwargs.get("M", None)
        if alpha2 is not None and delta2 is not None and M is not None:
            assert M >= math.log(1/((1-l)*delta2)) / (2*alpha2**2)
        elif alpha2 is not None and delta2 is not None and M is None:
            M = math.log(1/((1-l)*delta2)) / (2*alpha2**2)
            M = math.ceil(M)
            print("M: ", M)
        elif alpha2 is None and delta2 is not None and M is not None:
            alpha2 = math.sqrt(math.log(1/((1-l)*delta2)) / (2*M))
            print("alpha2: ", alpha2)
        elif alpha2 is not None and delta2 is None and M is not None:
            delta2 = 1 / (math.exp(2*M*alpha2**2) * (1-l))
            print("delta2: ", delta2)
        else:
            raise ValueError("At least two of alpha2, delta2, and N must be entered!")
        self.alpha2 = alpha2
        self.delta2 = delta2
        self.M = M
        print("N*M: ", N*M)

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

        h_x = self.templates.calc_values(x_data)

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = self.N

        h_fx = []
        constant = []

        num_iterations = (self.N + batch_size - 1) // batch_size

        for start in tqdm(range(0, self.N, batch_size), total=num_iterations):
            end = min(start + batch_size, self.N)
            x_batch = x_data[start:end]

            x_batch_repeat = np.repeat(x_batch, repeats=self.M, axis=0)

            d = sample_disturbance.sample_disturbance(self.M * x_batch.shape[0], distribution=self.model.disturbance_type,
                                                      d_para=self.model.d_para)
            # print("max_d: ", max(d))
            # print("min_d: ", min(d))

            fx_batch = np.array(self.model.fx(x_batch_repeat, d, None)).T

            is_in_safe_set_batch = sample_data.get_is_in_safe_set(fx_batch, safe_set)

            h_fx_batch = self.templates.calc_values(fx_batch)

            h_unsafe = np.zeros((h_fx_batch.shape[1],))
            h_fx_batch[~is_in_safe_set_batch] = h_unsafe

            constants_batch = (1 - is_in_safe_set_batch.astype(int)) * self.C

            h_fx.append(h_fx_batch)
            constant.append(constants_batch)

        h_x = np.repeat(h_x, repeats=self.M, axis=0)
        h_fx = np.concatenate(h_fx, axis=0)
        constant = np.concatenate(constant, axis=0)

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
