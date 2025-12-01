from __future__ import annotations
from .template import Templates
from .solver_lamda import SolverLamda
from .plot_manager import PlotManager
from .utils import sample_data, sample_disturbance
from tqdm import tqdm
import numpy as np
import math
import time
import warnings


class SBC3:
    def __init__(self, model: callable):
        self.tau = None
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        self.model = model

        self.templates = None
        self.solver = None
        self.verbose = None
        self.alpha1 = 0
        self.delta1 = 0
        self.delta2 = 0
        self.plot_manager = None
        self.traj = None
        self.coe = None
        self.N_o = 0
        self.N = 0
        self.M = 0
        self.C = 1
        self.obj_sample = None
        self.batch_size = 50

    def set_options(self, **kwargs):
        self.verbose = kwargs.get("verbose", 1)

        templ_type = kwargs.get("template_type", "bernstein")
        degree = kwargs.get("degree", None)
        x_lb = kwargs.get("x_lb", None)
        x_ub = kwargs.get("x_ub", None)

        self.templates = Templates(degree_systems=self.model.degree_state, temp_type=templ_type,
                                   degree=degree, x_lb=x_lb, x_ub=x_ub, verbose=self.verbose)

        num_vars = self.templates.num_vars + 1
        # print(f"num_vars: {num_vars}")

        U_a = kwargs.get("U_a", None)

        self.solver = SolverLamda(num_vars=num_vars, coe_lb=0, coe_ub=U_a)

        alpha1 = kwargs.get("alpha1", None)
        delta1 = kwargs.get("delta1", None)
        N = kwargs.get("N", None)
        if alpha1 is not None and delta1 is not None and N is not None:
            assert alpha1 >= (2 / N * (math.log(1 / delta1) + num_vars))
        elif alpha1 is not None and delta1 is not None and N is None:
            N = 2 * (math.log(1 / delta1) + num_vars) / alpha1
            N = math.ceil(N)
            print("N: ", N)
        elif alpha1 is None and delta1 is not None and N is not None:
            alpha1 = 2 / N * (math.log(1 / delta1) + num_vars)
            print("alpha1: ", alpha1)
        elif alpha1 is not None and delta1 is None and N is not None:
            beta1 = 1 / math.exp(0.5 * alpha1 * N - num_vars)
            print("beta1: ", beta1)
        else:
            raise ValueError("At least two of alpha1, beta1, and N must be entered!")
        self.alpha1 = alpha1
        self.delta1 = delta1
        self.N = N

        M = kwargs.get("M", None)
        l = kwargs.get("l", None)
        tau = kwargs.get("tau", None)
        delta2 = kwargs.get("delta2", None)
        if M is not None and tau is not None and delta2 is not None:
            pass
        elif M is None and tau is not None and delta2 is not None:
            M = U_a**2 * math.log(1/((1 - l) * delta2)) / (2 * tau ** 2)
            M = math.ceil(M)
            print("M = ", M)
        else:
            raise ValueError("At least two of M, tau, and beta2 must be entered!")
        self.M = M
        self.tau = tau
        self.delta2 = delta2
        print("N*M: ", N*M)

        self.batch_size = kwargs.get("batch_size", 50)
        random_seed = kwargs.get("random_seed", False)
        if random_seed is not False:
            np.random.seed(random_seed)

        plot_dim = kwargs.get("plot_dim", [[0, 1]])
        plot_project_values = kwargs.get("plot_project_values", {})
        self.plot_manager = PlotManager(dim=plot_dim, project_values=plot_project_values, grid=False,
                                        v_filled=True,
                                        save=True, prob_name=self.model.__class__.__name__, save_file='jpg')

    def get_N(self):
        return self.N

    def get_M(self):
        return self.M

    def solve(self, obj_x, safe_set):
        start_time = time.time()

        obj_h_x_data = self.templates.calc_values(obj_x)
        self.solver.set_objective(obj_h_x_data, minimize=True)

        x_data = sample_data.sample_data(self.N, safe_set, "random")

        # np.save(f"../exp/lotka_x_data/lotka_x_data_{self.M}.npy", x_data)
        # x_data = np.load(f"../exp/lotka_x_data/lotka_x_data_{self.M}.npy")
        # x_data = x_data[:self.N]

        h_x = self.templates.calc_values(x_data)

        batch_size = self.batch_size

        h_fx = []
        constant = []

        num_iterations = (self.N + batch_size - 1) // batch_size

        for start in tqdm(range(0, self.N, batch_size), total=num_iterations):
        # for start in range(0, self.N, batch_size):
            end = min(start + batch_size, self.N)
            x_batch = x_data[start:end]

            x_batch_repeat = np.repeat(x_batch, repeats=self.M, axis=0)

            d = sample_disturbance.sample_disturbance(self.M * x_batch.shape[0], distribution=self.model.disturbance_type,
                                                      d_para=self.model.d_para)

            fx_batch = np.array(self.model.fx(x_batch_repeat, d, None)).T

            is_in_safe_set_batch = sample_data.get_is_in_safe_set(fx_batch, safe_set)

            h_fx_batch = self.templates.calc_values(fx_batch)

            h_unsafe = np.zeros((h_fx_batch.shape[1],))
            h_fx_batch[~is_in_safe_set_batch] = h_unsafe

            s1, s2 = h_fx_batch.shape
            h_fx_batch = h_fx_batch.reshape(s1 // self.M, self.M, s2)
            is_in_safe_set_batch = is_in_safe_set_batch.reshape(s1 // self.M, self.M)

            weight = np.full((self.M,), 1 / self.M)
            h_fx_batch = np.tensordot(h_fx_batch, weight, axes=(1, 0))

            count_true = np.sum(is_in_safe_set_batch, axis=1)
            count_false = is_in_safe_set_batch.shape[1] - count_true
            C_M = count_false * self.C / self.M

            constant_batch = -C_M - self.tau

            h_fx.append(h_fx_batch)
            constant.append(constant_batch)

        h_fx = np.concatenate(h_fx, axis=0)
        constant = np.concatenate(constant, axis=0)

        constraint = h_fx - h_x        # <=

        constraint = np.insert(constraint, constraint.shape[1], -1, axis=1)    # add tau

        self.solver.add_constraint(constraint, constant)

        solution, obj_max = self.solver.solve()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The time required for solution is {elapsed_time} seconds.")


        lamda = solution[len(solution) - 1]
        self.coe = solution[0: len(solution) - 1]

        return self.coe, lamda

    def plot(self, h_str, sim_data=None):
        if sim_data is not None:
            self.plot_manager.add_monte_carlo(sim_data)

        self.plot_manager.add_v(v_str=h_str)
        self.plot_manager.add_safe_set()

        if self.traj is not None:
            for i in self.traj:
                self.plot_manager.add_traj(i)

        self.plot_manager.show()

    def get_h_value(self, x_data):
        h_x_mono = self.templates.calc_values(x_data)
        h_x_value = np.dot(h_x_mono, self.coe)
        return h_x_value

    def sim_traj(self, init_state, sim_time, lamda):
        x = np.array(init_state).reshape(1, -1)
        traj_array = [init_state]
        h_value = float(self.get_h_value(x)[0])
        h_value = 1 - h_value - lamda
        h_value = max(0, h_value)
        h_array = [h_value]
        # print(x)
        # print(self.get_h_value(x))
        for j in range(sim_time):
            d = sample_disturbance.sample_disturbance(x.shape[0], distribution=self.model.disturbance_type, d_para=self.model.d_para)
            x = self.model.fx(x, d)
            x = np.array(x)
            x = x.reshape(self.model.degree_state)
            traj_array.append(x)
            x = x.reshape(1, -1)

            h_value = float(self.get_h_value(x)[0])
            h_value = 1 - h_value - lamda
            h_value = max(0, h_value)
            # print(x)
            # print(h_value)
            h_array.append(h_value)
        return traj_array, h_array


