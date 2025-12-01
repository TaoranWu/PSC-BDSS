from src.utils.interval import Interval
from src.utils.ellipsoid import Ellipsoid
from src.utils.sample_data import sample_data, get_is_in_safe_set
from src.utils.sample_disturbance import sample_disturbance
import numpy as np


def monte_carlo_sis(safe_set, num_sample, sim_time, model, proj_dim=None):
    num_sample, sim_time = int(num_sample), int(sim_time)
    is_in_safe_set = np.ones(num_sample, dtype=bool)

    if proj_dim is None:
        if isinstance(safe_set, Interval):
            x_data = np.random.uniform(safe_set.inf, safe_set.sup, size=(num_sample, len(safe_set.inf)))
        elif isinstance(safe_set, Ellipsoid):
            x_data = safe_set.generate_data(num_sample, "random")
        else:
            raise NotImplementedError("Sampling within this type of safe set is not implemented")
    else:
        if isinstance(safe_set, Interval):
            inf_proj = np.array([safe_set.inf[proj_dim[0]], safe_set.inf[proj_dim[1]]])
            sup_proj = np.array([safe_set.sup[proj_dim[0]], safe_set.sup[proj_dim[1]]])
            x_data_proj = np.random.uniform(inf_proj, sup_proj, size=(num_sample, 2))
            x_data = np.zeros(shape=(num_sample, len(safe_set.inf)))
            x_data[:, proj_dim[0]] = x_data_proj[:, 0]
            x_data[:, proj_dim[1]] = x_data_proj[:, 1]
        else:
            raise NotImplementedError("Sampling within this type of safe set is not implemented")

    x_data_i = x_data.copy()

    for i in range(sim_time):
        fx_data = np.array(model.fx(x_data_i, None)).T
        if isinstance(safe_set, Interval):
            is_in_safe_set_i = np.all((fx_data >= safe_set.inf) & (fx_data <= safe_set.sup), axis=1)
        elif isinstance(safe_set, Ellipsoid):
            is_in_safe_set_i = safe_set.is_in_safe_set(fx_data)
        else:
            raise NotImplementedError("Sampling within this type of safe set is not implemented")
        is_in_safe_set = is_in_safe_set & is_in_safe_set_i

        x_data_i = fx_data

    x_data = x_data[is_in_safe_set]
    return x_data


def monte_carlo_st_one_step_safe(safe_set, n_x, n_d, model):
    n_x, n_d = int(n_x), int(n_d)

    x_data = sample_data(n_x, safe_set, method="random")
    d_data = sample_disturbance(n_x * n_d, model.disturbance_type, model.d_para)

    x_data_repeat = np.repeat(x_data, repeats=n_d, axis=0)

    fx_data = np.array(model.fx(x_data_repeat, d_data, None)).T

    is_in_safe_set = get_is_in_safe_set(fx_data, safe_set)

    is_in_safe_set = is_in_safe_set.reshape(n_x, n_d)

    count_safe = np.sum(is_in_safe_set, axis=1)

    safe_prob_per_state = count_safe / n_d

    safe_prob = np.mean(safe_prob_per_state)

    print(safe_prob)

def st_one_step_safe(x_data, n_d, model, safe_set):
    n_d = int(n_d)
    n_x = int(len(x_data))

    d_data = sample_disturbance(n_x * n_d, model.disturbance_type, model.d_para)

    x_data_repeat = np.repeat(x_data, repeats=n_d, axis=0)

    fx_data = np.array(model.fx(x_data_repeat, d_data, None)).T

    is_in_safe_set = get_is_in_safe_set(fx_data, safe_set)

    is_in_safe_set = is_in_safe_set.reshape(n_x, n_d)

    count_safe = np.sum(is_in_safe_set, axis=1)

    safe_prob_per_state = count_safe / n_d

    return safe_prob_per_state



