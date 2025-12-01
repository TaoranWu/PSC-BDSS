from src.utils.interval import Interval
from src.utils.ellipsoid import Ellipsoid
import numpy as np


def generate_data(safe_set, N, model, method="random"):
    if isinstance(safe_set, Interval):
        # assert self.model.degree_state == safe_set.inf.shape[0]
        x_data = safe_set.generate_data(N, method)
    elif isinstance(safe_set, Ellipsoid):
        # assert self.model.degree_state == safe_set.degree
        x_data = safe_set.generate_data(N, method)
    else:
        raise NotImplementedError("Sampling within this type of safe set is not implemented")
    fx_data = np.array(model.fx(x_data, None)).T
    return categorize_data(x_data, fx_data, safe_set)


def generate_data_stochastic(safe_set, N2, N3, model, method="random"):
    if isinstance(safe_set, Interval):
        x_data = safe_set.generate_data(N2, method)
    elif isinstance(safe_set, Ellipsoid):
        x_data = safe_set.generate_data(N2, method)
    else:
        raise NotImplementedError("Sampling within this type of safe set is not implemented")

    x_data_repeat = np.repeat(x_data, repeats=N3, axis=0)

    d = np.random.uniform(low=model.d_lb, high=model.d_ub, size=(N3 * N2, model.d_lb.shape[0]))
    fx_data = np.array(model.fx(x_data_repeat, d, None)).T

    is_in_safe_set = get_is_in_safe_set(fx_data, safe_set)

    return x_data, fx_data, is_in_safe_set


def categorize_data(x_data, fx_data, safe_set):
    is_in_safe_set = get_is_in_safe_set(fx_data, safe_set)
    x_safe = x_data[is_in_safe_set]
    fx_safe = fx_data[is_in_safe_set]
    x_unsafe = x_data[~is_in_safe_set]
    fx_unsafe = fx_data[~is_in_safe_set]
    return x_safe, fx_safe, x_unsafe, fx_unsafe


def get_is_in_safe_set(fx_data, safe_set):
    if isinstance(safe_set, Interval):
        is_in_safe_set = np.all((fx_data >= safe_set.inf) & (fx_data <= safe_set.sup), axis=1)
    elif isinstance(safe_set, Ellipsoid):
        is_in_safe_set = safe_set.is_in_safe_set(fx_data)
    else:
        raise NotImplementedError("Sampling within this type of safe set is not implemented")
    return is_in_safe_set


def sample_obj_data(N1, safe_set, method="grid"):
    N1 = int(N1)
    if isinstance(safe_set, Interval):
        data = safe_set.generate_data(N1, method)
    elif isinstance(safe_set, Ellipsoid):
        data = safe_set.generate_data(N1, method)
    else:
        raise NotImplementedError("Sampling within this type of safe set is not implemented")
    return data


def sample_data(N, safe_set, method="random"):
    N = int(N)
    if isinstance(safe_set, Interval):
        obj_data = safe_set.generate_data(N, method)
    elif isinstance(safe_set, Ellipsoid):
        obj_data = safe_set.generate_data(N, method)
    else:
        raise NotImplementedError("Sampling within this type of safe set is not implemented")
    return obj_data
