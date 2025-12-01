import numpy as np
from scipy.stats import truncnorm


def sample_disturbance(num_samples, distribution, d_para):
    if distribution == "uniform":
        lower_bound = d_para['lower_bound']
        upper_bound = d_para['upper_bound']
        d_array = np.random.uniform(low=lower_bound, high=upper_bound,
                                    size=(num_samples, lower_bound.shape[0]))
    elif distribution == "normal":
        mean = d_para['mean']
        if isinstance(mean, (int, float)):
            variance = d_para['variance']
            d_array = np.random.normal(loc=mean, scale=np.sqrt(variance), size=num_samples)
            d_array = d_array.reshape(-1, 1)
        else:
            cov = d_para['cov']
            d_array = np.random.multivariate_normal(mean=mean, cov=cov, size=num_samples)
    elif distribution == "truncated_normal":
        mean = d_para['mean']
        variance = d_para['variance']
        lower_bound = d_para['lower_bound']
        upper_bound = d_para['upper_bound']
        std = np.sqrt(variance)
        a = (lower_bound - mean) / std
        b = (upper_bound - mean) / std
        d_array = truncnorm.rvs(a=a, b=b, loc=mean, scale=std, size=num_samples)
        d_array = d_array.reshape(-1, 1)
    elif distribution == "beta":
        alpha = d_para['alpha']
        beta = d_para['beta']
        d_array = np.random.beta(a=alpha, b=beta, size=num_samples)
        d_array = d_array.reshape(-1, 1)
    else:
        raise NotImplementedError
    return d_array


