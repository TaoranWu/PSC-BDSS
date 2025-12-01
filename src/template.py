import math
import numpy as np
import itertools
from scipy.special import comb
import torch


class Templates:
    def __init__(self, degree_systems: int, temp_type: str = "poly", degree: int = 6, verbose: int = True,
                 x_lb=None, x_ub=None):
        if not isinstance(degree_systems, int):
            raise TypeError("'num_variables' must be of integer type!")
        if not isinstance(degree, int):
            raise TypeError("'degree' must be of integer type!")
        self.degree_systems = degree_systems
        self.temp_type = temp_type
        self.verbose = verbose
        self.num_vars = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if temp_type == "poly":
            self.degree_poly = degree
            self.num_vars = math.comb(self.degree_systems + self.degree_poly, self.degree_poly)
        elif temp_type == "bernstein" or temp_type == "bernstein2":
            self.degree_bernstein = degree
            self.num_vars = (self.degree_bernstein + 1) ** self.degree_systems
            self.x_lb = x_lb
            self.x_ub = x_ub
            if torch.cuda.is_available():
                # self.x_lb = torch.from_numpy(x_lb).double().to(self.device)
                # self.x_ub = torch.from_numpy(x_ub).double().to(self.device)
                self.x_lb = torch.from_numpy(x_lb).float().to(self.device)
                self.x_ub = torch.from_numpy(x_ub).float().to(self.device)
        elif temp_type == "handelman":
            self.degree_handelman = degree
            self.num_vars = (self.degree_handelman + 1) ** self.degree_systems
            self.x_lb = x_lb
            self.x_ub = x_ub
            if torch.cuda.is_available():
                # self.x_lb = torch.from_numpy(x_lb).double().to(self.device)
                # self.x_ub = torch.from_numpy(x_ub).double().to(self.device)
                self.x_lb = torch.from_numpy(x_lb).float().to(self.device)
                self.x_ub = torch.from_numpy(x_ub).float().to(self.device)
        elif temp_type == "ex":
            self.degree_ex = degree
            self.num_vars = math.comb(self.degree_systems + self.degree_ex, self.degree_ex)
        else:
            raise NotImplementedError

    def calc_values(self, data):
        result = None
        if self.temp_type == "poly":
            result = self._calc_values_poly(data)
        elif self.temp_type == "bernstein" or self.temp_type == "bernstein2":
            if torch.cuda.is_available():
                result = self._calc_values_bernstein_cuda(data)
            else:
                result = self._calc_values_bernstein_cpu(data)
        elif self.temp_type == "handelman":
            if torch.cuda.is_available():
                result = self._calc_values_handelman_cuda(data)
            else:
                result = self._calc_values_handelman_cpu(data)
        elif self.temp_type == "ex":
            result = self._calc_values_ex(data)
        return result

    def _calc_values_poly(self, data):
        terms = []

        for degree_combination in itertools.product(range(self.degree_poly + 1), repeat=self.degree_systems):
            if sum(degree_combination) <= self.degree_poly:
                terms.append(degree_combination)

        terms = np.array(terms)

        num_points = data.shape[0]
        num_terms = terms.shape[0]

        result = np.ones((num_points, num_terms))
        for i, degree_combination in enumerate(terms):
            for j in range(self.degree_systems):
                result[:, i] *= data[:, j] ** degree_combination[j]

        return result

    # def _calc_values_bernstein(self, data):
    #     num_points = data.shape[0]
    #     data_norm = (data - self.x_lb) / (self.x_ub - self.x_lb)
    #
    #     multi_indices = np.array(list(itertools.product(range(self.degree_bernstein + 1), repeat=self.degree_systems)))
    #
    #     basis_vals = np.ones((num_points, self.num_vars))
    #     for i in range(self.degree_systems):
    #         xi = data_norm[:, i][:, None]  # (N, 1)
    #         ki = multi_indices[:, i][None, :]  # (1, M)
    #         binom = special.comb(self.degree_bernstein, ki)  # (1, M)
    #         term = binom * (xi ** ki) * ((1 - xi) ** (self.degree_bernstein - ki))  # (N, M)
    #         basis_vals *= term
    #
    #     return basis_vals

    def _calc_values_bernstein_cpu(self, data):
        d = self.degree_bernstein
        data_norm = (data - self.x_lb) / (self.x_ub - self.x_lb)  # (N, m)

        # shape: (M, m)
        multi_indices = np.array(list(itertools.product(range(d + 1), repeat=self.degree_systems)))

        # shape: (m, d+1) —— binomial coefficients for each dimension
        binoms = comb(d, np.arange(d + 1))

        # shape: (N, m, 1)
        xi = data_norm[:, :, None]  # broadcast input
        # shape: (1, m, M)
        ki = multi_indices.T[None, :, :]  # broadcast indices

        # shape: (1, m, M)
        binom_coeffs = binoms[ki]

        # Compute (xi ** ki) * ((1 - xi) ** (d - ki))
        term = binom_coeffs * (xi ** ki) * ((1 - xi) ** (d - ki))  # shape: (N, m, M)

        basis_vals = np.prod(term, axis=1)  # shape: (N, M)

        return basis_vals

    def _calc_values_bernstein_cuda(self, data_np):
        d = self.degree_bernstein
        m = self.degree_systems

        data = torch.from_numpy(data_np).float().to(self.device)
        # data = torch.from_numpy(data_np).double().to(self.device)

        data_norm = (data - self.x_lb) / (self.x_ub - self.x_lb)

        multi_indices = torch.tensor(
            list(itertools.product(range(d + 1), repeat=m)),
            dtype=torch.long,
            device=self.device
        )

        binoms = torch.tensor([math.comb(d, k) for k in range(d + 1)], dtype=torch.float32, device=self.device)

        xi = data_norm[:, :, None]
        ki = multi_indices.T[None, :, :]
        binom_coeffs = binoms[ki]
        term = binom_coeffs * (xi ** ki) * ((1 - xi) ** (d - ki))
        basis_vals = torch.prod(term, dim=1)

        return basis_vals.cpu().numpy()

    def _calc_values_handelman_cpu(self, data):
        d = self.degree_handelman
        num_points = data.shape[0]
        x_l = data - self.x_lb
        u_l = self.x_ub - data

        multi_indices = np.array(list(itertools.product(range(self.degree_handelman + 1), repeat=self.degree_systems)))

        basis_vals = np.ones((num_points, self.num_vars))
        for i in range(self.degree_systems):
            x_l_i = x_l[:, i][:, None]  # (N, 1)
            u_l_i = u_l[:, i][:, None]
            ki = multi_indices[:, i][None, :]  # (1, M)
            # term = (x_l_i ** ki) * (u_l_i ** ki)  # (N, M)
            term = (x_l_i ** ki) * (u_l_i ** (d - ki))  # (N, M)
            basis_vals *= term

        return basis_vals

    def _calc_values_handelman_cuda(self, data_np):
        d = self.degree_handelman
        m = self.degree_systems

        data = torch.from_numpy(data_np).float().to(self.device)
        # data = torch.from_numpy(data_np).double().to(self.device)

        multi_indices = torch.tensor(
            list(itertools.product(range(d + 1), repeat=m)),
            dtype=torch.long,
            device=self.device
        )

        xi_l = data - self.x_lb
        xi_l = xi_l[:, :, None]
        xi_u = self.x_ub - data
        xi_u = xi_u[:, :, None]

        ki = multi_indices.T[None, :, :]
        term = (xi_l ** ki) * (xi_u ** (d - ki))
        basis_vals = torch.prod(term, dim=1)

        return basis_vals.cpu().numpy()

    def _calc_values_ex(self, data):
        data = np.exp(data)
        terms = []

        for degree_combination in itertools.product(range(self.degree_ex + 1), repeat=self.degree_systems):
            if sum(degree_combination) <= self.degree_ex:
                terms.append(degree_combination)

        terms = np.array(terms)

        num_points = data.shape[0]
        num_terms = terms.shape[0]

        result = np.ones((num_points, num_terms))
        for i, degree_combination in enumerate(terms):
            for j in range(self.degree_systems):
                result[:, i] *= data[:, j] ** degree_combination[j]

        return result

    def output(self, coefficients):
        if self.temp_type == "poly":
            return self._output_poly(coefficients)
        elif self.temp_type == "ex":
            return self._output_ex(coefficients)
        elif self.temp_type == "bernstein" or self.temp_type == "bernstein2":
            raise NotImplementedError
            # return self._output_bernstein(coefficients)
        elif self.temp_type == "handelman":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _output_poly(self, coefficients):
        terms = []
        index = 0

        for powers in itertools.product(range(self.degree_poly + 1), repeat=self.degree_systems):
            if sum(powers) <= self.degree_poly:
                coef = coefficients[index]
                term_parts = []
                for var_idx, power in enumerate(powers):
                    if power > 0:
                        term_parts.append(f"x{var_idx}^{power}" if power > 1 else f"x{var_idx}")
                term_str = " * ".join(term_parts)
                if term_str:
                    terms.append(f"{coef:+g} * {term_str}")
                else:
                    terms.append(f"{coef:+g}")
                index += 1

        polynomial_str = " ".join(terms)
        if polynomial_str.startswith("+"):
            polynomial_str = polynomial_str[1:].strip()

        return polynomial_str

    def _output_ex(self, coefficients):
        terms = []
        index = 0

        for powers in itertools.product(range(self.degree_ex + 1), repeat=self.degree_systems):
            if sum(powers) <= self.degree_ex:
                coef = coefficients[index]
                term_parts = []
                for var_idx, power in enumerate(powers):
                    if power > 0:
                        term_parts.append(f"exp(x{var_idx})^{power}" if power > 1 else f"exp(x{var_idx})")
                term_str = " * ".join(term_parts)
                if term_str:
                    terms.append(f"{coef:+g} * {term_str}")
                else:
                    terms.append(f"{coef:+g}")
                index += 1

        polynomial_str = " ".join(terms)
        if polynomial_str.startswith("+"):
            polynomial_str = polynomial_str[1:].strip()

        return polynomial_str