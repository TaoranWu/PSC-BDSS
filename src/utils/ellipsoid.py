from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from numbers import Real


class Ellipsoid:
    def __init__(self, center: ArrayLike, coefficients: ArrayLike):
        center = center if isinstance(center, np.ndarray) else np.atleast_1d(center).astype(float)
        coefficients = coefficients if isinstance(coefficients, np.ndarray) else np.atleast_1d(coefficients).astype(float)
        assert center.shape == coefficients.shape

        self._center = center
        self._coefficients = coefficients
        self._degree = self._center.shape[0]

        radii = 1 / np.sqrt(coefficients)
        # Compute the bounds by shifting the ellipsoid center
        self._inf = center - radii
        self._sup = center + radii

    @property
    def inf(self) -> np.ndarray:
        return self._inf

    @property
    def sup(self) -> np.ndarray:
        return self._sup

    @property
    def center(self) -> np.ndarray:
        return self._center

    @property
    def coefficients(self) -> np.ndarray:
        return self._coefficients

    @property
    def degree(self) -> int:
        return self._degree

    def generate_data(self, N, method):
        data = np.empty((0, self._degree))
        if method == "random":
            scaling_factors = 1 / np.sqrt(self.coefficients)
            while len(data) < N:
                candidate_points = np.random.uniform(-1, 1, size=(N, self._degree))

                norms_squared = np.sum((candidate_points / scaling_factors) ** 2, axis=1)
                valid_points = candidate_points[norms_squared <= 1]

                data = np.vstack((data, valid_points))

            data = data[:N]
        elif method == "grid":
            N_each_degree = int(np.ceil(N ** (1 / self._degree)))
            grids = [np.linspace(self._inf[i], self._sup[i], N_each_degree) for i in range(self._degree)]
            data = np.array(np.meshgrid(*grids)).T.reshape(-1, self._degree)
            is_in_safe_set = self.is_in_safe_set(data)
            data = data[is_in_safe_set]
        else:
            raise NotImplementedError()
        return data

    def is_in_safe_set(self, fx_data):
        # Compute squared norm for each point relative to the ellipsoid
        norms_squared = np.sum(((fx_data - self._center) ** 2) * self._coefficients, axis=1)
        return norms_squared <= 1

