import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None


class RandomParameters:
    def __init__(self, distributions, backend=None):
        self.distributions = list(distributions)
        self.backend = backend if backend is not None else np

    def _resolve_backend(self, betas_random):
        if self.backend is not None:
            return self.backend
        if jnp is not None and "jax" in type(betas_random).__module__:
            return jnp
        return np

    def _apply_single_distribution(self, backend, values, distribution_name):
        if distribution_name == "ln":
            return backend.exp(values)
        if distribution_name == "tn":
            return backend.maximum(values, 0)
        if distribution_name == "t":
            return backend.where(
                values <= 0.5,
                backend.sqrt(2 * values) - 1,
                1 - backend.sqrt(2 * (1 - values)),
            )
        if distribution_name == "u":
            return 2 * values - 1
        return values

    def apply_distribution(self, betas_random, index=None):
        backend = self._resolve_backend(betas_random)
        active_distributions = self.distributions if index is None else list(index)
        transformed = betas_random

        for parameter_index, distribution_name in enumerate(active_distributions):
            values = transformed[:, parameter_index, :]
            updated_values = self._apply_single_distribution(backend, values, distribution_name)
            if hasattr(transformed, "at"):
                transformed = transformed.at[:, parameter_index, :].set(updated_values)
            else:
                transformed[:, parameter_index, :] = updated_values

        return transformed
