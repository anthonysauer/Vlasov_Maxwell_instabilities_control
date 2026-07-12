"""
Adam optimizer, adapted from: https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py
"""
from collections.abc import Callable
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np


def adam(step_size, b1=0.9, b2=0.999, eps=1e-8):
    """Construct optimizer triple for Adam.

    Args:
      step_size: positive scalar, or a callable representing a step size schedule
        that maps the iteration index to a positive scalar.
      b1: optional, a positive scalar value for beta_1, the exponential decay rate
        for the first moment estimates (default 0.9).
      b2: optional, a positive scalar value for beta_2, the exponential decay rate
        for the second moment estimates (default 0.999).
      eps: optional, a positive scalar value for epsilon, a small constant for
        numerical stability (default 1e-8).

    Returns:
      An (init_fun, update_fun, get_params) triple.
    """

    def init(x0):
        m0 = jnp.zeros_like(x0)
        v0 = jnp.zeros_like(x0)
        return x0, m0, v0

    def update(i, g, state):
        x, m, v = state
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * jnp.square(g) + b2 * v  # Second moment estimate.
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        x = x - step_size * mhat / (jnp.sqrt(vhat) + eps)
        return x, m, v

    def get_params(state):
        x, _, _ = state
        return x

    return init, update, get_params


def optimize_adam(
        J: Callable[[jax.Array], jax.Array],
        params_init,
        step_size: float,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        max_iterations: int = 20000,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    opt_init, opt_update, get_params = adam(
        step_size=step_size,
        b1=b1,
        b2=b2,
        eps=eps,
    )
    opt_state = opt_init(params_init)
    value_and_grad_fn = jax.jit(jax.value_and_grad(J))

    def step(iteration, state):
        value, grad = value_and_grad_fn(get_params(state))
        next_state = opt_update(iteration, grad, state)
        return value, next_state

    loss_history = []
    best_params = params_init
    best_objective = np.inf
    best_epoch = 0

    for i in tqdm(range(max_iterations)):
        opt_value, opt_state = step(i, opt_state)

        loss_history.append(opt_value)
        if opt_value < best_objective:
            best_params = get_params(opt_state)
            best_objective = opt_value
            best_epoch = i

    return np.asarray(loss_history), best_params, best_objective, best_epoch
