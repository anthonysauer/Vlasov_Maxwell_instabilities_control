"""
Modified Armijo method with added Brownian motion, as described in Appendix C
"""
from collections.abc import Callable
from functools import partial
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np

from jax import jit, random


def update_step(J: Callable[[jax.Array], jax.Array]):
    value_and_grad_fn = jax.value_and_grad(J)

    @partial(jit, static_argnums=(5,))
    def _update_step(
            key,
            p: jax.Array,
            sigma: float,
            alpha_init: float,
            c: float,
            max_halvings: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        key, subkey = random.split(key)

        J_val, grad = value_and_grad_fn(p)
        s = -jnp.sum(grad ** 2)

        # Armijo backtracking
        init_condition = J(p - alpha_init * grad) <= J_val + c * alpha_init * s
        init_state = (0, alpha_init, p - alpha_init * grad, J_val, init_condition)

        def cond_fn(state):
            i, _, _, _, condition_met = state
            return (i < max_halvings) & (~condition_met)

        def body_fn(state):
            i, alpha, _, _, _ = state

            alpha_next = alpha * 0.5
            p_next = p - alpha_next * grad
            J_next = J(p_next)
            condition_met = J_next <= J_val + c * alpha_next * s

            return i + 1, alpha_next, p_next, J_next, condition_met

        _, _, final_p, final_J, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)

        xi = sigma * random.normal(subkey, p.shape)
        return final_p + xi, J_val, key

    return _update_step


def optimize_armijo(
        J: Callable[[jax.Array], jax.Array],
        params_init,
        sigma: float = 1e-7,
        alpha_init: float = 1e-6,
        c: float = 1e-4,
        max_halvings: int = 20,
        max_iterations: int = 20000,
        tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    init_key = random.PRNGKey(0)

    loss_history = []
    params_history = []

    compiled_update_step = update_step(J)

    p = jnp.asarray(params_init)
    for _ in tqdm(range(max_iterations)):
        p, J_val, init_key = compiled_update_step(init_key, p, sigma, alpha_init, c, max_halvings)
        loss_history.append(J_val)
        params_history.append(p)

    loss_history = np.asarray(loss_history)
    params_history = np.asarray(params_history)

    # Find if tolerance threshold was crossed
    under_tol = loss_history < tol
    if np.any(under_tol):
        cutoff_idx = np.argmax(under_tol) + 1
        loss_history = loss_history[:cutoff_idx]
        params_history = params_history[:cutoff_idx]

    best_epoch = np.argmin(loss_history)
    best_objective = loss_history[best_epoch]
    best_params = params_history[best_epoch]

    return loss_history, best_params, best_objective, int(best_epoch)
