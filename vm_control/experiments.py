"""Reusable experiment builders for the Vlasov--Maxwell control examples.

The default grids are deliberately small so the functions can be smoke-tested on a
laptop.  The manuscript-scale parameters are documented in the notebook and
README; use those values when running the expensive full simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, pi
from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np

from .profiles import two_stream_best_params
from .solver import solver_jit


@dataclass(frozen=True)
class GridSpec:
    """Phase-space grid specification."""

    n_x: int
    n_vx: int
    n_vy: int
    x_min: float
    x_max: float
    vx_min: float
    vx_max: float
    vy_min: float
    vy_max: float


def make_grids(spec: GridSpec):
    """Return JAX arrays ``grid_x, grid_vx, grid_vy`` from a ``GridSpec``."""
    grid_x = jnp.linspace(spec.x_min, spec.x_max, spec.n_x, endpoint=False)
    grid_vx = jnp.linspace(spec.vx_min, spec.vx_max, spec.n_vx)
    grid_vy = jnp.linspace(spec.vy_min, spec.vy_max, spec.n_vy)
    return grid_x, grid_vx, grid_vy


def make_weibel_initial_condition(
    grid_x,
    grid_vx,
    grid_vy,
    *,
    k0: float = 1.25,
    mode: int = 3,
    v_th: float = 0.3,
    temperature_ratio: float = 12.0,
    alpha: float = 1e-3,
    net_B_amplitude: float = 1e-3,
):
    """Construct the Weibel initial condition used by the linear-control test.

    ``net_B_amplitude`` is the *net* magnetic amplitude passed to the solver.
    For the no-control case use ``1e-3``.  For exact cancellation control use
    ``0.0``.  For mismatch tests use the residual amplitude, e.g. ``1e-5``.
    """
    X, VX, VY = jnp.meshgrid(grid_x, grid_vx, grid_vy, indexing="ij")
    vth2 = float(v_th) ** 2
    tr = float(temperature_ratio)
    f0 = (1.0 / (jnp.pi * vth2 * jnp.sqrt(tr))) * jnp.exp(-((VX**2 + (VY**2) / tr) / vth2))
    f0 = f0 * (1.0 + alpha * jnp.cos(mode * k0 * X))
    B0 = net_B_amplitude * jnp.cos(mode * k0 * grid_x)
    Ey0 = jnp.zeros_like(grid_x)
    return f0, B0, Ey0


def run_weibel_simulation(
    *,
    n_x: int = 32,
    n_vx: int = 64,
    n_vy: int = 64,
    t_end: float = 2.0,
    delta_t: float = 0.1,
    k0: float = 1.25,
    mode: int = 3,
    v_th: float = 0.3,
    temperature_ratio: float = 12.0,
    alpha: float = 1e-3,
    net_B_amplitude: float = 1e-3,
) -> Dict[str, np.ndarray]:
    """Run one Weibel simulation and return NumPy arrays.

    The submitted high-resolution histories used ``t_end=500`` and smaller time
    steps for some panels; that run is intentionally not the default.
    """
    length_x = 2.0 * pi / float(k0)
    spec = GridSpec(n_x, n_vx, n_vy, 0.0, length_x, -2.5, 2.5, -2.5, 2.5)
    grid_x, grid_vx, grid_vy = make_grids(spec)
    f0, B0, Ey0 = make_weibel_initial_condition(
        grid_x,
        grid_vx,
        grid_vy,
        k0=k0,
        mode=mode,
        v_th=v_th,
        temperature_ratio=temperature_ratio,
        alpha=alpha,
        net_B_amplitude=net_B_amplitude,
    )
    n_steps = ceil(float(t_end) / float(delta_t))
    result = solver_jit(f0, B0, Ey0, grid_x, grid_vx, grid_vy, float(delta_t), n_steps, int(mode))
    return _solver_result_to_dict(result, delta_t, grid_x, grid_vx, grid_vy)


def make_two_stream_initial_condition(
    grid_x,
    grid_vx,
    grid_vy,
    *,
    beta: float = 0.5,
    v_th: float = 0.2,
    v_bar: float = 0.7,
    alpha: float = 1e-3,
    params=None,
):
    """Construct the two-stream initial condition.

    ``params`` may be ``None`` for the no-control baseline, a flat array of
    length ``4K``, or an array of shape ``(K, 4)`` with columns
    ``(a_k, b_k, c_k, d_k)``.
    """
    X, VX, VY = jnp.meshgrid(grid_x, grid_vx, grid_vy, indexing="ij")
    prefactor = 1.0 / (2.0 * jnp.pi * v_th**2)
    term_x = jnp.exp(-((VX - v_bar) ** 2) / v_th**2) + jnp.exp(-((VX + v_bar) ** 2) / v_th**2)
    term_y = jnp.exp(-(VY**2) / v_th**2)
    f0 = prefactor * term_x * term_y * (1.0 + alpha * jnp.sin(beta * X))

    if params is None:
        B0 = jnp.zeros_like(grid_x)
        Ey0 = jnp.zeros_like(grid_x)
    else:
        params = jnp.asarray(params, dtype=jnp.float32).reshape((-1, 4))
        k_count = params.shape[0]
        cos_array = jnp.stack([jnp.cos((k + 1) * beta * grid_x) for k in range(k_count)], axis=0)
        sin_array = jnp.stack([jnp.sin((k + 1) * beta * grid_x) for k in range(k_count)], axis=0)
        a, b, c, d = params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4]
        B0 = jnp.sum((a - d) * cos_array + (b - c) * sin_array, axis=0)
        Ey0 = jnp.sum((a + d) * cos_array + (b + c) * sin_array, axis=0)
    return f0, B0, Ey0


def run_two_stream_simulation(
    *,
    n_x: int = 32,
    n_vx: int = 64,
    n_vy: int = 64,
    t_end: float = 2.0,
    delta_t: float = 0.1,
    beta: float = 0.5,
    v_th: float = 0.2,
    v_bar: float = 0.7,
    alpha: float = 1e-3,
    params=None,
    use_best_params: bool = False,
) -> Dict[str, np.ndarray]:
    """Run one two-stream simulation and return NumPy arrays."""
    if use_best_params:
        params = two_stream_best_params()
    length_x = 2.0 * pi / float(beta)
    spec = GridSpec(n_x, n_vx, n_vy, 0.0, length_x, -2.5, 2.5, -2.5, 2.5)
    grid_x, grid_vx, grid_vy = make_grids(spec)
    f0, B0, Ey0 = make_two_stream_initial_condition(
        grid_x,
        grid_vx,
        grid_vy,
        beta=beta,
        v_th=v_th,
        v_bar=v_bar,
        alpha=alpha,
        params=params,
    )
    n_steps = ceil(float(t_end) / float(delta_t))
    result = solver_jit(f0, B0, Ey0, grid_x, grid_vx, grid_vy, float(delta_t), n_steps, 1)
    return _solver_result_to_dict(result, delta_t, grid_x, grid_vx, grid_vy)


def _solver_result_to_dict(result, delta_t: float, grid_x, grid_vx, grid_vy) -> Dict[str, np.ndarray]:
    (
        f_end,
        E_x_energy,
        E_y_energy,
        B_energy,
        kinetic_energy,
        r_x_history,
        r_y_history,
        F_B,
        F_Ex,
        F_Ey,
        Ex_final,
        Ey_final,
        B_final,
    ) = result
    n = int(np.asarray(E_x_energy).shape[0])
    return {
        "time": np.arange(n, dtype=np.float64) * float(delta_t),
        "f_end": np.asarray(f_end),
        "E_x_energy": np.asarray(E_x_energy),
        "E_y_energy": np.asarray(E_y_energy),
        "B_energy": np.asarray(B_energy),
        "kinetic_energy": np.asarray(kinetic_energy),
        "r_x_history": np.asarray(r_x_history),
        "r_y_history": np.asarray(r_y_history),
        "F_B": np.asarray(F_B),
        "F_Ex": np.asarray(F_Ex),
        "F_Ey": np.asarray(F_Ey),
        "Ex_final": np.asarray(Ex_final),
        "Ey_final": np.asarray(Ey_final),
        "B_final": np.asarray(B_final),
        "grid_x": np.asarray(grid_x),
        "grid_vx": np.asarray(grid_vx),
        "grid_vy": np.asarray(grid_vy),
    }


__all__ = [
    "GridSpec",
    "make_grids",
    "make_weibel_initial_condition",
    "make_two_stream_initial_condition",
    "run_weibel_simulation",
    "run_two_stream_simulation",
]
