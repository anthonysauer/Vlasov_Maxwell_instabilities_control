"""Reusable experiment builders for the Vlasov--Maxwell control examples.

The default grids are deliberately small so the functions can be smoke-tested on a
laptop.  The manuscript-scale parameters are documented in the notebook and
README; use those values when running the expensive full simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, pi
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from .profiles import two_stream_best_params
from .solver import solver_jit, compute_rho


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
    f0 = (1.0 / (jnp.pi * vth2 * jnp.sqrt(tr))) * jnp.exp(-((VX ** 2 + (VY ** 2) / tr) / vth2))
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
    prefactor = 1.0 / (2.0 * jnp.pi * v_th ** 2)
    term_x = jnp.exp(-((VX - v_bar) ** 2) / v_th ** 2) + jnp.exp(-((VX + v_bar) ** 2) / v_th ** 2)
    term_y = jnp.exp(-(VY ** 2) / v_th ** 2)
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


def run_two_stream_simulation_jnp(
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
) -> tuple[tuple[jax.Array, ...], jax.Array, jax.Array, jax.Array]:
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
    return result, grid_x, grid_vx, grid_vy


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
    result, grid_x, grid_vx, grid_vy = run_two_stream_simulation_jnp(
        n_x=n_x,
        n_vx=n_vx,
        n_vy=n_vy,
        t_end=t_end,
        delta_t=delta_t,
        beta=beta,
        v_th=v_th,
        v_bar=v_bar,
        alpha=alpha,
        params=params,
        use_best_params=use_best_params,
    )
    return _solver_result_to_dict(result, delta_t, grid_x, grid_vx, grid_vy)


def evolve_external_fields(params, t, grid_x, beta: float = 0.5):
    params = jnp.asarray(params, dtype=jnp.float32).reshape((-1, 4))
    k_count = params.shape[0]
    a, b, c, d = params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4]
    cos_minus_array = jnp.stack(
        [jnp.cos((k + 1) * beta * (grid_x - t)) for k in range(k_count)], axis=0
    )
    cos_plus_array = jnp.stack(
        [jnp.cos((k + 1) * beta * (grid_x + t)) for k in range(k_count)], axis=0
    )
    sin_minus_array = jnp.stack(
        [jnp.sin((k + 1) * beta * (grid_x - t)) for k in range(k_count)], axis=0
    )
    sin_plus_array = jnp.stack(
        [jnp.sin((k + 1) * beta * (grid_x + t)) for k in range(k_count)], axis=0
    )
    B_ext = jnp.sum(
        a * cos_minus_array - b * sin_plus_array + c * sin_minus_array - d * cos_plus_array, axis=0
    )
    Ey_ext = jnp.sum(
        a * cos_minus_array + b * sin_plus_array + c * sin_minus_array + d * cos_plus_array, axis=0
    )
    return B_ext, Ey_ext


def run_two_stream_simulation_adaptive_jnp(
        *,
        n_x: int = 32,
        n_vx: int = 64,
        n_vy: int = 64,
        t_end: float = 2.0,
        delta_t: float = 0.1,
        delta_t_control: float = 1.0,
        beta: float = 0.5,
        v_th: float = 0.2,
        v_bar: float = 0.7,
        alpha: float = 1e-3,
        params=None,
        # use_best_params: bool = False,
) -> tuple[tuple[jax.Array, ...], jax.Array, jax.Array, jax.Array]:
    """Run two-stream simulation with adaptive external fields and return jax arrays."""
    # if use_best_params:
    #     params = two_stream_best_params()
    length_x = 2.0 * pi / float(beta)
    spec = GridSpec(n_x, n_vx, n_vy, 0.0, length_x, -2.5, 2.5, -2.5, 2.5)
    grid_x, grid_vx, grid_vy = make_grids(spec)

    n_control = floor(float(t_end) / float(delta_t_control))
    params = jnp.asarray(params, dtype=jnp.float32).reshape((n_control, -1))

    n_steps_per_control = ceil(float(delta_t_control) / float(delta_t))

    def iteration_step(carry, precomputed_fields):
        f_old, _, Ey_old, B_old = carry
        B_ext_old, Ey_ext_old, B_ext_new, Ey_ext_new = precomputed_fields

        # Substitute in new external fields
        B_init_new = B_old - B_ext_old + B_ext_new
        Ey_init_new = Ey_old - Ey_ext_old + Ey_ext_new

        # Solve until next external field update
        result = solver_jit(f_old, B_init_new, Ey_init_new, grid_x, grid_vx, grid_vy, float(delta_t),
                            n_steps_per_control, 1)
        (
            f_new,
            E_x_energy,
            E_y_energy,
            B_energy,
            kinetic_energy,
            r_x,
            r_y,
            F_B,
            F_Ex,
            F_Ey,
            Ex_new,
            Ey_new,
            B_new,
        ) = result

        carry = (f_new, Ex_new, Ey_new, B_new)
        outputs = (
            E_x_energy,
            E_y_energy,
            B_energy,
            kinetic_energy,
            r_x,
            r_y,
            F_B,
            F_Ex,
            F_Ey,
        )
        return carry, outputs

    # Initialize scan with first solve
    f0, B0, Ey0 = make_two_stream_initial_condition(
        grid_x,
        grid_vx,
        grid_vy,
        beta=beta,
        v_th=v_th,
        v_bar=v_bar,
        alpha=alpha,
        params=params[0],
    )
    result_init = solver_jit(f0, B0, Ey0, grid_x, grid_vx, grid_vy, float(delta_t), n_steps_per_control, 1)
    (
        f_init,
        E_x_energy_init,
        E_y_energy_init,
        B_energy_init,
        kinetic_energy_init,
        r_x_init,
        r_y_init,
        F_B_init,
        F_Ex_init,
        F_Ey_init,
        Ex_init,
        Ey_init,
        B_init,
    ) = result_init
    initial_carry = (
        f_init,
        Ex_init,
        Ey_init,
        B_init,
    )

    # Pre-compute the external field evolution for the entire simulation
    params_old = params[0: n_control - 1]
    params_new = params[1: n_control]
    ts = jnp.arange(1, n_control, dtype=jnp.float32) * float(delta_t_control)

    vmap_evolve = jax.vmap(evolve_external_fields, in_axes=(0, 0, None, None))
    B_ext_old_history, Ey_ext_old_history = vmap_evolve(params_old, ts, grid_x, beta)
    B_ext_new_history, Ey_ext_new_history = vmap_evolve(params_new, ts, grid_x, beta)

    # Perform the scan over the remaining (n_control - 1) iterations
    scan_inputs = (B_ext_old_history, Ey_ext_old_history, B_ext_new_history, Ey_ext_new_history)
    final_carry, final_outputs = jax.lax.scan(iteration_step, initial_carry, xs=scan_inputs, length=n_control - 1)

    # Unpack the final carry
    f_end, Ex_final, Ey_final, B_final = final_carry

    # Unpack the outputs
    (
        electric_energy_E_x,
        electric_energy_E_y,
        magnetic_energy_history,
        kinetic_energy_history,
        r_x_history,
        r_y_history,
        Fourier_mode_B,
        Fourier_mode_E_x,
        Fourier_mode_E_y,
    ) = final_outputs

    # Add the initial output to the beginning
    electric_energy_E_x_full = jnp.concatenate([E_x_energy_init[None, :], electric_energy_E_x], axis=0)
    electric_energy_E_y_full = jnp.concatenate([E_y_energy_init[None, :], electric_energy_E_y], axis=0)
    magnetic_energy_history_full = jnp.concatenate([B_energy_init[None, :], magnetic_energy_history], axis=0)
    kinetic_energy_history_full = jnp.concatenate([kinetic_energy_init[None, :], kinetic_energy_history], axis=0)
    r_x_history_full = jnp.concatenate([r_x_init[None, :], r_x_history], axis=0)
    r_y_history_full = jnp.concatenate([r_y_init[None, :], r_y_history], axis=0)
    Fourier_mode_B_full = jnp.concatenate([F_B_init[None, :], Fourier_mode_B], axis=0)
    Fourier_mode_E_x_full = jnp.concatenate([F_Ex_init[None, :], Fourier_mode_E_x], axis=0)
    Fourier_mode_E_y_full = jnp.concatenate([F_Ey_init[None, :], Fourier_mode_E_y], axis=0)

    # Return the final state and histories
    return (
        f_end,
        electric_energy_E_x_full,
        electric_energy_E_y_full,
        magnetic_energy_history_full,
        kinetic_energy_history_full,
        r_x_history_full,
        r_y_history_full,
        Fourier_mode_B_full,
        Fourier_mode_E_x_full,
        Fourier_mode_E_y_full,
        Ex_final,
        Ey_final,
        B_final
    ), grid_x, grid_vx, grid_vy


def run_two_stream_simulation_adaptive(
        *,
        n_x: int = 32,
        n_vx: int = 64,
        n_vy: int = 64,
        t_end: float = 2.0,
        delta_t: float = 0.1,
        delta_t_control: float = 1.0,
        beta: float = 0.5,
        v_th: float = 0.2,
        v_bar: float = 0.7,
        alpha: float = 1e-3,
        params=None,
        # use_best_params: bool = False,
) -> Dict[str, np.ndarray]:
    """Run one two-stream simulation with adaptive external fields and return NumPy arrays."""
    result, grid_x, grid_vx, grid_vy = run_two_stream_simulation_adaptive_jnp(
        n_x=n_x,
        n_vx=n_vx,
        n_vy=n_vy,
        t_end=t_end,
        delta_t=delta_t,
        delta_t_control=delta_t_control,
        beta=beta,
        v_th=v_th,
        v_bar=v_bar,
        alpha=alpha,
        params=params,
        # use_best_params=use_best_params,
    )
    return _solver_result_to_dict(result, delta_t, grid_x, grid_vx, grid_vy)


def _solve_result_to_scan_outputs(f_old, B_old, Ey_old, grid_x, grid_vx, grid_vy, delta_t, n_steps_per_control, *args):
    result = solver_jit(f_old, B_old, Ey_old, grid_x, grid_vx, grid_vy, float(delta_t), n_steps_per_control, 1)
    (
        f_new,
        E_x_energy,
        E_y_energy,
        B_energy,
        kinetic_energy_history,
        r_x,
        r_y,
        F_B,
        F_Ex,
        F_Ey,
        Ex_new,
        Ey_new,
        B_new,
    ) = result

    carry = (f_new, Ex_new, Ey_new, B_new, *args)
    outputs = (E_x_energy, E_y_energy, B_energy, kinetic_energy_history, r_x, r_y, F_B, F_Ex, F_Ey)
    return carry, outputs


def burn_in(grid_x, grid_vx, grid_vy, delta_t, n_steps_per_control, n_sensors, window, beta, v_th, v_bar, alpha):
    n_x = grid_x.shape[0]
    dvx = grid_vx[1] - grid_vx[0]
    dvy = grid_vy[1] - grid_vy[0]

    # Collect initial observations with no control
    def no_control_iteration_step(carry, _):
        f_old, _, Ey_old, B_old, observations_old = carry

        # Density observations
        rho = compute_rho(f_old, dvx, dvy)
        rho_observations = jnp.array([jnp.mean(chunk) for chunk in jnp.array_split(rho, n_sensors)])
        observations_new = jnp.concatenate([observations_old[1:], rho_observations[None, ...]], axis=0)

        # Solve until next observation (no control)
        return _solve_result_to_scan_outputs(f_old, B_old, Ey_old, grid_x, grid_vx, grid_vy, delta_t,
                                             n_steps_per_control, observations_new)

    # Initial solve from initial conditions
    f0, B0, Ey0 = make_two_stream_initial_condition(
        grid_x,
        grid_vx,
        grid_vy,
        beta=beta,
        v_th=v_th,
        v_bar=v_bar,
        alpha=alpha,
    )
    observations_no_control_init = jnp.zeros((window, n_sensors))
    initial_carry_no_control = (f0, jnp.zeros(n_x), Ey0, B0, observations_no_control_init)

    final_carry_no_control, final_outputs_no_control = jax.lax.scan(no_control_iteration_step,
                                                                    initial_carry_no_control,
                                                                    length=window - 1)
    return *final_carry_no_control, *final_outputs_no_control


def run_two_stream_simulation_nnx_jnp(
        *,
        n_x: int = 32,
        n_vx: int = 64,
        n_vy: int = 64,
        n_sensors: int = 8,
        t_end: float = 2.0,
        delta_t: float = 0.1,
        delta_t_control: float = 1.0,
        window: int = 2,
        beta: float = 0.5,
        v_th: float = 0.2,
        v_bar: float = 0.7,
        alpha: float = 1e-3,
        model: nnx.Module,
) -> tuple[tuple[jax.Array, ...], jax.Array, jax.Array, jax.Array]:
    """Run two-stream simulation with adaptive external fields from an nnx model and return jax arrays."""
    length_x = 2.0 * pi / float(beta)
    spec = GridSpec(n_x, n_vx, n_vy, 0.0, length_x, -2.5, 2.5, -2.5, 2.5)
    grid_x, grid_vx, grid_vy = make_grids(spec)
    dvx = grid_vx[1] - grid_vx[0]
    dvy = grid_vy[1] - grid_vy[0]

    n_control = floor(float(t_end) / float(delta_t_control))
    n_steps_per_control = ceil(float(delta_t_control) / float(delta_t))
    # observation_indices = jnp.round(jnp.linspace(0, n_x - 1, num=n_sensors)).astype(int)

    burn_in_outputs = burn_in(grid_x, grid_vx, grid_vy, delta_t, n_steps_per_control, n_sensors, window, beta, v_th,
                              v_bar, alpha, )

    (
        f_end_no_control,
        Ex_final_no_control,
        Ey_final_no_control,
        B_final_no_control,
        observations_final_no_control,
        electric_energy_E_x_no_control,
        electric_energy_E_y_no_control,
        magnetic_energy_no_control,
        kinetic_energy_no_control,
        r_x_history_no_control,
        r_y_history_no_control,
        Fourier_mode_B_no_control,
        Fourier_mode_E_x_no_control,
        Fourier_mode_E_y_no_control,
    ) = burn_in_outputs

    # Split the object into a static graph definition and a dynamic state dictionary
    graphdef, model_state = nnx.split(model)

    # Perform the scan over the remaining iterations with control
    def iteration_step(carry, t):
        f_old, _, Ey_old, B_old, params_old, observations_old, current_model_state = carry

        # Merge model state back into a functional, callable model inside the loop
        model_inside = nnx.merge(graphdef, current_model_state)

        # Density observations
        rho = compute_rho(f_old, dvx, dvy)
        rho_observations = jnp.array([jnp.mean(chunk) for chunk in jnp.array_split(rho, n_sensors)])
        observations_new = jnp.concatenate([observations_old[1:], rho_observations[None, ...]], axis=0)

        # Compute new params based on density observations
        params_new = model_inside(observations_new)

        # Split again to capture any state changes (like RNG updates)
        _, updated_model_state = nnx.split(model_inside)

        # Substitute in new external fields
        B_ext_old, Ey_ext_old = evolve_external_fields(params_old, t, grid_x, beta=beta)
        B_ext_new, Ey_ext_new = evolve_external_fields(params_new, t, grid_x, beta=beta)

        B_init_new = B_old - B_ext_old + B_ext_new
        Ey_init_new = Ey_old - Ey_ext_old + Ey_ext_new

        # Solve until next external field update
        return _solve_result_to_scan_outputs(
            f_old,
            B_init_new,
            Ey_init_new,
            grid_x,
            grid_vx,
            grid_vy,
            delta_t,
            n_steps_per_control,
            params_new,
            observations_new,
            updated_model_state
        )

    # Use shape of model output to initialize parameters
    dummy_input = jnp.zeros(window * n_sensors)
    output_shape = jax.eval_shape(model, dummy_input)
    params_init = jnp.zeros(output_shape.shape)

    initial_carry = (f_end_no_control, Ex_final_no_control, Ey_final_no_control, B_final_no_control, params_init,
                     observations_final_no_control, model_state)
    ts = jnp.arange(window - 1, n_control, dtype=jnp.float32) * float(delta_t_control)

    final_carry, final_outputs = jax.lax.scan(iteration_step, initial_carry, xs=ts, length=n_control - window + 1)

    # Unpack the final carry
    f_end, Ex_final, Ey_final, B_final, _, _, _ = final_carry

    # Unpack the outputs
    (
        electric_energy_E_x,
        electric_energy_E_y,
        magnetic_energy,
        kinetic_energy,
        r_x_history,
        r_y_history,
        Fourier_mode_B,
        Fourier_mode_E_x,
        Fourier_mode_E_y,
    ) = final_outputs

    # Add the initial and no-control output to the beginning
    electric_energy_E_x_full = jnp.concatenate([electric_energy_E_x_no_control, electric_energy_E_x], axis=0)
    electric_energy_E_y_full = jnp.concatenate([electric_energy_E_y_no_control, electric_energy_E_y], axis=0)
    magnetic_energy_full = jnp.concatenate([magnetic_energy_no_control, magnetic_energy], axis=0)
    kinetic_energy_full = jnp.concatenate([kinetic_energy_no_control, kinetic_energy], axis=0)
    r_x_history_full = jnp.concatenate([r_x_history_no_control, r_x_history], axis=0)
    r_y_history_full = jnp.concatenate([r_y_history_no_control, r_y_history], axis=0)
    Fourier_mode_B_full = jnp.concatenate([Fourier_mode_B_no_control, Fourier_mode_B], axis=0)
    Fourier_mode_E_x_full = jnp.concatenate([Fourier_mode_E_x_no_control, Fourier_mode_E_x], axis=0)
    Fourier_mode_E_y_full = jnp.concatenate([Fourier_mode_E_y_no_control, Fourier_mode_E_y], axis=0)

    # Return the final state and histories
    return (
        f_end,
        electric_energy_E_x_full,
        electric_energy_E_y_full,
        magnetic_energy_full,
        kinetic_energy_full,
        r_x_history_full,
        r_y_history_full,
        Fourier_mode_B_full,
        Fourier_mode_E_x_full,
        Fourier_mode_E_y_full,
        Ex_final,
        Ey_final,
        B_final
    ), grid_x, grid_vx, grid_vy


def run_two_stream_simulation_nnx(
        *,
        n_x: int = 32,
        n_vx: int = 64,
        n_vy: int = 64,
        n_sensors: int = 8,
        t_end: float = 2.0,
        delta_t: float = 0.1,
        delta_t_control: float = 1.0,
        window: int = 2,
        beta: float = 0.5,
        v_th: float = 0.2,
        v_bar: float = 0.7,
        alpha: float = 1e-3,
        model: nnx.Module,
) -> Dict[str, np.ndarray]:
    """Run one two-stream simulation with adaptive external fields and return NumPy arrays."""
    result, grid_x, grid_vx, grid_vy = run_two_stream_simulation_nnx_jnp(
        n_x=n_x,
        n_vx=n_vx,
        n_vy=n_vy,
        n_sensors=n_sensors,
        t_end=t_end,
        delta_t=delta_t,
        delta_t_control=delta_t_control,
        window=window,
        beta=beta,
        v_th=v_th,
        v_bar=v_bar,
        alpha=alpha,
        model=model,
    )
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
    n = int(np.asarray(E_x_energy).flatten().shape[0])
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
    "run_two_stream_simulation_jnp",
    "run_two_stream_simulation",
    "run_two_stream_simulation_adaptive_jnp",
    "run_two_stream_simulation_adaptive",
    "run_two_stream_simulation_nnx_jnp",
    "run_two_stream_simulation_nnx",
    "burn_in",
    "evolve_external_fields",
]
