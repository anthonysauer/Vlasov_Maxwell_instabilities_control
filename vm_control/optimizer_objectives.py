import jax.numpy as jnp
from jax import jit
from flax import nnx

from vm_control.experiments import run_two_stream_simulation_jnp, run_two_stream_simulation_adaptive_jnp, \
    run_two_stream_simulation_nnx_jnp


def two_stream_initial_objective(
        n_x,
        n_vx,
        n_vy,
        t_end,
        delta_t,
):
    @jit
    def objective(params):
        two_stream_results, _, _, _ = run_two_stream_simulation_jnp(
            n_x=n_x,
            n_vx=n_vx,
            n_vy=n_vy,
            t_end=t_end,
            delta_t=delta_t,
            params=params,
        )
        E_x_energy = two_stream_results[1]
        electric_energy_E_x_total = jnp.sum(E_x_energy) * delta_t
        return electric_energy_E_x_total

    return objective


def two_stream_adaptive_objective(
        n_x,
        n_vx,
        n_vy,
        t_end,
        delta_t,
        delta_t_control,
):
    @jit
    def objective(params):
        two_stream_results, _, _, _ = run_two_stream_simulation_adaptive_jnp(
            n_x=n_x,
            n_vx=n_vx,
            n_vy=n_vy,
            t_end=t_end,
            delta_t=delta_t,
            delta_t_control=delta_t_control,
            params=params,
        )
        E_x_energy = two_stream_results[1]
        electric_energy_E_x_total = jnp.sum(E_x_energy) * delta_t
        return electric_energy_E_x_total

    return objective


def two_stream_nnx_objective(
        n_x,
        n_vx,
        n_vy,
        n_sensors,
        t_end,
        delta_t,
        delta_t_control,
        window,
):
    @nnx.jit
    def objective(
            model: nnx.Module,
    ):
        two_stream_results, _, _, _ = run_two_stream_simulation_nnx_jnp(
            n_x=n_x,
            n_vx=n_vx,
            n_vy=n_vy,
            n_sensors=n_sensors,
            t_end=t_end,
            delta_t=delta_t,
            delta_t_control=delta_t_control,
            window=window,
            model=model,
        )
        E_x_energy = two_stream_results[1]
        electric_energy_E_x_total = jnp.sum(E_x_energy) * delta_t
        return electric_energy_E_x_total

    return objective
