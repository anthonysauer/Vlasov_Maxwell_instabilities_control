import numpy as np

from math import floor

from vm_control.data import PRECOMPUTED_DIR
from vm_control.optimizer_adam import optimize_adam
from vm_control.optimizer_objectives import two_stream_adaptive_objective


def main() -> None:
    t_end = 30
    delta_t = 0.1
    delta_t_control = 1.0
    objective = two_stream_adaptive_objective(
        n_x=16,
        n_vx=32,
        n_vy=32,
        t_end=t_end,
        delta_t=delta_t,
        delta_t_control=delta_t_control,
    )

    K = 5
    n_control = floor(float(t_end) / float(delta_t_control))
    params_init = np.zeros(4 * K * n_control)
    loss_history, best_params, best_objective, best_epoch = optimize_adam(
        objective,
        params_init,
        step_size=0.001,
        max_iterations=10
    )

    output_dict = {
        "loss_history": loss_history,
        "best_params": best_params,
        "best_objective": best_objective,
        "best_epoch": best_epoch,
    }
    np.savez(PRECOMPUTED_DIR / "two_stream_optimization_adam_noise_adaptive.npz", **output_dict)


if __name__ == "__main__":
    main()
