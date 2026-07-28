import numpy as np
import orbax.checkpoint as ocp
import optax
import jax
import jax.numpy as jnp

from flax import nnx
from tqdm import tqdm
from jax import random

from vm_control.data import PRECOMPUTED_DIR
from vm_control.optimizer_objectives import two_stream_nnx_objective
from vm_control.simple_nn import SimpleNN

BATCH_SIZE = 4
ACCUMULATION_STEPS = 8


def main() -> None:
    t_end = 30
    delta_t = 0.1
    delta_t_control = 1.0
    n_sensors = 8
    window = 2
    objective = two_stream_nnx_objective(
        n_x=16,
        n_vx=32,
        n_vy=32,
        n_sensors=n_sensors,
        t_end=t_end,
        delta_t=delta_t,
        delta_t_control=delta_t_control,
        window=window,
    )

    K = 5
    model = SimpleNN(
        n_features=n_sensors * window,
        n_targets=K * 4,
        n_hidden=100,
        rngs=nnx.Rngs(0),
    )

    initial_learning_rate = 0.001
    max_iterations = 10
    scheduler = optax.cosine_decay_schedule(init_value=initial_learning_rate, decay_steps=max_iterations)
    base_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=scheduler)
    )
    optimizer_def = optax.MultiSteps(
        base_optimizer,
        every_k_schedule=ACCUMULATION_STEPS
    )
    optimizer = nnx.Optimizer(model, optimizer_def, wrt=nnx.Param)

    def batched_objective(model_opt: nnx.Module, phase_shifts: jax.Array):
        vmapped_objective = nnx.vmap(objective, in_axes=(None, 0))

        # Compute the losses for the entire batch and average them
        losses = vmapped_objective(model_opt, phase_shifts)
        return jnp.mean(losses)

    @nnx.jit
    def train_step(model_opt: nnx.Module, opt: nnx.Optimizer, key_opt):
        phase_shifts = random.uniform(
            key_opt,
            shape=(BATCH_SIZE,),
            minval=0.0,
            maxval=2 * jnp.pi,
        )
        value_and_grad_fn = nnx.value_and_grad(batched_objective, argnums=0)
        value, grad = value_and_grad_fn(model_opt, phase_shifts)
        opt.update(model_opt, grad)
        return value

    loss_history = []
    best_objective = np.inf
    best_epoch = 0
    key = random.PRNGKey(0)

    for i in tqdm(range(max_iterations)):
        key, subkey = random.split(key)
        val = train_step(model, optimizer, subkey)
        loss_history.append(val)
        if val < best_objective:
            best_objective = val
            best_epoch = i

    output_dict = {
        "loss_history": np.asarray(loss_history),
        "best_objective": best_objective,
        "best_epoch": best_epoch,
    }
    np.savez(PRECOMPUTED_DIR / "two_stream_optimization_nn.npz", **output_dict)

    _, state = nnx.split(model)

    # Define checkpoint directory and options
    checkpoint_dir = PRECOMPUTED_DIR / "checkpoints" / "two_stream_optimization_nn"
    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)

    # CheckpointManager context to safely handle background tasks
    with ocp.CheckpointManager(checkpoint_dir, options=options) as mngr:
        mngr.save(max_iterations - 1, args=ocp.args.StandardSave(state))
        mngr.wait_until_finished()


if __name__ == "__main__":
    main()
