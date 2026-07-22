from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from vm_control.data import PRECOMPUTED_DIR
from vm_control.rl.two_stream_env import TwoStreamEnv


def main() -> None:
    log_dir = str(PRECOMPUTED_DIR / "rl_logs/")

    env = TwoStreamEnv()
    env = Monitor(env, log_dir)

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10)
    model.save(PRECOMPUTED_DIR / "sac")


if __name__ == "__main__":
    main()
