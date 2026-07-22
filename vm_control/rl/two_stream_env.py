import gymnasium as gym
import numpy as np

from gymnasium import spaces
from math import ceil, pi
from vm_control.experiments import GridSpec, make_grids, evolve_external_fields, burn_in
from vm_control.solver import solver_jit


class TwoStreamEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(
            self,
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
            K: int = 5,
            max_amplitude: float = 0.1,
    ):
        super().__init__()

        self.n_sensors = n_sensors
        self.t_end = t_end
        self.delta_t = delta_t
        self.delta_t_control = delta_t_control
        self.window = window
        self.beta = beta
        self.v_th = v_th
        self.v_bar = v_bar
        self.alpha = alpha
        self.K = K
        self.max_amplitude = max_amplitude

        length_x = 2.0 * pi / float(self.beta)
        spec = GridSpec(n_x, n_vx, n_vy, 0.0, length_x, -2.5, 2.5, -2.5, 2.5)
        self.grid_x, self.grid_vx, self.grid_vy = make_grids(spec)
        self.dvx = self.grid_vx[1] - self.grid_vx[0]
        self.dvy = self.grid_vy[1] - self.grid_vy[0]

        self.n_steps_per_control = ceil(float(self.delta_t_control) / float(self.delta_t))

        # State variables
        self.f = None
        self.Ey = None
        self.B = None
        self.t = -1
        self.params_prev = None
        self.observations = None

        # Define action and observation space
        self.action_space = spaces.Box(
            low=-self.max_amplitude,
            high=self.max_amplitude,
            shape=(self.K * 4,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.window * self.n_sensors,),
            dtype=np.float32
        )

    def step(self, action):
        # Substitute in new external fields
        B_ext_old, Ey_ext_old = evolve_external_fields(self.params_prev, self.t, self.grid_x, beta=self.beta)
        B_ext_new, Ey_ext_new = evolve_external_fields(action, self.t, self.grid_x, beta=self.beta)

        B_init_new = self.B - B_ext_old + B_ext_new
        Ey_init_new = self.Ey - Ey_ext_old + Ey_ext_new

        # Solve to next control timestep
        result = solver_jit(self.f, B_init_new, Ey_init_new, self.grid_x, self.grid_vx, self.grid_vy,
                            float(self.delta_t), self.n_steps_per_control, 1)
        (f_new, E_x_energy, _, _, _, _, _, _, _, _, _, Ey_new, B_new,) = result

        # Reward is negative total Ex energy
        reward = -float(np.sum(E_x_energy) * self.delta_t)

        # Update state
        self.f = f_new
        self.Ey = Ey_new
        self.B = B_new
        self.t = self.t + self.delta_t_control
        self.params_prev = action

        # Add next density observation
        rho = np.sum(self.f, axis=(1, 2)) * self.dvx * self.dvy
        rho_observations = np.array([np.mean(chunk) for chunk in np.array_split(rho, self.n_sensors)])
        self.observations = np.concatenate([self.observations[1:], rho_observations[None, ...]], axis=0)

        # Only end once t_end reached
        terminated = False
        truncated = (self.t == self.t_end)
        info = {}

        return self.observations.flatten(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Run burn-in phase
        burn_in_outputs = burn_in(self.grid_x,
                                  self.grid_vx,
                                  self.grid_vy,
                                  self.delta_t,
                                  self.n_steps_per_control,
                                  self.n_sensors,
                                  self.window,
                                  self.beta,
                                  self.v_th,
                                  self.v_bar,
                                  self.alpha)
        (f_burn_in, _, Ey_burn_in, B_burn_in, observations_burn_in, _, _, _, _, _, _, _, _, _,) = burn_in_outputs

        # Reset state
        self.f = f_burn_in
        self.Ey = Ey_burn_in
        self.B = B_burn_in
        self.t = (self.window - 1) * self.delta_t_control
        self.params_prev = np.zeros((self.K * 4,), dtype=np.float32)

        # Add final density observation to observations from burn-in phase
        rho = np.sum(self.f, axis=(1, 2)) * self.dvx * self.dvy
        rho_observations = np.array([np.mean(chunk) for chunk in np.array_split(rho, self.n_sensors)])
        self.observations = np.concatenate([observations_burn_in[1:], rho_observations[None, ...]], axis=0)
        info = {}

        return self.observations.flatten(), info

    def render(self):
        pass
