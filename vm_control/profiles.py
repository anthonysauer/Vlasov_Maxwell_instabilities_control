"""Equilibria, perturbations, and control-field parameterizations."""

from __future__ import annotations

import numpy as np


def weibel_equilibrium(vx, vy, v_th: float = 0.3, temperature_ratio: float = 12.0):
    """Anisotropic Weibel equilibrium used in the submitted manuscript.

    Parameters follow the paper notation:
    mu(vx, vy) = 1/(pi v_th^2 sqrt(T_r)) exp(-(vx^2 + vy^2/T_r)/v_th^2).
    """
    vx = np.asarray(vx)
    vy = np.asarray(vy)
    vth2 = float(v_th) ** 2
    tr = float(temperature_ratio)
    return (1.0 / (np.pi * vth2 * np.sqrt(tr))) * np.exp(
        -((vx**2) + (vy**2) / tr) / vth2
    )


def two_stream_equilibrium(vx, vy, v_th: float = 0.2, v_bar: float = 0.7):
    """Two-stream equilibrium used for the nonlinear-control experiment."""
    vx = np.asarray(vx)
    vy = np.asarray(vy)
    vth2 = float(v_th) ** 2
    prefactor = 1.0 / (2.0 * np.pi * vth2)
    term_x = np.exp(-((vx - v_bar) ** 2) / vth2) + np.exp(-((vx + v_bar) ** 2) / vth2)
    term_y = np.exp(-(vy**2) / vth2)
    return prefactor * term_x * term_y


def two_stream_best_params() -> np.ndarray:
    """Best-so-far coefficients reported in Table 1, shaped as rows (a_k,b_k,c_k,d_k)."""
    return np.array(
        [
            [-1.6697127e-02, -1.2023713e-02, +1.1340540e-02, -1.7294141e-02],
            [-2.0891201e-02, +4.5357272e-03, +5.7352898e-03, +2.1741122e-02],
            [+1.3055697e-03, +2.6091118e-03, -1.1644824e-03, -2.5653021e-04],
            [-2.1751216e-03, +9.8036020e-04, -1.0026476e-03, +1.2053368e-03],
            [-8.1608276e-04, +3.4699708e-04, +9.0509515e-05, +7.6559186e-04],
        ],
        dtype=np.float64,
    )


def build_two_stream_fields(params, x, k0: float = 0.5):
    """Build the initial transverse fields from harmonic coefficients.

    Parameters
    ----------
    params:
        Array with shape (K, 4) or flat length 4K. Columns are (a_k, b_k, c_k, d_k).
    x:
        Spatial grid.
    k0:
        Fundamental wave number. The manuscript uses k0=0.5.

    Returns
    -------
    Ey, Bz:
        External transverse electric and magnetic fields on the x-grid.
    """
    p = np.asarray(params, dtype=np.float64).reshape((-1, 4))
    x = np.asarray(x, dtype=np.float64)
    Ey = np.zeros_like(x, dtype=np.float64)
    Bz = np.zeros_like(x, dtype=np.float64)
    for m, (a, b, c, d) in enumerate(p, start=1):
        cos_m = np.cos(m * k0 * x)
        sin_m = np.sin(m * k0 * x)
        Ey += (a + d) * cos_m + (b + c) * sin_m
        Bz += (a - d) * cos_m + (b - c) * sin_m
    return Ey, Bz
