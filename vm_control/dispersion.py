"""Dispersion-function utilities for the Weibel linear analysis."""

from __future__ import annotations

import numpy as np


def weibel_L_G(s, k: float = 1.25, v_th: float = 0.3, temperature_ratio: float = 12.0):
    """Analytic Laplace transform of G(t) for the Gaussian Weibel equilibrium.

    For mu(vx,vy)=1/(pi v_th^2 sqrt(T_r)) exp(-(vx^2+vy^2/T_r)/v_th^2),
    G(t)=t*(T_r v_th^2/2)*exp(-(v_th^2 k^2 t^2)/4).

    The expression is evaluated with the Faddeeva function to avoid forming
    exp(s**2/(4a))*erfc(s/(2*sqrt(a))) directly on large complex grids.
    """
    from scipy.special import wofz

    s = np.asarray(s, dtype=np.complex128)
    a = (float(v_th) * float(k)) ** 2 / 4.0
    c = float(temperature_ratio) * float(v_th) ** 2 / 2.0
    sqrt_a = np.sqrt(a)
    F = np.sqrt(np.pi) / (2.0 * sqrt_a) * wofz(1j * s / (2.0 * sqrt_a))
    return c / (2.0 * a) * (1.0 - s * F)


def weibel_D2(s, k: float = 1.25, v_th: float = 0.3, temperature_ratio: float = 12.0):
    """Magnetic/transverse dispersion function D2(k,s)."""
    return k ** 2 + np.asarray(s) ** 2 + 1.0 - k ** 2 * weibel_L_G(s, k, v_th, temperature_ratio)


def weibel_D2_grid(
        k: float = 1.25,
        sigma_points: int = 501,
        omega_points: int = 501,
        chunk_size: int = 64,
):
    """Compute |D2| on a complex-s grid for the Weibel dispersion relation."""
    sigma = np.linspace(0.0, 1.0, sigma_points)
    omega = np.linspace(-1.0, 1.0, omega_points)
    magnitude = np.empty((sigma_points, omega_points), dtype=float)
    for start in range(0, sigma_points, chunk_size):
        stop = min(start + chunk_size, sigma_points)
        S = sigma[start:stop, None] + 1j * omega[None, :]
        magnitude[start:stop, :] = np.abs(weibel_D2(S, k=k))
    idx = np.argmin(magnitude)
    i, j = np.unravel_index(idx, magnitude.shape)
    minimum = {"abs_D2": float(magnitude[i, j]), "sigma": float(sigma[i]), "omega": float(omega[j])}
    return sigma, omega, magnitude, minimum
