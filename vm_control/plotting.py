"""Figure-generation utilities for the reproducibility notebook.

The functions in this module intentionally use compact precomputed arrays shipped
with this repository.  The expensive raw distribution histories and optimizer
checkpoints from the working directory are not needed for reproducing the paper
figures.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import PACKAGE_ROOT, PRECOMPUTED_DIR, REFERENCE_FIGURE_DIR, load_npz
from .profiles import two_stream_best_params, two_stream_equilibrium, weibel_equilibrium


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _progress(message: str) -> None:
    print(message, flush=True)


def _save(fig: plt.Figure, path: str | Path, dpi: int = 110) -> Path:
    """Save and close a figure using a notebook-safe noninteractive path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def copy_submitted_reference_figures(output_dir: str | Path, overwrite: bool = True) -> List[Path]:
    """Copy exact submitted manuscript PNGs into an output directory."""
    output_dir = _ensure_dir(output_dir)
    copied: List[Path] = []
    for src in sorted(REFERENCE_FIGURE_DIR.glob("*.png")):
        dst = output_dir / src.name
        if overwrite or not dst.exists():
            shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def plot_weibel_equilibria(output_dir: str | Path) -> List[Path]:
    """Regenerate the Weibel equilibrium panel used in Figure 2a."""
    output_dir = _ensure_dir(output_dir)
    vx = np.linspace(-2.5, 2.5, 241)
    vy = np.linspace(-2.5, 2.5, 241)
    VX, VY = np.meshgrid(vx, vy, indexing="ij")
    mu = weibel_equilibrium(VX, VY)

    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    im = ax.contourf(VX, VY, mu, levels=40)
    ax.set_xlabel(r"$v_x$")
    ax.set_ylabel(r"$v_y$")
    ax.set_title("Weibel equilibrium")
    fig.colorbar(im, ax=ax, label=r"$\mu(v_x,v_y)$")
    return [_save(fig, output_dir / "Figure-2a-equilibrium.regenerated.png")]


def plot_weibel_dispersion(output_dir: str | Path) -> List[Path]:
    """Regenerate the |D2(k,s)| landscape from the packaged compact grid.

    The analytic calculator remains available in ``vm_control.dispersion``.  The
    default notebook uses this stored grid so the one-click reproduction path is
    fast and deterministic on machines without a tuned SciPy installation.
    """
    output_dir = _ensure_dir(output_dir)
    z = load_npz("weibel_dispersion_grid.npz")
    sigma = z["sigma"]
    omega = z["omega"]
    magnitude = z["magnitude"]
    S, W = np.meshgrid(sigma, omega, indexing="ij")
    minimum = {
        "sigma": float(z["minimum_sigma"]),
        "omega": float(z["minimum_omega"]),
        "abs_D2": float(z["minimum_abs_D2"]),
    }

    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    im = ax.contourf(S, W, np.log10(magnitude + 1e-14), levels=50)
    ax.plot(minimum["sigma"], minimum["omega"], marker="o", markersize=4)
    ax.set_xlabel(r"$\sigma=\mathrm{Re}\,s$")
    ax.set_ylabel(r"$\omega=\mathrm{Im}\,s$")
    ax.set_title(r"$\log_{10}|D_2(k,s)|$, $k=1.25$")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}|D_2|$")
    path = output_dir / "Figure-1-D2-landscape.regenerated.png"
    return [_save(fig, path)]


def plot_weibel_energy_histories(output_dir: str | Path) -> List[Path]:
    """Regenerate Weibel energy histories for no-control and cancellation-control runs."""
    output_dir = _ensure_dir(output_dir)
    z = load_npz("weibel_fig4_histories.npz")
    paths: List[Path] = []

    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    t = z["no_time"]
    ax.semilogy(t, z["no_E_x"] + 1e-18, label=r"$E_x$")
    ax.semilogy(t, z["no_E_y"] + 1e-18, label=r"$E_y$")
    ax.semilogy(t, z["no_B"] + 1e-18, label=r"$B_z$")
    ax.set_xlabel("time")
    ax.set_ylabel("field energy")
    ax.set_title("Weibel, no control")
    ax.legend(frameon=False)
    paths.append(_save(fig, output_dir / "Figure-2b-field-energy-no-control.regenerated.png"))

    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    t = z["ctrl_time"]
    ax.semilogy(t, z["ctrl_E_x"] + 1e-22, label=r"$E_x$")
    ax.semilogy(t, z["ctrl_E_y"] + 1e-22, label=r"$E_y$")
    ax.semilogy(t, z["ctrl_B"] + 1e-22, label=r"$B_z$")
    ax.set_xlabel("time")
    ax.set_ylabel("field energy")
    ax.set_title("Weibel, analytic cancellation control")
    ax.legend(frameon=False)
    paths.append(_save(fig, output_dir / "Weibel-field-energy-with-control.regenerated.png"))

    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    ax.plot(z["no_time"], z["no_K"] - z["no_K"][0], label="no control")
    ax.plot(z["ctrl_time"], z["ctrl_K"] - z["ctrl_K"][0], label="with control")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$K(t)-K(0)$")
    ax.set_title("Change in kinetic energy")
    ax.legend(frameon=False)
    paths.append(_save(fig, output_dir / "Figure-4c-change-of-kinetic-energy.regenerated.png"))
    return paths


def plot_weibel_fourier_modes(output_dir: str | Path) -> List[Path]:
    """Regenerate Fourier-mode panels for the Weibel experiment."""
    output_dir = _ensure_dir(output_dir)
    z = load_npz("weibel_fig4_histories.npz")
    paths: List[Path] = []

    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    t = z["no_time"]
    ax.semilogy(t, np.abs(z["no_F_B"]) + 1e-18, label=r"$|\hat B_3|$")
    ax.semilogy(t, np.abs(z["no_F_Ex"]) + 1e-18, label=r"$|\hat E_{x,3}|$")
    ax.semilogy(t, np.abs(z["no_F_Ey"]) + 1e-18, label=r"$|\hat E_{y,3}|$")
    ax.set_xlabel("time")
    ax.set_ylabel("Fourier amplitude")
    ax.set_title("Weibel mode 3, no control")
    ax.legend(frameon=False)
    paths.append(_save(fig, output_dir / "Figure-4a-Fourier-no-control.regenerated.png"))

    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    t = z["ctrl_time"]
    ax.semilogy(t, np.abs(z["ctrl_F_B"]) + 1e-22, label=r"$|\hat B_3|$")
    ax.semilogy(t, np.abs(z["ctrl_F_Ex"]) + 1e-22, label=r"$|\hat E_{x,3}|$")
    ax.semilogy(t, np.abs(z["ctrl_F_Ey"]) + 1e-22, label=r"$|\hat E_{y,3}|$")
    ax.set_xlabel("time")
    ax.set_ylabel("Fourier amplitude")
    ax.set_title("Weibel mode 3, with control")
    ax.legend(frameon=False)
    paths.append(_save(fig, output_dir / "Figure-4b-Fourier-with-control.regenerated.png"))
    return paths


def plot_weibel_distribution_samples(output_dir: str | Path) -> List[Path]:
    """Regenerate compact initial/final Weibel distribution slices."""
    output_dir = _ensure_dir(output_dir)
    z = load_npz("weibel_no_control_distribution_samples.npz")
    vx = z["grid_vx"]
    vy = z["grid_vy"]
    x = z["grid_x"]
    x_indices = z["x_indices"]
    # Draw downsampled copies to keep the one-click notebook fast. The full
    # 256 x 256 slices remain stored unchanged in data/precomputed/.
    f_initial = z["f_initial"][:, ::2, ::2]
    f_end = z["f_end"][:, ::2, ::2]
    paths: List[Path] = []

    fig, axes = plt.subplots(2, len(x_indices), figsize=(3.0 * len(x_indices), 5.6), sharex=True, sharey=True)
    vmax = max(float(np.nanmax(f_initial)), float(np.nanmax(f_end)))
    vmin = min(float(np.nanmin(f_initial)), float(np.nanmin(f_end)))
    for j, idx in enumerate(x_indices):
        im0 = axes[0, j].imshow(
            f_initial[j].T,
            origin="lower",
            extent=[vx.min(), vx.max(), vy.min(), vy.max()],
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        axes[0, j].set_title(fr"initial, $x={x[idx]:.2f}$")
        im1 = axes[1, j].imshow(
            f_end[j].T,
            origin="lower",
            extent=[vx.min(), vx.max(), vy.min(), vy.max()],
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        axes[1, j].set_title(fr"final, $x={x[idx]:.2f}$")
        axes[1, j].set_xlabel(r"$v_x$")
    axes[0, 0].set_ylabel(r"$v_y$")
    axes[1, 0].set_ylabel(r"$v_y$")
    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85, label="f")
    paths.append(_save(fig, output_dir / "Figure-3-distribution-samples.regenerated.png"))
    return paths


def plot_weibel_landscapes(output_dir: str | Path) -> List[Path]:
    """Regenerate the linear-control Delta-K landscapes for Figure 6."""
    output_dir = _ensure_dir(output_dir)
    z = load_npz("weibel_landscape_deltaK.npz")
    paths: List[Path] = []
    AA, BB = z["AA"], z["BB"]
    for i, t in enumerate(z["times"]):
        fig, ax = plt.subplots(figsize=(5.2, 4.3))
        im = ax.contourf(AA, BB, z["deltaK"][i], levels=40)
        ax.plot(0.001, -0.001, marker="o", markersize=4, label="analytic cancellation")
        ax.set_xlabel(r"$A_B$")
        ax.set_ylabel(r"$B_B$")
        ax.set_title(fr"Weibel landscape, $t={int(t)}$")
        ax.legend(frameon=False, loc="best")
        fig.colorbar(im, ax=ax, label=r"$\Delta K$")
        paths.append(_save(fig, output_dir / f"Figure-6-Landscape-t-{int(t)}.regenerated.png"))
    return paths


def plot_weibel_sensitivity(output_dir: str | Path) -> List[Path]:
    """Regenerate the sensitivity-to-mismatch panels in Figure 7."""
    output_dir = _ensure_dir(output_dir)
    z = load_npz("weibel_deviation_histories.npz")
    paths: List[Path] = []
    deltas = z["deltas"]
    t = z["time"]

    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    for i, d in enumerate(deltas):
        ax.semilogy(t, z["E_y"][i] + z["B"][i] + 1e-20, label=fr"$\delta={d:g}$")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\mathcal{E}_y+\mathcal{B}$")
    ax.set_title("Control mismatch: transverse field energy")
    ax.legend(frameon=False)
    paths.append(_save(fig, output_dir / "Figure-7-field-energy-vs-deviation.regenerated.png"))

    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    for i, d in enumerate(deltas):
        ax.semilogy(t, np.abs(z["F_B"][i]) + 1e-20, label=fr"$\delta={d:g}$")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$|\hat B_3|$")
    ax.set_title("Control mismatch: magnetic Fourier mode")
    ax.legend(frameon=False)
    paths.append(_save(fig, output_dir / "Figure-7-Fourier-vs-deviation.regenerated.png"))
    return paths


def plot_two_stream_equilibrium(output_dir: str | Path) -> List[Path]:
    """Regenerate the two-stream equilibrium panel."""
    output_dir = _ensure_dir(output_dir)
    vx = np.linspace(-1.8, 1.8, 241)
    vy = np.linspace(-1.2, 1.2, 241)
    VX, VY = np.meshgrid(vx, vy, indexing="ij")
    mu = two_stream_equilibrium(VX, VY)

    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    im = ax.contourf(VX, VY, mu, levels=40)
    ax.set_xlabel(r"$v_x$")
    ax.set_ylabel(r"$v_y$")
    ax.set_title("Two-stream equilibrium")
    fig.colorbar(im, ax=ax, label=r"$\mu(v_x,v_y)$")
    return [_save(fig, output_dir / "Figure-8a-equilibrium.regenerated.png")]


def plot_two_stream_optimizer(output_dir: str | Path) -> List[Path]:
    """Regenerate the nonlinear-control optimization loss and parameter table."""
    output_dir = _ensure_dir(output_dir)
    z = load_npz("two_stream_optimization.npz")
    paths: List[Path] = []

    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    loss = np.asarray(z["loss_history"])
    epoch = np.arange(1, loss.size + 1)
    ax.loglog(epoch, loss)
    ax.scatter([int(z["epoch"])], [float(z["best_objective"])], marker="o", s=25)
    ax.set_xlabel("optimization step")
    ax.set_ylabel("objective")
    ax.set_title("Two-stream control optimization")
    paths.append(_save(fig, output_dir / "Figure-10-loglog-loss.regenerated.png"))

    params = two_stream_best_params()
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    ax.axis("off")
    table = ax.table(
        cellText=[[f"{x:.6g}" for x in row] for row in params],
        rowLabels=[f"k={i}" for i in range(1, params.shape[0] + 1)],
        colLabels=["a_k", "b_k", "c_k", "d_k"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    ax.set_title("Best two-stream control coefficients")
    paths.append(_save(fig, output_dir / "Table-1-two-stream-best-params.regenerated.png"))
    return paths


def plot_two_stream_landscapes(output_dir: str | Path) -> List[Path]:
    """Regenerate selected two-stream objective landscapes."""
    output_dir = _ensure_dir(output_dir)
    z = load_npz("two_stream_landscapes.npz")
    paths: List[Path] = []

    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    im = ax.contourf(z["A_ad"], z["D_ad"], z["deltaK_ad"], levels=40)
    ax.set_xlabel("a")
    ax.set_ylabel("d")
    ax.set_title("Two-stream objective slice: a-d")
    fig.colorbar(im, ax=ax, label=r"$K(T)-K(0)$")
    paths.append(_save(fig, output_dir / "fig_landscape_a_d.regenerated.png"))

    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    im = ax.contourf(z["B_bc"], z["C_bc"], z["deltaK_bc"], levels=40)
    ax.set_xlabel("b")
    ax.set_ylabel("c")
    ax.set_title("Two-stream objective slice: b-c")
    fig.colorbar(im, ax=ax, label=r"$K(T)-K(0)$")
    paths.append(_save(fig, output_dir / "fig_landscape_b_c.regenerated.png"))

    obj = z["obj_full"]
    A, B, C, D = z["A_full"], z["B_full"], z["C_full"], z["D_full"]
    cd_min = obj.min(axis=(0, 1))
    C_grid, D_grid = np.meshgrid(C, D, indexing="ij")
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    im = ax.contourf(C_grid, D_grid, cd_min, levels=40)
    ax.set_xlabel("c")
    ax.set_ylabel("d")
    ax.set_title("Two-stream landscape, min over a-b")
    fig.colorbar(im, ax=ax, label=r"$K(T)-K(0)$")
    paths.append(_save(fig, output_dir / "fig_landscape_min_cd.regenerated.png"))
    return paths


def make_all_outputs(output_root: str | Path | None = None) -> Dict[str, List[Path]]:
    """Create all notebook outputs.

    Returns a dictionary with two keys:
    - ``regenerated``: figures produced from compact numeric arrays.
    - ``submitted_reference``: exact PNGs copied from the submitted manuscript folder.
    """
    if output_root is None:
        output_root = PACKAGE_ROOT / "outputs" / "figures"
    output_root = Path(output_root)
    regenerated_dir = _ensure_dir(output_root / "regenerated")
    submitted_dir = _ensure_dir(output_root / "submitted_reference")

    regenerated: List[Path] = []
    for func in (
            plot_weibel_dispersion,
            plot_weibel_equilibria,
            plot_weibel_energy_histories,
            plot_weibel_fourier_modes,
            plot_weibel_distribution_samples,
            plot_weibel_landscapes,
            plot_weibel_sensitivity,
            plot_two_stream_equilibrium,
            plot_two_stream_optimizer,
            plot_two_stream_landscapes,
    ):
        _progress(f"Generating {func.__name__}...")
        generated_now = func(regenerated_dir)
        _progress(f"Finished {func.__name__}: {len(generated_now)} files")
        regenerated.extend(generated_now)

    submitted_reference = copy_submitted_reference_figures(submitted_dir, overwrite=True)
    return {"regenerated": regenerated, "submitted_reference": submitted_reference}
