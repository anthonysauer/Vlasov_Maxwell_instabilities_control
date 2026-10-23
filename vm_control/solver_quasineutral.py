"""Core 1.5D Vlasov--Maxwell Strang-splitting solver used by the paper experiments,
with the x-field given by the quasineutral (massless-electron) equation
    eps^2 d_xx U = exp(U) - rho,   E_x = -d_x U.

The implementation is a cleaned extraction of the notebook solver: JAX/JIT kernels,
periodic Fourier updates in x, velocity interpolation, and electromagnetic diagnostics.
"""

from functools import partial

import jax.numpy as jnp
from jax import jit, vmap, lax


@jit
def bispline_interp(xnew, ynew, xp, yp, zp):
    """
    (xnew,ynew): two 1D vector  of same size where to perform predictions  f(xnew[i],ynew[i])
    (xp,yp): original grid points 1D vector
    zp: original values of functions  zp[i,j] = value at xp[i], yp[j]
    """

    M = 1. / 16 * jnp.array([[0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, -8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 16, -40, 32, -8, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, -8, 24, -24, 8, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, -8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
                             [4, 0, -4, 0, 0, 0, 0, 0, -4, 0, 4, 0, 0, 0, 0, 0],
                             [-8, 20, -16, 4, 0, 0, 0, 0, 8, -20, 16, -4, 0, 0, 0, 0],
                             [4, -12, 12, -4, 0, 0, 0, 0, -4, 12, -12, 4, 0, 0, 0, 0],
                             [0, 16, 0, 0, 0, -40, 0, 0, 0, 32, 0, 0, 0, -8, 0, 0],
                             [-8, 0, 8, 0, 20, 0, -20, 0, -16, 0, 16, 0, 4, 0, -4, 0],
                             [16, -40, 32, -8, -40, 100, -80, 20, 32, -80, 64, -16, -8, 20, -16, 4],
                             [-8, 24, -24, 8, 20, -60, 60, -20, -16, 48, -48, 16, 4, -12, 12, -4],
                             [0, -8, 0, 0, 0, 24, 0, 0, 0, -24, 0, 0, 0, 8, 0, 0],
                             [4, 0, -4, 0, -12, 0, 12, 0, 12, 0, -12, 0, -4, 0, 4, 0],
                             [-8, 20, -16, 4, 24, -60, 48, -12, -24, 60, -48, 12, 8, -20, 16, -4],
                             [4, -12, 12, -4, -12, 36, -36, 12, 12, -36, 36, -12, -4, 12, -12, 4]]
                            )

    M1 = jnp.array([[1., 0., 0., 0.],
                    [-1., 1., 0., 0.],
                    [-1., 0., 1., 0.],
                    [1., -1., -1., 1.]])

    def built_Ivec(zp, ix, iy):
        return jnp.array([zp[ix + i, iy + j] for j in range(-1, 3) for i in range(-1, 3)])

    def built_Ivec1(zp, ix, iy):
        return jnp.array([zp[ix + i, iy + j] for j in range(0, 2) for i in range(0, 2)])

    def compute_basis(x, order=3):
        """
        x in [0,1]
        """
        return jnp.array([x ** i for i in jnp.arange(0, order + 1)])

    def tval(xnew, ix, xp):
        return (xnew - xp[ix - 1]) / (xp[ix] - xp[ix - 1])

    ix = jnp.clip(jnp.searchsorted(xp, xnew, side="right"), 0, len(xp) - 1)
    iy = jnp.clip(jnp.searchsorted(yp, ynew, side="right"), 0, len(yp) - 1)

    def bilinear_interp(ix, iy):
        Iv = built_Ivec1(zp, ix - 1, iy - 1)
        av = M1 @ Iv
        amtx = av.reshape(2, 2, -1)
        tx = tval(xnew, ix, xp)
        ty = tval(ynew, iy, yp)
        basis_x = compute_basis(tx, order=1)
        basis_y = compute_basis(ty, order=1)
        res = jnp.einsum("i...,ij...,j...", basis_y, amtx, basis_x)
        return res

    def bispline_interp(ix, iy):
        Iv = built_Ivec(zp, ix - 1, iy - 1)
        av = M @ Iv
        amtx = av.reshape(4, 4, -1)
        tx = tval(xnew, ix, xp)
        ty = tval(ynew, iy, yp)
        basis_x = compute_basis(tx)
        basis_y = compute_basis(ty)
        res = jnp.einsum("i...,ij...,j...", basis_y, amtx, basis_x)
        return res

    condx = jnp.logical_and(ix >= 2, ix <= len(xp) - 2)
    condy = jnp.logical_and(iy >= 2, iy <= len(yp) - 2)

    cond = jnp.logical_and(condx, condy)
    return jnp.where(cond,
                     bispline_interp(ix, iy),
                     bilinear_interp(ix, iy))


@jit
def compute_rho(f, dvx, dvy):
    """
    Compute charge density rho from distribution function f.
    """
    rho = jnp.sum(f, axis=(1, 2)) * dvx * dvy
    return rho


@partial(jit, static_argnums=(3,))
def solve_quasineutral(rho, grid_x, eps, n_newton=6):
    """
    Solve the massless-electron (quasineutral) field equation on the periodic grid

        eps^2 * d_xx U = exp(U) - rho,        E = - d_x U,

    by Newton's method. The equation is nonlinear in U, so it cannot be inverted
    in Fourier space like the linear Poisson equation; instead each Newton step
    solves a dense N_x x N_x linear system (cheap for the 1D grids used here).
    The spectral second-derivative matrix is used so that E is consistent with
    compute_derivative_fourier.

    The initial guess U = log(rho) is the eps -> 0 (quasineutral) limit.

    Parameters:
    - rho: 1D array of ion charge density (shape: [N_x]).
    - grid_x: 1D array of x-coordinates (shape: [N_x]).
    - eps: Scaling parameter (scalar).
    - n_newton: Number of Newton iterations (static integer).

    Returns:
    - E: 1D array of electric field E = -d_x U (shape: [N_x]).
    - U: 1D array of electric potential (shape: [N_x]).
    """
    N_x = grid_x.shape[0]
    dx = grid_x[1] - grid_x[0]
    k = jnp.fft.fftfreq(N_x, d=dx) * 2.0 * jnp.pi  # Shape: [N_x]

    # Dense spectral second-derivative matrix: D2 = Re( F^{-1} diag(-k^2) F )
    D2 = jnp.real(jnp.fft.ifft(-(k[:, None] ** 2) * jnp.fft.fft(jnp.eye(N_x), axis=0), axis=0))
    L = eps ** 2 * D2  # Shape: [N_x, N_x]

    # Quasineutral limit as initial guess (clip to guard against tiny negative rho from interpolation)
    U0 = jnp.log(jnp.maximum(rho, 1e-12))

    def newton_step(_, U):
        expU = jnp.exp(U)
        residual = L @ U - expU + rho  # R(U)  = eps^2 d_xx U - e^U + rho
        jacobian = L - jnp.diag(expU)  # R'(U) = eps^2 d_xx - e^U   (symmetric negative definite)
        delta = jnp.linalg.solve(jacobian, -residual)
        return U + delta

    U = lax.fori_loop(0, n_newton, newton_step, U0)

    # E = -d_x U, evaluated spectrally
    E = -compute_derivative_fourier(U, grid_x)

    return E, U


@jit
def update_f_E_drift_system(grid_x, grid_vx, grid_vy, f_old, Ex_old, Ey_old, delta_t, eps):
    """
    Update the distribution function f and electric fields E_x and E_y using the drift equation.

    Parameters:
    - grid_x: 1D array of x-coordinates (shape: [N_x]).
    - grid_vx: 1D array of v_x-coordinates (shape: [N_vx]).
    - grid_vy: 1D array of v_y-coordinates (shape: [N_vy]).
    - f_old: 3D array of distribution function at current time (shape: [N_x, N_vx, N_vy]).
    - Ex_old: 1D array of E_x electric field at current time (shape: [N_x]).
    - Ey_old: 1D array of E_y electric field at current time (shape: [N_x]).
    - delta_t: Time step (scalar).
    - eps: Scaling parameter of the quasineutral field equation (scalar).

    Returns:
    - f_new: Updated distribution function (shape: [N_x, N_vx, N_vy]).
    - Ex_new: Updated E_x electric field (shape: [N_x]).
    - Ey_new: Updated E_y electric field (shape: [N_x]).
    """

    # This is code with threshold using for update of E_y

    # Number of points
    N_x = grid_x.shape[0]
    N_vx = grid_vx.shape[0]
    N_vy = grid_vy.shape[0]

    # Compute grid spacing
    dx = grid_x[1] - grid_x[0]
    dvx = grid_vx[1] - grid_vx[0]
    dvy = grid_vy[1] - grid_vy[0]

    # Fourier wave numbers
    k = jnp.fft.fftfreq(N_x, d=dx) * 2.0 * jnp.pi  # Shape: [N_x]
    ik = 1j * k
    # To handle division by zero for k=0, we'll set ik=1 for k=0 temporarily
    ik_safe = jnp.where(k != 0, ik, 1.0)

    # Fourier transform of f_old along x
    f_hat = jnp.fft.fft(f_old, axis=0)  # Shape: [N_x, N_vx, N_vy]

    # Reshape k, v_x, and v_y for broadcasting
    k_reshaped = k[:, None, None]  # Shape: [N_x, 1, 1]
    v_x_reshaped = grid_vx[None, :, None]  # Shape: [1, N_vx, 1]
    v_y_reshaped = grid_vy[None, None, :]  # Shape: [1, 1, N_vy]

    # Compute the exponential factor e^{-i k v_x Δt}
    exp_factor = jnp.exp(-1j * k_reshaped * v_x_reshaped * delta_t)  # Shape: [N_x, N_vx, N_vy]

    # Update f_hat in Fourier space
    f_new_hat = exp_factor * f_hat  # Shape: [N_x, N_vx, N_vy]

    # Inverse Fourier transform to get f_new in physical space
    f_new = jnp.fft.ifft(f_new_hat, axis=0).real  # Taking the real part assuming f is real

    # Fourier transform of E_y_old (E_x is recomputed from rho below, so its transform is not needed)
    Ey_hat = jnp.fft.fft(Ey_old)  # Shape: [N_x]

    # ---------------------------
    # Update E_x
    # ---------------------------

    # In the Vlasov--Poisson version, E_x was advanced by Ampere's law in Fourier space,
    #     Ex_new_hat = Ex_hat + (1/ik) * int (exp_factor - 1) f_hat dv,
    # which (since Gauss's law ik E_hat = rho_hat - 1_hat holds) is exactly equivalent to
    # re-solving the linear Poisson equation with the transported density.
    #
    # In the quasineutral (massless-electron) model the E_x -- rho relation is nonlinear,
    #     eps^2 d_xx U = exp(U) - rho,   E_x = -d_x U,
    # so there is no equivalent linear Fourier update: E_x has to be recomputed from
    # the transported density directly.
    rho_new = compute_rho(f_new, dvx, dvy)  # Shape: [N_x]
    Ex_new, _ = solve_quasineutral(rho_new, grid_x, eps)  # Shape: [N_x]

    # ---------------------------
    # Update E_y
    # ---------------------------

    # Compute the ratio v_y / v_x, handling v_x = 0
    # Shape: [1, N_vx, N_vy]
    v_x_safe = jnp.where(v_x_reshaped != 0, v_x_reshaped, 1.0)
    ratio_vy_vx = jnp.where(v_x_reshaped != 0, v_y_reshaped / v_x_safe, 0.0)

    # Compute the integrand for E_y_new in Fourier space:
    # When v_x != 0: (v_y / v_x) * (exp_factor - 1) * f_hat
    # When v_x == 0: -1j * k * v_y * f_hat * delta_t
    integrand_Ey = jnp.where(
        v_x_reshaped != 0,
        ratio_vy_vx * (exp_factor - 1.0) * f_hat,
        -1j * k_reshaped * v_y_reshaped * f_hat * delta_t
    )  # Shape: [N_x, N_vx, N_vy]

    # Integrate over v_x and v_y using the rectangle rule: sum * dvx * dvy
    integral_Ey = jnp.sum(integrand_Ey, axis=(1, 2)) * dvx * dvy  # Shape: [N_x]

    # ---------------------------
    # Apply Threshold to integral_Ey
    # ---------------------------
    threshold = 1e-7  # Define a reasonable threshold for float32

    # Set integral_Ey to zero where its absolute value is below the threshold
    integral_Ey = jnp.where(jnp.abs(integral_Ey) < threshold, 0.0, integral_Ey)

    # Update Ey_hat
    Ey_new_hat = Ey_hat + (1.0 / ik_safe) * integral_Ey  # Shape: [N_x]

    # Handle k=0 separately to avoid division by zero
    # Physically, the k=0 mode corresponds to charge neutrality, so we set Ey_new_hat[0] = Ey_hat[0]
    Ey_new_hat = jnp.where(k != 0, Ey_new_hat, Ey_hat)

    # Inverse Fourier transform to get Ey_new in physical space
    Ey_new = jnp.fft.ifft(Ey_new_hat).real  # Shape: [N_x]

    return f_new, Ex_new, Ey_new


@jit
def compute_derivative_fourier(h, grid_x):
    """
    Compute the derivative of h with respect to x using Fourier transforms.
    
    Parameters:
    - h: 1D array of function values at grid points (shape: [N_x]).
    - grid_x: 1D array of x-coordinates (shape: [N_x]).
    
    Returns:
    - dh_dx: 1D array of derivative values at grid points (shape: [N_x]).
    """
    N_x = grid_x.shape[0]
    dx = grid_x[1] - grid_x[0]

    # Compute Fourier coefficients of h
    h_hat = jnp.fft.fft(h)

    # Compute wave numbers
    k = jnp.fft.fftfreq(N_x, d=dx) * 2.0 * jnp.pi  # Shape: [N_x]

    # Compute derivative in Fourier space: (ik) * h_hat
    dh_hat = 1j * k * h_hat

    # Inverse Fourier transform to get derivative in physical space
    dh_dx = jnp.fft.ifft(dh_hat).real  # Assuming h is real

    return dh_dx


@jit
def Hf_update(grid_x, grid_vx, grid_vy, f_old, Ex_old, Ey_old, B_old, delta_t, eps):
    """
    Update the distribution function f and electric fields E_x and E_y using the first subsystem equations:
    
    ∂_t f + v_x ∂_x f = 0,
    ∂_t E = -J,
    ∂_t B = 0,
    ∂_t r_x = ∂_t r_y = 0.
    
    Parameters:
    - grid_x: 1D array of x-coordinates (shape: [N_x]).
    - grid_vx: 1D array of v_x-coordinates (shape: [N_vx]).
    - grid_vy: 1D array of v_y-coordinates (shape: [N_vy]).
    - f_old: 3D array of distribution function at current time (shape: [N_x, N_vx, N_vy]).
    - Ex_old: 1D array of E_x electric field at current time (shape: [N_x]).
    - Ey_old: 1D array of E_y electric field at current time (shape: [N_x]).
    - B_old: 1D array of B field at current time (shape: [N_x]).
    - delta_t: Time step (scalar).
    
    Returns:
    - f_new: Updated distribution function (shape: [N_x, N_vx, N_vy]).
    - Ex_new: Updated E_x electric field (shape: [N_x]).
    - Ey_new: Updated E_y electric field (shape: [N_x]).
    - B_new: Unchanged B field (shape: [N_x]).
    """
    # Update f, E_x, and E_y using the existing drift update function
    f_new, Ex_new, Ey_new = update_f_E_drift_system(grid_x, grid_vx, grid_vy, f_old, Ex_old, Ey_old, delta_t, eps)

    # B, r_x, and r_y remain unchanged in this subsystem
    B_new = B_old

    return f_new, Ex_new, Ey_new, B_new


@jit
def HE_update(grid_x, grid_vx, grid_vy, f_old, Ex_old, Ey_old, B_old, delta_t):
    """
    Update the distribution function f, magnetic field B, and quantities r_x and r_y
    using the second subsystem equations.
    
    Parameters:
    - grid_x: 1D array of x-coordinates (shape: [N_x]).
    - grid_vx: 1D array of v_x-coordinates (shape: [N_vx]).
    - grid_vy: 1D array of v_y-coordinates (shape: [N_vy]).
    - f_old: 3D array of distribution function at current time (shape: [N_x, N_vx, N_vy]).
    - Ex_old: 1D array of E_x electric field at current time (shape: [N_x]).
    - Ey_old: 1D array of E_y electric field at current time (shape: [N_x]).
    - B_old: 1D array of B field at current time (shape: [N_x]).
    - delta_t: Time step (scalar).
    
    Returns:
    - f_new: Updated distribution function (shape: [N_x, N_vx, N_vy]).
    - Ex_new: Unchanged E_x electric field (shape: [N_x]).
    - Ey_new: Unchanged E_y electric field (shape: [N_x]).
    - B_new: Updated B field (shape: [N_x]).
    """

    # ---------------------------
    # Step 1: Update B_new
    # ---------------------------
    d_x_Ey = compute_derivative_fourier(Ey_old, grid_x)
    B_new = B_old - delta_t * d_x_Ey  # Shape: [N_x]

    # ---------------------------
    # Step 2: Update f using bispline interpolation
    # ---------------------------

    # Compute shifts
    shift_vx = delta_t * Ex_old  # Shape: [N_x]
    shift_vy = delta_t * Ey_old  # Shape: [N_x]

    def interpolate_f_per_x(x_shift, y_shift, f_x):
        """
        Perform bispline interpolation for a single x-slice.
        
        Parameters:
        - x_shift: Scalar shift in v_x for this x.
        - y_shift: Scalar shift in v_y for this x.
        - f_x: 2D array of f values at this x (shape: [N_vx, N_vy]).
        
        Returns:
        - f_new_x: 2D array of interpolated f values at this x (shape: [N_vx, N_vy]).
        """
        # Compute shifted coordinates
        v_x_new = grid_vx - x_shift  # Shape: [N_vx]
        v_y_new = grid_vy - y_shift  # Shape: [N_vy]

        # Create meshgrid for shifted coordinates
        X_new, Y_new = jnp.meshgrid(v_x_new, v_y_new, indexing='ij')  # Shape: [N_vx, N_vy]

        # Flatten the shifted coordinates for interpolation
        x_new_flat = X_new.ravel()  # Shape: [N_vx * N_vy]
        y_new_flat = Y_new.ravel()  # Shape: [N_vx * N_vy]

        # Perform bispline interpolation
        f_interp_flat = bispline_interp(x_new_flat, y_new_flat, grid_vx, grid_vy, f_x)  # Shape: [N_vx * N_vy]

        # Reshape back to [N_vx, N_vy]
        f_new_x = f_interp_flat.reshape((grid_vx.size, grid_vy.size))  # Shape: [N_vx, N_vy]

        return f_new_x

    # Vectorize the interpolation over all x-slices using vmap
    interpolate_f_vmapped = vmap(interpolate_f_per_x, in_axes=(0, 0, 0))

    # Apply the interpolation to get f_new
    f_new = interpolate_f_vmapped(shift_vx, shift_vy, f_old)  # Shape: [N_x, N_vx, N_vy]

    # ---------------------------
    # Step 3: E_new remains the same as E_old
    # ---------------------------
    Ex_new = Ex_old
    Ey_new = Ey_old

    # ---------------------------
    # Return updated variables
    # ---------------------------
    return f_new, Ex_new, Ey_new, B_new


@jit
def HB_update(grid_x, grid_vx, grid_vy, f_old, Ex_old, Ey_old, B_old, delta_t):
    """
    Update the distribution function f, electric field E_y, and ensure B and r_x, r_y remain unchanged.

    Parameters:
    - grid_x: 1D array of x-coordinates.
    - grid_vx: 1D array of v_x coordinates.
    - grid_vy: 1D array of v_y coordinates.
    - f_old: 3D array of shape [N_x, N_vx, N_vy] representing the distribution function at current time.
    - Ex_old: 1D array of shape [N_x] representing E_x electric field (unchanged).
    - Ey_old: 1D array of shape [N_x] representing E_y electric field.
    - B_old: 1D array of shape [N_x] representing B field (unchanged).
    - delta_t: Time step.

    Returns:
    - f_new: Updated distribution function [N_x, N_vx, N_vy].
    - Ex_new: Unchanged E_x electric field [N_x].
    - Ey_new: Updated E_y electric field [N_x].
    - B_new: Unchanged B field [N_x].
    """

    # ---------------------------
    # Compute partial_x B_old using Fourier derivative
    # ---------------------------
    partial_x_B = compute_derivative_fourier(B_old, grid_x)  # Shape: [N_x]

    # ---------------------------
    # Update E_y
    # ---------------------------
    Ey_new = Ey_old - delta_t * partial_x_B  # Shape: [N_x]

    # ---------------------------
    # B_new remains the same as B_old
    # ---------------------------
    B_new = B_old

    # ---------------------------
    # Update f using rotation and bilinear interpolation
    # ---------------------------

    # Define the skew-symmetric matrix J
    J = jnp.array([[0.0, 1.0],
                   [-1.0, 0.0]])

    # Compute rotation angle theta = B_old * delta_t
    theta = - B_old * delta_t  # Shape: [N_x]
    # theta =  B_old * delta_t  # Shape: [N_x]

    # Compute rotation matrices R = exp(J theta) = [[cos(theta), sin(theta)], [-sin(theta), cos(theta)]]
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    R = jnp.stack([jnp.stack([cos_theta, sin_theta], axis=1),
                   jnp.stack([-sin_theta, cos_theta], axis=1)], axis=1)  # Shape: [N_x, 2, 2]

    # Define a helper function to rotate and interpolate f for a single x
    def rotate_and_interpolate(x_idx):
        """
        Rotate the velocity coordinates and interpolate f_old at the rotated positions.

        Parameters:
        - x_idx: Index along the spatial grid.

        Returns:
        - f_new_slice: 2D array of shape [N_vx, N_vy] representing f_new at this x.
        """
        # Extract the rotation matrix for this x
        R_x = R[x_idx]  # Shape: [2,2]

        # Extract the velocity grids
        Vx, Vy = jnp.meshgrid(grid_vx, grid_vy, indexing='ij')  # Shape: [N_vx, N_vy]

        # Flatten the velocity grids
        V = jnp.stack([Vx.flatten(), Vy.flatten()], axis=1).T  # Shape: [2, N_vx*N_vy]

        # Apply rotation: v_star = R_x @ v
        V_star = R_x @ V  # Shape: [2, N_vx*N_vy]

        # Extract rotated velocities
        Vx_star = V_star[0, :]  # Shape: [N_vx*N_vy]
        Vy_star = V_star[1, :]  # Shape: [N_vx*N_vy]

        # Perform bilinear interpolation
        f_new_flat = bispline_interp(Vx_star, Vy_star, grid_vx, grid_vy, f_old[x_idx])  # Shape: [N_vx*N_vy]

        # Reshape back to [N_vx, N_vy]
        f_new_slice = f_new_flat.reshape((len(grid_vx), len(grid_vy)))

        return f_new_slice

    # Vectorize the helper function over all x indices
    f_new = vmap(rotate_and_interpolate)(jnp.arange(len(grid_x)))  # Shape: [N_x, N_vx, N_vy]

    # ---------------------------
    # E_x remains unchanged
    # ---------------------------
    Ex_new = Ex_old

    return f_new, Ex_new, Ey_new, B_new


@partial(jit, static_argnums=(7,))  # N_steps is the 8th argument, index 7
def solver_quasineutral_jit(f_iv, B_iv, E_y_iv, grid_x, grid_vx, grid_vy, delta_t, N_steps, k_track, eps=1e-3):
    """
    Solver to evolve the system using Strang splitting, including kinetic energy calculation
    and tracking of specified Fourier modes.

    Parameters:
    - f_iv: Initial distribution function (3D array [N_x, N_vx, N_vy]).
    - B_iv: Initial magnetic field (1D array [N_x]).
    - E_y_iv: Initial electric field E_y (1D array [N_x]).
    - grid_x: 1D array of x-coordinates (shape: [N_x]).
    - grid_vx: 1D array of v_x-coordinates (shape: [N_vx]).
    - grid_vy: 1D array of v_y-coordinates (shape: [N_vy]).
    - delta_t: Time step (scalar).
    - N_steps: Number of iterations (integer).
    - k_track: Integer specifying the wave number to track in Fourier modes.
    - eps: Scaling parameter of the quasineutral field equation (default 1e-3).

    Returns:
    - Tuple containing:
        - f_end: Final distribution function (3D array [N_x, N_vx, N_vy]).
        - electric_energy_E_x: Array of electric energy 0.5 * int E_x^2 dx over time (1D array).
        - electric_energy_E_y: Array of electric energy from E_y over time (1D array).
        - magnetic_energy: Array of magnetic energy over time (1D array).
        - kinetic_energy: Array of kinetic energy over time (1D array).
        - Fourier_mode_B: Array of Fourier coefficients of B at k_track over time (1D array of complex numbers).
        - Fourier_mode_E_x: Array of Fourier coefficients of E_x at k_track over time (1D array of complex numbers).
        - Fourier_mode_E_y: Array of Fourier coefficients of E_y at k_track over time (1D array of complex numbers).
        - E_x_final: Final E_x electric field (1D array).
        - E_y_final: Final E_y electric field (1D array).
        - B_final: Final B magnetic field (1D array).
    """

    # Compute grid spacings
    dx = grid_x[1] - grid_x[0]
    dvx = grid_vx[1] - grid_vx[0]
    dvy = grid_vy[1] - grid_vy[0]

    # Compute initial charge density rho_iv from f_iv
    rho_iv = compute_rho(f_iv, dvx, dvy)  # Shape: [N_x]

    # Solve the quasineutral field equation to obtain initial E_x_iv (and U_iv)
    E_x_iv, U_iv = solve_quasineutral(rho_iv, grid_x, eps)  # Shape: [N_x]

    # Initialize state variables
    f_old = f_iv
    Ex_old = E_x_iv
    Ey_old = E_y_iv
    B_old = B_iv

    # Threshold check function
    @jit
    def apply_threshold(new_val, old_val, threshold=2e-8):
        return jnp.where(jnp.abs(new_val - old_val) < threshold, old_val, new_val)

    # Define the single iteration step function
    @jit
    def iteration_step(carry, _):
        """
        Perform one iteration step using Strang splitting and track Fourier modes.

        Parameters:
        - carry: Tuple containing current state variables.
        - _: Placeholder for scan (unused).

        Returns:
        - Updated carry with new state variables.
        - Outputs to be recorded (energies, r histories, Fourier modes).
        """
        f_old, Ex_old, Ey_old, B_old = carry

        # Strang Splitting Steps:
        # 1. Hf for dt/2
        f_new, Ex_new, Ey_new, B_new = Hf_update(
            grid_x, grid_vx, grid_vy,
            f_old, Ex_old, Ey_old,
            B_old, delta_t / 2.0,
            eps
        )

        # 2. HE for dt/2
        f_new, Ex_new, Ey_new, B_new = HE_update(
            grid_x, grid_vx, grid_vy,
            f_new, Ex_new, Ey_new,
            B_new, delta_t / 2.0,
        )

        # 3. HB for dt
        f_new, Ex_new, Ey_new, B_new = HB_update(
            grid_x, grid_vx, grid_vy,
            f_new, Ex_new, Ey_new,
            B_new, delta_t,
        )

        # 4. HE for dt/2
        f_new, Ex_new, Ey_new, B_new = HE_update(
            grid_x, grid_vx, grid_vy,
            f_new, Ex_new, Ey_new,
            B_new, delta_t / 2.0,
        )

        # 5. Hf for dt/2
        f_new, Ex_new, Ey_new, B_new = Hf_update(
            grid_x, grid_vx, grid_vy,
            f_new, Ex_new, Ey_new,
            B_new, delta_t / 2.0,
            eps
        )

        f_new = apply_threshold(f_new, f_old)
        Ex_new = apply_threshold(Ex_new, Ex_old)
        Ey_new = apply_threshold(Ey_new, Ey_old)
        B_new = apply_threshold(B_new, B_old)

        # Update state variables for next iteration
        carry = (f_new, Ex_new, Ey_new, B_new)

        # Compute energies
        # Electric energy in x
        electric_E_x_energy = 0.5 * jnp.sum(Ex_new ** 2) * dx
        electric_E_y_energy = 0.5 * jnp.sum(Ey_new ** 2) * dx
        magnetic_Energy = 0.5 * jnp.sum(B_new ** 2) * dx
        kinetic_Energy = jnp.sum(
            f_new * (grid_vx[None, :, None] ** 2 + grid_vy[None, None, :] ** 2) / 2) * dvx * dvy * dx

        # Compute Fourier modes at k_track
        # Compute FFT of B_new, Ex_new, Ey_new
        B_hat = jnp.fft.fft(B_new)
        E_x_hat = jnp.fft.fft(Ex_new)
        E_y_hat = jnp.fft.fft(Ey_new)

        # Extract the Fourier coefficient at wave number k_track
        Fourier_mode_B = B_hat[k_track]
        Fourier_mode_E_x = E_x_hat[k_track]
        Fourier_mode_E_y = E_y_hat[k_track]

        # Outputs to collect
        outputs = (
            electric_E_x_energy,
            electric_E_y_energy,
            magnetic_Energy,
            kinetic_Energy,
            Fourier_mode_B,
            Fourier_mode_E_x,
            Fourier_mode_E_y
        )

        return carry, outputs

    # Initialize the carry for scan
    initial_carry = (
        f_old,  # f_old
        Ex_old,  # Ex_old
        Ey_old,  # Ey_old
        B_old,  # B_old
    )

    # Perform the scan over N_steps iterations
    final_carry, outputs = lax.scan(iteration_step, initial_carry, None, length=N_steps)

    # Unpack the final carry
    f_end, Ex_final, Ey_final, B_final = final_carry

    # Unpack the outputs
    (electric_energy_E_x,
     electric_energy_E_y,
     magnetic_energy,
     kinetic_energy,
     Fourier_mode_B,
     Fourier_mode_E_x,
     Fourier_mode_E_y) = outputs

    # Collect the Fourier modes history
    Fourier_mode_B_history = Fourier_mode_B
    Fourier_mode_E_x_history = Fourier_mode_E_x
    Fourier_mode_E_y_history = Fourier_mode_E_y

    # Return the final state and histories
    return (
        f_end,
        electric_energy_E_x,
        electric_energy_E_y,
        magnetic_energy,
        kinetic_energy,
        Fourier_mode_B_history,
        Fourier_mode_E_x_history,
        Fourier_mode_E_y_history,
        Ex_final,
        Ey_final,
        B_final
    )
