"""Core 1.5D Vlasov--Maxwell Strang-splitting solver used by the paper experiments.

The implementation is a cleaned extraction of the notebook solver: JAX/JIT kernels,
periodic Fourier updates in x, velocity interpolation, and electromagnetic diagnostics.
"""

from functools import partial

import jax.numpy as jnp
from jax import jit, vmap, lax

@jit
def bispline_interp(xnew,ynew,xp,yp,zp):
    """
    (xnew,ynew): two 1D vector  of same size where to perform predictions  f(xnew[i],ynew[i])
    (xp,yp): original grid points 1D vector
    zp: original values of functions  zp[i,j] = value at xp[i], yp[j]
    """
    
    
    M = 1./16 * jnp.array([[0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
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
    
    M1 = jnp.array([[1.,0.,0.,0.],
                    [-1.,1.,0.,0.],
                    [-1.,0.,1.,0.],
                    [1.,-1.,-1.,1.]])

    def built_Ivec(zp,ix,iy):
        return jnp.array([zp[ix+i,iy+j] for j in range(-1,3) for i in range(-1,3)])


    def built_Ivec1(zp,ix,iy):
        return jnp.array([zp[ix+i,iy+j] for j in range(0,2) for i in range(0,2)])

    
    
    def compute_basis(x,order=3):
        """
        x in [0,1]
        """ 
        return jnp.array([x**i for i in jnp.arange(0, order+1)])
    
    def tval(xnew,ix,xp):
        return (xnew-xp[ix-1])/(xp[ix]-xp[ix-1])
    
    ix = jnp.clip(jnp.searchsorted(xp, xnew, side="right"), 0, len(xp)-1)
    iy = jnp.clip(jnp.searchsorted(yp, ynew, side="right"), 0, len(yp)-1)

    def bilinear_interp(ix,iy):
        Iv = built_Ivec1(zp,ix-1,iy-1)
        av = M1 @ Iv
        amtx = av.reshape(2,2,-1)
        tx = tval(xnew,ix,xp)
        ty = tval(ynew,iy,yp)
        basis_x = compute_basis(tx,order=1)
        basis_y = compute_basis(ty,order=1)
        res = jnp.einsum("i...,ij...,j...",basis_y,amtx,basis_x)
        return res

    def bispline_interp(ix,iy):
        Iv = built_Ivec(zp,ix-1,iy-1)
        av = M @ Iv
        amtx = av.reshape(4,4,-1)
        tx = tval(xnew,ix,xp)
        ty = tval(ynew,iy,yp)
        basis_x = compute_basis(tx)
        basis_y = compute_basis(ty)
        res = jnp.einsum("i...,ij...,j...",basis_y,amtx,basis_x)
        return res
    
    condx = jnp.logical_and(ix>=2, ix<=len(xp)-2)
    condy = jnp.logical_and(iy>=2, iy<=len(yp)-2)
    
    cond = jnp.logical_and(condx,condy)
    return jnp.where(cond,
             bispline_interp(ix,iy),
             bilinear_interp(ix,iy))

@jit
def compute_rho(f, dvx, dvy):
    """
    Compute charge density rho from distribution function f.
    """
    rho = jnp.sum(f, axis=(1, 2)) * dvx * dvy
    return rho

@jit
def solve_poisson(rho, grid_x, dx):
    """
    Solve the Poisson equation in Fourier space to obtain electric field E.

    Parameters:
    - rho: 1D array of charge density (shape: [N_x]).
    - grid_x: 1D array of x-coordinates (shape: [N_x]).
    - dx: Grid spacing in x.

    Returns:
    - E: 1D array of electric field (shape: [N_x]).
    """
    N_x = grid_x.shape[0]
    # Compute Fourier wave numbers
    k = jnp.fft.fftfreq(N_x, d=dx) * 2.0 * jnp.pi  # Shape: [N_x]
    ik = 1j * k

    # Fourier transform of rho
    rho_hat = jnp.fft.fft(rho)  # Shape: [N_x]

    # Fourier transform of the constant function 1
    one_array = jnp.ones(N_x)
    one_hat = jnp.fft.fft(one_array)  # Shape: [N_x]
    # Note: For a constant function 1, the Fourier transform is [N_x, 0, 0, ..., 0]

    # Compute E_hat using Poisson's equation: E_k = (hat{1} - rho_hat) / (ik)
    # Handle k=0 to avoid division by zero by setting E_hat[0] = 0
    E_hat = jnp.where(k != 0, (- one_hat + rho_hat) / ik, 0.0)

    # Inverse Fourier transform to get E in physical space
    E = jnp.fft.ifft(E_hat).real  # Assuming E is real

    return E

@jit
def update_f_E_drift_system(grid_x, grid_vx, grid_vy, f_old, Ex_old, Ey_old, delta_t):
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
    k_reshaped = k[:, None, None]      # Shape: [N_x, 1, 1]
    v_x_reshaped = grid_vx[None, :, None]  # Shape: [1, N_vx, 1]
    v_y_reshaped = grid_vy[None, None, :]  # Shape: [1, 1, N_vy]

    # Compute the exponential factor e^{-i k v_x Δt}
    exp_factor = jnp.exp(-1j * k_reshaped * v_x_reshaped * delta_t)  # Shape: [N_x, N_vx, N_vy]

    # Update f_hat in Fourier space
    f_new_hat = exp_factor * f_hat  # Shape: [N_x, N_vx, N_vy]

    # Inverse Fourier transform to get f_new in physical space
    f_new = jnp.fft.ifft(f_new_hat, axis=0).real  # Taking the real part assuming f is real

    # Fourier transform of E_x_old and E_y_old
    Ex_hat = jnp.fft.fft(Ex_old)  # Shape: [N_x]
    Ey_hat = jnp.fft.fft(Ey_old)  # Shape: [N_x]

    # ---------------------------
    # Update E_x
    # ---------------------------

    # Compute the integrand for E_x_new in Fourier space: (exp_factor - 1) * f_hat
    integrand_Ex = (exp_factor - 1.0) * f_hat  # Shape: [N_x, N_vx, N_vy]

    # Integrate over v_x and v_y using the rectangle rule: sum * dvx * dvy
    integral_Ex = jnp.sum(integrand_Ex, axis=(1, 2)) * dvx * dvy  # Shape: [N_x]

    # Update Ex_hat
    Ex_new_hat = Ex_hat + (1.0 / ik_safe) * integral_Ex  # Shape: [N_x]

    # Handle k=0 separately to avoid division by zero
    # Physically, the k=0 mode corresponds to charge neutrality, so we set Ex_new_hat[0] = Ex_hat[0]
    Ex_new_hat = jnp.where(k != 0, Ex_new_hat, Ex_hat)

    # Inverse Fourier transform to get Ex_new in physical space
    Ex_new = jnp.fft.ifft(Ex_new_hat).real  # Shape: [N_x]

    # ---------------------------
    # Update E_y
    # ---------------------------

    # Compute the ratio v_y / v_x, handling v_x = 0
    # Shape: [1, N_vx, N_vy]
    ratio_vy_vx = jnp.where(v_x_reshaped != 0, v_y_reshaped / v_x_reshaped, 0.0)

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
def compute_kinetic_energy(f, grid_vx, grid_vy, dvx, dvy, dx):
    """
    Compute the kinetic energy:
        KE = 0.5 * ∫ |v|^2 f(x, v_x, v_y) dx dv_x dv_y
    
    Parameters:
    - f: Distribution function (3D array [N_x, N_vx, N_vy]).
    - grid_vx: 1D array of v_x-coordinates.
    - grid_vy: 1D array of v_y-coordinates.
    - dvx: Grid spacing in v_x.
    - dvy: Grid spacing in v_y.
    - dx: Grid spacing in x.
    
    Returns:
    - KE: Scalar kinetic energy.
    """
    # Create meshgrid for v_x and v_y
    VX, VY = jnp.meshgrid(grid_vx, grid_vy, indexing='ij')  # Shape: [N_vx, N_vy]
    
    # Compute |v|^2 = v_x^2 + v_y^2
    v_squared = VX**2 + VY**2  # Shape: [N_vx, N_vy]
    
    # Expand v_squared to match f's shape: [N_x, N_vx, N_vy]
    v_squared_expanded = v_squared[None, :, :]  # Shape: [1, N_vx, N_vy]
    
    # Compute kinetic energy density: 0.5 * |v|^2 * f
    ke_density = 0.5 * v_squared_expanded * f  # Shape: [N_x, N_vx, N_vy]
    
    # Integrate over space and velocity space: ∫ ke_density dx dv_x dv_y
    ke_integrated = jnp.sum(ke_density) * dvx * dvy * dx  # Scalar
    
    
    return ke_integrated


@jit
def compute_Jx(f, grid_vx, grid_vy, dvx, dvy):
    """
    Compute current density J_x by integrating v_x * f over v_x and v_y.
    
    Parameters:
    - f: 3D array of distribution function (shape: [N_x, N_vx, N_vy]).
    - grid_vx: 1D array of v_x coordinates (shape: [N_vx]).
    - grid_vy: 1D array of v_y coordinates (shape: [N_vy]).
    - dvx: Grid spacing in v_x.
    - dvy: Grid spacing in v_y.
    
    Returns:
    - Jx: 1D array of J_x (shape: [N_x]).
    """
    # Multiply v_x with f: broadcasting grid_vx over x and v_y
    Jx = jnp.sum(grid_vx[None, :, None] * f, axis=(1, 2)) * dvx * dvy
    return Jx

@jit
def compute_Jy(f, grid_vx, grid_vy, dvx, dvy):
    """
    Compute current density J_y by integrating v_y * f over v_x and v_y.
    
    Parameters:
    - f: 3D array of distribution function (shape: [N_x, N_vx, N_vy]).
    - grid_vx: 1D array of v_x coordinates (shape: [N_vx]).
    - grid_vy: 1D array of v_y coordinates (shape: [N_vy]).
    - dvx: Grid spacing in v_x.
    - dvy: Grid spacing in v_y.
    
    Returns:
    - Jy: 1D array of J_y (shape: [N_x]).
    """
    # Multiply v_y with f: broadcasting grid_vy over x and v_x
    Jy = jnp.sum(grid_vy[None, None, :] * f, axis=(1, 2)) * dvx * dvy
    return Jy

@jit
def compute_electric_energy(E, dx):
    """
    Compute the electric energy by integrating (1/2) * E^2 over x.
    
    Parameters:
    - E: 1D array of electric field (shape: [N_x]).
    - dx: Grid spacing in x.
    
    Returns:
    - electric_energy: Scalar representing the total electric energy.
    """
    electric_energy = 0.5 * jnp.sum(E**2) * dx
    return electric_energy


@jit
def compute_r(E, dx):
    """
    Compute the electric energy r from the electric field E.
    
    Parameters:
    - E: 1D array of electric field (shape: [N_x]).
    - dx: Grid spacing in x.
    
    Returns:
    - r: Scalar representing the total electric energy.
    """
    C = 1
    
    return jnp.sqrt(compute_electric_energy(E, dx) + C)

@jit
def compute_g(E, r):
    """
    Compute the ratio g = E / r for each point in the electric field.
    
    Parameters:
    - E: 1D array of electric field (shape: [N_x]).
    - r: Scalar representing the total electric energy.
    
    Returns:
    - g: 1D array of the ratio E/r (shape: [N_x]).
    """
    g = E / r
    return g

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
def update_r_x(delta_t, g_x, rho_old, r_x_old, J_x_old, dx):
    """
    Update the quantity r_x based on the provided formula.

    Parameters:
    - delta_t: Time step (scalar).
    - g_x: 1D array of g_x values at grid points (shape: [N_x]).
    - rho_old: 1D array of charge density at current time (shape: [N_x]).
    - r_x_old: Scalar or 1D array of r_x at current time (shape: [N_x] or scalar).
    - J_x_old: 1D array of current density at current time (shape: [N_x]).
    - dx: Spatial grid spacing (scalar).

    Returns:
    - r_x_new: Updated r_x (scalar or 1D array, matching the shape of r_x_old).
    """
    # Compute integrals using the rectangle (midpoint) rule
    integral_gx2_rho = jnp.sum(g_x**2 * rho_old) * dx      # Scalar
    integral_gx_Jx = jnp.sum(g_x * J_x_old) * dx          # Scalar

    # Compute the numerator
    numerator = r_x_old - (delta_t / 2.0) * integral_gx_Jx - (delta_t**2 / 8.0) * r_x_old * integral_gx2_rho

    # Compute the denominator
    denominator = 1.0 + (delta_t**2 / 8.0) * integral_gx2_rho

    # Update r_x
    r_x_new = numerator / denominator

    return r_x_new



@jit
def update_r_y(delta_t, g_y, d_x_g_y, rho_old, r_y_old, J_y_old, B_old, dx):
    """
    Update the quantity r_y based on the provided formula.

    Parameters:
    - delta_t: Time step (scalar).
    - g_y: 1D array of g_y values at grid points (shape: [N_x]).
    - d_x_g_y: 1D array of spatial derivatives of g_y (shape: [N_x]).
    - rho_old: 1D array of charge density at current time (shape: [N_x]).
    - r_y_old: Scalar or 1D array of r_y at current time (shape: [N_x] or scalar).
    - J_y_old: 1D array of current density at current time (shape: [N_x]).
    - B_old: 1D array of B field at current time (shape: [N_x]).
    - dx: Spatial grid spacing (scalar).

    Returns:
    - r_y_new: Updated r_y (scalar or 1D array, matching the shape of r_y_old).
    """
    # Compute integrals using the rectangle (midpoint) rule
    integral_gy2_rho = jnp.sum(g_y**2 * rho_old) * dx          # Scalar
    integral_dgx2 = jnp.sum(d_x_g_y**2) * dx                  # Scalar
    integral_gy_Jy = jnp.sum(g_y * J_y_old) * dx              # Scalar
    integral_dgx_By = jnp.sum(d_x_g_y * B_old) * dx            # Scalar

    # Compute the denominator
    denominator = 1.0 + (delta_t**2 / 8.0) * (integral_gy2_rho + integral_dgx2)

    # Compute the numerator
    numerator = (
        r_y_old
        - (delta_t / 2.0) * integral_gy_Jy
        - (delta_t**2 / 8.0) * r_y_old * integral_gy2_rho
        + (delta_t / 2.0) * integral_dgx_By
        - (delta_t**2 / 8.0) * r_y_old * integral_dgx2
    )

    # Update r_y
    r_y_new = numerator / denominator

    return r_y_new


@jit
def Hf_update(grid_x, grid_vx, grid_vy, f_old, Ex_old, Ey_old, B_old, r_x_old, r_y_old, delta_t):
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
    - r_x_old: Scalar or 1D array of r_x at current time (shape: [N_x] or scalar).
    - r_y_old: Scalar or 1D array of r_y at current time (shape: [N_x] or scalar).
    - delta_t: Time step (scalar).
    
    Returns:
    - f_new: Updated distribution function (shape: [N_x, N_vx, N_vy]).
    - Ex_new: Updated E_x electric field (shape: [N_x]).
    - Ey_new: Updated E_y electric field (shape: [N_x]).
    - B_new: Unchanged B field (shape: [N_x]).
    - r_x_new: Unchanged r_x (shape: [N_x] or scalar).
    - r_y_new: Unchanged r_y (shape: [N_x] or scalar).
    """
    # Update f, E_x, and E_y using the existing drift update function
    f_new, Ex_new, Ey_new = update_f_E_drift_system(grid_x, grid_vx, grid_vy, f_old, Ex_old, Ey_old, delta_t)
    
    # B, r_x, and r_y remain unchanged in this subsystem
    B_new = B_old
    r_x_new = r_x_old
    r_y_new = r_y_old
    
    return f_new, Ex_new, Ey_new, B_new, r_x_new, r_y_new

@jit
def HE_update(grid_x, grid_vx, grid_vy, f_old, Ex_old, Ey_old, B_old, r_x_old, r_y_old, delta_t):
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
    - r_x_old: Scalar representing r_x at current time.
    - r_y_old: Scalar representing r_y at current time.
    - delta_t: Time step (scalar).
    
    Returns:
    - f_new: Updated distribution function (shape: [N_x, N_vx, N_vy]).
    - Ex_new: Unchanged E_x electric field (shape: [N_x]).
    - Ey_new: Unchanged E_y electric field (shape: [N_x]).
    - B_new: Updated B field (shape: [N_x]).
    - r_x_new: Updated r_x (scalar).
    - r_y_new: Updated r_y (scalar).
    """
    
    # ---------------------------
    # Step 1: Compute rho_old
    # ---------------------------
    
    rho_old = compute_rho(f_old, dvx=(grid_vx[1] - grid_vx[0]), dvy=(grid_vy[1] - grid_vy[0]))
    
    # ---------------------------
    # Step 2: Compute r_x_old and r_y_old from E_x_old and E_y_old
    
    r_x_old_from_E = compute_r(Ex_old, dx=(grid_x[1] - grid_x[0]))
    r_y_old_from_E = compute_r(Ey_old, dx=(grid_x[1] - grid_x[0]))
    
    
    # ---------------------------
    # Step 3: Compute g_x and g_y
    # ---------------------------
    
#     g_x = compute_g(Ex_old, r_x_old)  # Shape: [N_x]
#     g_y = compute_g(Ey_old, r_y_old)  # Shape: [N_x]
    g_x = compute_g(Ex_old, r_x_old_from_E)  # Shape: [N_x]
    g_y = compute_g(Ey_old, r_y_old_from_E)  # Shape: [N_x]
    
    # ---------------------------
    # Step 4: Compute Jx_old and Jy_old
    # ---------------------------
    
    Jx_old = compute_Jx(f_old, grid_vx, grid_vy, dvx=(grid_vx[1] - grid_vx[0]), dvy=(grid_vy[1] - grid_vy[0]))
    Jy_old = compute_Jy(f_old, grid_vx, grid_vy, dvx=(grid_vx[1] - grid_vx[0]), dvy=(grid_vy[1] - grid_vy[0]))
    
    # ---------------------------
    # Step 5: Compute d_x_g_y
    # ---------------------------
    
    d_x_g_y = compute_derivative_fourier(g_y, grid_x)  # Shape: [N_x]
    
    # ---------------------------
    # Step 6: Update r_x_new and r_y_new
    # ---------------------------
    
    r_x_new = update_r_x(delta_t, g_x, rho_old, r_x_old, Jx_old, dx=(grid_x[1] - grid_x[0]))
    r_y_new = update_r_y(delta_t, g_y, d_x_g_y, rho_old, r_y_old, Jy_old, B_old, dx=(grid_x[1] - grid_x[0]))
    
#     print(r_x_new)
#     print(r_y_new)
    
    # ---------------------------
    # Step 7: Compute r_half_x and r_half_y
    # ---------------------------
    r_half_x = 0.5 * (r_x_old + r_x_new)
    r_half_y = 0.5 * (r_y_old + r_y_new)
    
    # ---------------------------
    # Step 8: Update B_new
    # ---------------------------
    B_new = B_old - delta_t * r_half_y * d_x_g_y  # Shape: [N_x]
    
    # ---------------------------
    # Step 9: Update f using bispline interpolation
    # ---------------------------
    
    # Compute shifts
    shift_vx = delta_t * r_half_x * g_x  # Shape: [N_x]
    shift_vy = delta_t * r_half_y * g_y  # Shape: [N_x]
    
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
    # Step 10: E_new remains the same as E_old
    # ---------------------------
    Ex_new = Ex_old
    Ey_new = Ey_old
    
    # ---------------------------
    # Return updated variables
    # ---------------------------
    return f_new, Ex_new, Ey_new, B_new, r_x_new, r_y_new




@jit
def HB_update(grid_x, grid_vx, grid_vy, f_old, Ex_old, Ey_old, B_old, r_x_old, r_y_old, delta_t):
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
    - r_x_old: Scalar representing r_x (unchanged).
    - r_y_old: Scalar representing r_y (unchanged).
    - delta_t: Time step.

    Returns:
    - f_new: Updated distribution function [N_x, N_vx, N_vy].
    - Ex_new: Unchanged E_x electric field [N_x].
    - Ey_new: Updated E_y electric field [N_x].
    - B_new: Unchanged B field [N_x].
    - r_x_new: Unchanged r_x (scalar).
    - r_y_new: Unchanged r_y (scalar).
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
    # r_x_new and r_y_new remain the same as r_x_old and r_y_old
    # ---------------------------
    r_x_new = r_x_old
    r_y_new = r_y_old

    # ---------------------------
    # Update f using rotation and bilinear interpolation
    # ---------------------------

    # Define the skew-symmetric matrix J
    J = jnp.array([[0.0, 1.0],
                   [-1.0, 0.0]])

    # Compute rotation angle theta = B_old * delta_t
    theta =   - B_old * delta_t  # Shape: [N_x]
    #theta =  B_old * delta_t  # Shape: [N_x]

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

    return f_new, Ex_new, Ey_new, B_new, r_x_new, r_y_new

@partial(jit, static_argnums=(7,))  # N_steps is the 8th argument, index 7
def solver_jit(f_iv, B_iv, E_y_iv, grid_x, grid_vx, grid_vy, delta_t, N_steps, k_track):
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

    Returns:
    - Tuple containing:
        - f_end: Final distribution function (3D array [N_x, N_vx, N_vy]).
        - electric_energy_E_x: Array of electric energy from E_x over time (1D array).
        - electric_energy_E_y: Array of electric energy from E_y over time (1D array).
        - magnetic_energy: Array of magnetic energy over time (1D array).
        - kinetic_energy: Array of kinetic energy over time (1D array).
        - r_x_history: Array of r_x values over time (1D array).
        - r_y_history: Array of r_y values over time (1D array).
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
    
    # Solve Poisson equation to obtain initial E_x_iv
    E_x_iv = solve_poisson(rho_iv, grid_x, dx)  # Shape: [N_x]
    
    # Compute initial r_x and r_y from E_x_iv and E_y_iv
    r_x_iv = compute_r(E_x_iv, dx)  # Shape: [N_x]
    r_y_iv = compute_r(E_y_iv, dx)  # Shape: [N_x]
    
    # Initialize state variables
    f_old = f_iv
    Ex_old = E_x_iv
    Ey_old = E_y_iv
    B_old = B_iv
    r_x_old = r_x_iv
    r_y_old = r_y_iv
    
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
        f_old, Ex_old, Ey_old, B_old, r_x_old, r_y_old = carry
        
        # Strang Splitting Steps:
        # 1. Hf for dt/2
        f_new, Ex_new, Ey_new, B_new, r_x_new, r_y_new = Hf_update(
            grid_x, grid_vx, grid_vy,
            f_old, Ex_old, Ey_old,
            B_old, r_x_old, r_y_old,
            delta_t / 2.0
        )
        
        # 2. HE for dt/2
        f_new, Ex_new, Ey_new, B_new, r_x_new, r_y_new = HE_update(
            grid_x, grid_vx, grid_vy,
            f_new, Ex_new, Ey_new,
            B_new, r_x_new, r_y_new,
            delta_t / 2.0
        )
        
        # 3. HB for dt
        f_new, Ex_new, Ey_new, B_new, r_x_new, r_y_new = HB_update(
            grid_x, grid_vx, grid_vy,
            f_new, Ex_new, Ey_new,
            B_new, r_x_new, r_y_new,
            delta_t
        )
        
        # 4. HE for dt/2
        f_new, Ex_new, Ey_new, B_new, r_x_new, r_y_new = HE_update(
            grid_x, grid_vx, grid_vy,
            f_new, Ex_new, Ey_new,
            B_new, r_x_new, r_y_new,
            delta_t / 2.0
        )
        
        # 5. Hf for dt/2
        f_new, Ex_new, Ey_new, B_new, r_x_new, r_y_new = Hf_update(
            grid_x, grid_vx, grid_vy,
            f_new, Ex_new, Ey_new,
            B_new, r_x_new, r_y_new,
            delta_t / 2.0
        )

        f_new = apply_threshold(f_new, f_old)
        Ex_new = apply_threshold(Ex_new, Ex_old)
        Ey_new = apply_threshold(Ey_new, Ey_old)
        B_new  = apply_threshold(B_new, B_old)
        r_x_new = apply_threshold(r_x_new, r_x_old)
        r_y_new = apply_threshold(r_y_new, r_y_old)

        
        # Update state variables for next iteration
        carry = (f_new, Ex_new, Ey_new, B_new, r_x_new, r_y_new)
        
        # Compute energies
        electric_E_x_energy = 0.5 * jnp.sum(Ex_new ** 2) * dx
        electric_E_y_energy = 0.5 * jnp.sum(Ey_new ** 2) * dx
        magnetic_Energy = 0.5 * jnp.sum(B_new ** 2) * dx
        kinetic_Energy = jnp.sum(f_new * (grid_vx[None, :, None] ** 2 + grid_vy[None, None, :] ** 2) / 2) * dvx * dvy * dx
        
        # Compute r_x and r_y
        r_x_current = jnp.mean(r_x_new)  # Assuming r_x is scalar per x
        r_y_current = jnp.mean(r_y_new)  # Assuming r_y is scalar per x
        
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
            r_x_current,
            r_y_current,
            Fourier_mode_B,
            Fourier_mode_E_x,
            Fourier_mode_E_y
        )
        
        return carry, outputs
    
    # Initialize the carry for scan
    initial_carry = (
        f_old,    # f_old
        Ex_old,   # Ex_old
        Ey_old,   # Ey_old
        B_old,    # B_old
        r_x_old,  # r_x_old
        r_y_old   # r_y_old
    )
    
    # Perform the scan over N_steps iterations
    final_carry, outputs = lax.scan(iteration_step, initial_carry, None, length=N_steps)
    
    # Unpack the final carry
    f_end, Ex_final, Ey_final, B_final, r_x_final, r_y_final = final_carry
    
    # Unpack the outputs
    (electric_energy_E_x, 
     electric_energy_E_y, 
     magnetic_energy, 
     kinetic_energy, 
     r_x_history, 
     r_y_history, 
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
        r_x_history,
        r_y_history,
        Fourier_mode_B_history,
        Fourier_mode_E_x_history,
        Fourier_mode_E_y_history,
        Ex_final,
        Ey_final,
        B_final
    )
