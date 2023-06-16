import sys

import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
import exactRP

# --------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------
# constants
L = 1.0  # 1-D computational domain size
N_In = 128  # number of computing cells
cfl = 1.0  # Courant factor
nghost = 2  # number of ghost zones
gamma = 5.0 / 3.0  # ratio of specific heats
end_time = 0.25  # simulation time

# derived constants
N = N_In + 2 * nghost  # total number of cells including ghost zones
dx = L / N_In  # spatial resolution

def InitialCondition(x):
    #  Sod shock tube
    if (x < 0.5 * L):
        d = 1.0  # density
        u = 0.0  # velocity x
        v = 0.0  # velocity y
        w = 0.0  # velocity z
        P = 1.0  # pressure
        E = P / (gamma - 1.0) + 0.5 * d * (u ** 2.0 + v ** 2.0 + w ** 2.0)  # energy density
    else:
        d = 0.125
        u = 0.0
        v = 0.0
        w = 0.0
        P = 0.1
        E = P / (gamma - 1.0) + 0.5 * d * (u ** 2.0 + v ** 2.0 + w ** 2.0)

    #  conserved variables [0/1/2/3/4] <--> [density/momentum x/momentum y/momentum z/energy]
    return np.array([d, d * u, d * v, d * w, E])


def BoundaryCondition(U):
    #  outflow
    U[0:nghost] = U[nghost]
    U[N - nghost:N] = U[N - nghost - 1]


def ComputePressure(d, px, py, pz, e):
    P = (gamma - 1.0) * (e - 0.5 * (px ** 2.0 + py ** 2.0 + pz ** 2.0) / d)
    # assert np.all(P > 0), "negative pressure !!"
    return P


def ComputeTimestep(U):
    P = ComputePressure(U[:, 0], U[:, 1], U[:, 2], U[:, 3], U[:, 4])
    a = (gamma * P / U[:, 0]) ** 0.5
    u = np.abs(U[:, 1] / U[:, 0])
    v = np.abs(U[:, 2] / U[:, 0])
    w = np.abs(U[:, 3] / U[:, 0])

    #  maximum information speed in 3D
    max_info_speed = np.amax(u + a)
    dt_cfl = cfl * dx / max_info_speed
    dt_end = end_time - t

    return min(dt_cfl, dt_end)


def ComputeLimitedSlope(L, C, R):
    #  compute the left and right slopes
    slope_L = C - L
    slope_R = R - C

    #  apply the van-Leer limiter
    # slope_LR = slope_L * slope_R
    # slope_limited = np.where(slope_LR > 0.0, 2.0 * slope_LR / (slope_L + slope_R), 0.0)

    #  apply the minmod limiter
    slope_limited = np.where(slope_L * slope_R > 0.0, np.where(np.abs(slope_L) < np.abs(slope_R), slope_L, slope_R), 0.0)

    return slope_limited


def Conserved2Primitive(U):
    W = np.empty(5)

    W[0] = U[0]
    W[1] = U[1] / U[0]
    W[2] = U[2] / U[0]
    W[3] = U[3] / U[0]
    W[4] = ComputePressure(U[0], U[1], U[2], U[3], U[4])

    return W


def Primitive2Conserved(W):
    U = np.empty(5)

    U[0] = W[0]
    U[1] = W[0] * W[1]
    U[2] = W[0] * W[2]
    U[3] = W[0] * W[3]
    U[4] = W[4] / (gamma - 1.0) + 0.5 * W[0] * (W[1] ** 2.0 + W[2] ** 2.0 + W[3] ** 2.0)

    return U


def DataReconstruction_PLM(U):
    #  allocate memory
    W = np.empty((N, 5))
    L = np.empty((N, 5))
    R = np.empty((N, 5))

    #  conserved variables --> primitive variables
    for j in range(N):
        W[j] = Conserved2Primitive(U[j])

    for j in range(1, N - 1):
        #     compute the left and right states of each cell
        slope_limited = ComputeLimitedSlope(W[j - 1], W[j], W[j + 1])

        #     get the face-centered variables
        L[j] = W[j] - 0.5 * slope_limited
        R[j] = W[j] + 0.5 * slope_limited

        #     ensure face-centered variables lie between nearby volume-averaged (~cell-centered) values
        L[j] = np.maximum(L[j], np.minimum(W[j - 1], W[j]))
        L[j] = np.minimum(L[j], np.maximum(W[j - 1], W[j]))
        R[j] = 2.0 * W[j] - L[j]

        R[j] = np.maximum(R[j], np.minimum(W[j + 1], W[j]))
        R[j] = np.minimum(R[j], np.maximum(W[j + 1], W[j]))
        L[j] = 2.0 * W[j] - R[j]

        #     primitive variables --> conserved variables
        L[j] = Primitive2Conserved(L[j])
        R[j] = Primitive2Conserved(R[j])

    return L, R


def Conserved2Flux(U):
    flux = np.empty(5)

    P = ComputePressure(U[0], U[1], U[2], U[3], U[4])
    u = U[1] / U[0]

    flux[0] = U[1]
    flux[1] = u * U[1] + P
    flux[2] = u * U[2]
    flux[3] = u * U[3]
    flux[4] = u * (U[4] + P)

    return flux


def exactFlux(L, R):
    #  compute the enthalpy of the left and right states: H = (E+P)/rho
    P_L = ComputePressure(L[0], L[1], L[2], L[3], L[4])
    P_R = ComputePressure(R[0], R[1], R[2], R[3], R[4])
    H_L = (L[4] + P_L) / L[0]
    H_R = (R[4] + P_R) / R[0]

    #  compute Roe average values
    rhoL_sqrt = L[0] ** 0.5
    rhoR_sqrt = R[0] ** 0.5

    u = (L[1] / rhoL_sqrt + R[1] / rhoR_sqrt) / (rhoL_sqrt + rhoR_sqrt)
    v = (L[2] / rhoL_sqrt + R[2] / rhoR_sqrt) / (rhoL_sqrt + rhoR_sqrt)
    w = (L[3] / rhoL_sqrt + R[3] / rhoR_sqrt) / (rhoL_sqrt + rhoR_sqrt)
    H = (rhoL_sqrt * H_L + rhoR_sqrt * H_R) / (rhoL_sqrt + rhoR_sqrt)
    V2 = u * u + v * v + w * w
    #  check negative pressure
    assert H - 0.5 * V2 > 0.0, "negative pressure!"
    a = ((gamma - 1.0) * (H - 0.5 * V2)) ** 0.5

    #  compute the amplitudes of different characteristic waves
    dU = R - L
    amp = np.empty(5)
    amp[2] = dU[2] - v * dU[0]
    amp[3] = dU[3] - w * dU[0]
    amp[1] = (gamma - 1.0) / a ** 2.0 \
             * (dU[0] * (H - u ** 2.0) + u * dU[1] - dU[4] + v * amp[2] + w * amp[3])
    amp[0] = 0.5 / a * (dU[0] * (u + a) - dU[1] - a * amp[1])
    amp[4] = dU[0] - amp[0] - amp[1]

    #  compute the eigenvalues and right eigenvector matrix
    EigenValue = np.array([u - a, u, u, u, u + a])
    EigenVector_R = np.array([[1.0, u - a, v, w, H - u * a],
                              [1.0, u, v, w, 0.5 * V2],
                              [0.0, 0.0, 1.0, 0.0, v],
                              [0.0, 0.0, 0.0, 1.0, w],
                              [1.0, u + a, v, w, H + u * a]])

    #  compute the fluxes of the left and right states
    flux_L = Conserved2Flux(L)
    flux_R = Conserved2Flux(R)

    #  compute the Roe flux
    amp *= np.abs(EigenValue)
    flux = 0.5 * (flux_L + flux_R) - 0.5 * amp.dot(EigenVector_R)
    return flux

def primitive_to_conservative_flux_derivatives(rho, u, p, p_star, c, Gamma, Gamma1, G1o2):
    if p_star > p:  # Shock wave
        A = 2. / ((Gamma + 1.) * rho)
        B = (Gamma - 1.) / Gamma * p
        sqrt_term = max(0., A / (B + p_star))
        f = (p_star - p) * sqrt_term**0.5
        df = (1. - 0.5 * (p_star - p) / (B + p_star)) * sqrt_term
    else:  # Rarefaction wave
        a = 2. / (Gamma1 * rho)
        b = p * Gamma1 / Gamma
        f = 2. * c / Gamma1 * ((p_star / p)**G1o2 - 1.)
        df = ((p_star / p)**-G1o2) / (rho * c)

    return f, df


def exact_flux_conserved(L, R):
    from math import fabs

    # Define the constant values.
    Gamma = 5.0 / 3.0
    Gamma1 = Gamma - 1.
    G1o2 = Gamma1 / (2. * Gamma)

    # Transform conserved variables to primitive variables.
    rho_l = L[0]
    u_l = L[1] / rho_l
    E_l = L[4]
    p_l = (Gamma - 1.) * (E_l - 0.5 * rho_l * u_l**2)

    rho_r = R[0]
    u_r = R[1] / rho_r
    E_r = R[4]
    p_r = (Gamma - 1.) * (E_r - 0.5 * rho_r * u_r**2)

    # Compute sound speeds of left and right states.
    c_l = (Gamma * p_l / rho_l)**0.5
    c_r = (Gamma * p_r / rho_r)**0.5

    # Iteratively compute pressure in star region.
    # Initial guess for pressure
    p_star = 0.5 * (p_l + p_r)
    p_old = p_star
    delta_u = u_r - u_l

    # Define solver criteria
    MAX_ITER = 100
    TOL_PRES = 1e-8

    # Perform Newton-Raphson iteration to determine pressure in star region
    nrSuccess = False
    for it in range(1, MAX_ITER+1):
        f_l_star, df_l_star = primitive_to_conservative_flux_derivatives(rho_l, u_l, p_l, p_old, c_l, Gamma, Gamma1, G1o2)
        f_r_star, df_r_star = primitive_to_conservative_flux_derivatives(rho_r, u_r, p_r, p_old, c_r, Gamma, Gamma1, G1o2)
        p_star = p_old - (f_l_star + f_r_star + delta_u) / (df_l_star + df_r_star)
        dp = 2.0 * fabs((p_star - p_old) / (p_star + p_old))
        if dp <= TOL_PRES:
            nrSuccess = True
            break
        if p_star < 0.0:
            p_star = TOL_PRES
        p_old = p_star

    if not nrSuccess:
        print('[exact_flux_conserved] Newton-Raphson unable to converge')
    else:
        u_star = 0.5 * (u_l + u_r + f_r_star - f_l_star)

    # Compute fluxes
    flux_l = np.array(
        [rho_l * u_l, rho_l * u_l ** 2 + p_l, u_l * (rho_l * (c_l ** 2 / Gamma1 + 0.5 * u_l ** 2) + p_l / Gamma1)])
    flux_r = np.array(
        [rho_r * u_r, rho_r * u_r ** 2 + p_r, u_r * (rho_r * (c_r ** 2 / Gamma1 + 0.5 * u_r ** 2) + p_r / Gamma1)])

    return flux_r if u_star >= 0.0 else flux_l




# --------------------------------------------------------------------
# main
# --------------------------------------------------------------------
# set initial condition
t = 0.0
x = np.empty(N_In)
U = np.empty((N, 5))
for j in range(N_In):
    x[j] = (j + 0.5) * dx  # cell-centered coordinates
    U[j + nghost] = InitialCondition(x[j])



while (t < 0.001):
    # set boundary condition
    BoundaryCondition(U)
    dt = ComputeTimestep(U)
    # print("t = %13.7e --> %13.7e, dt = %13.7e" % (t, t + dt, dt))
    L, R = DataReconstruction_PLM(U)

    for j in range(1, N - 1):
        flux_L = Conserved2Flux(L[j])
        flux_R = Conserved2Flux(R[j])
        dflux = 0.5 * dt / dx * (flux_R - flux_L)
        L[j] -= dflux
        R[j] -= dflux

    flux = np.empty((N, 5))
    for j in range(nghost, N - nghost + 1):
        # flux[j] = exactFlux(R[j - 1], L[j])
        flux[j] = exact_flux_conserved(R[j - 1], L[j])
        # print(flux[j])
    U[nghost:N - nghost] -= dt / dx * (flux[nghost + 1:N - nghost + 1] - flux[nghost:N - nghost])
    t = t + dt



d = U[nghost:N - nghost, 0]
u = U[nghost:N - nghost, 1] / U[nghost:N - nghost, 0]
P = ComputePressure(U[nghost:N - nghost, 0], U[nghost:N - nghost, 1], U[nghost:N - nghost, 2], U[nghost:N - nghost, 3],
                    U[nghost:N - nghost, 4])
data = np.column_stack((x, d, u, P))
np.savetxt('approx_result_1.txt', data, header='x d u P', comments='')


# Create a new figure
plt.figure()

# Plot density
plt.subplot(3, 1, 1)  # 3 rows, 1 column, first subplot
# plt.plot(x, d)
plt.scatter(x, d, s=1)
plt.ylabel('Density')

# Plot pressure
plt.subplot(3, 1, 2)  # 3 rows, 1 column, second subplot
# plt.plot(x, P)
plt.scatter(x, P, s=1)
plt.ylabel('Pressure')

# Plot velocity
plt.subplot(3, 1, 3)  # 3 rows, 1 column, third subplot
# plt.plot(x, u)
plt.scatter(x, u, s=1)
plt.ylabel('Velocity')
plt.xlabel('Position')

# Display the figure
plt.tight_layout()  # adjust spacing between subplots
plt.savefig('result.png')
plt.show()