import math
import sys

import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, float32, jit
from numba import njit, prange
import matplotlib.animation as animation
import exactRP

# --------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------
# constants
L = 1.0  # 1-D computational domain size
N_In = 1024  # number of computing cells
cfl = 1.0  # Courant factor
nghost = 2  # number of ghost zones
gamma = 5.0 / 3.0  # ratio of specific heats
end_time = 0.4  # simulation time

# derived constants
N = N_In + 2 * nghost  # total number of cells including ghost zones
dx = L / N_In  # spatial resolution


# -------------------------------------------------------------------
# define initial condition
# -------------------------------------------------------------------
def InitialCondition(x):
    #  Sod shock tube
    if (x < 0.5 * L):
        d = 1250  # density
        u = 0.0  # velocity x
        v = 0.0  # velocity y
        w = 0.0  # velocity z
        P = 500  # pressure
        E = P / (gamma - 1.0) + 0.5 * d * (u ** 2.0 + v ** 2.0 + w ** 2.0)  # energy density
    else:
        d = 125
        u = 0.0
        v = 0.0
        w = 0.0
        P = 5
        E = P / (gamma - 1.0) + 0.5 * d * (u ** 2.0 + v ** 2.0 + w ** 2.0)

    #  conserved variables [0/1/2/3/4] <--> [density/momentum x/momentum y/momentum z/energy]
    return np.array([d, d * u, d * v, d * w, E])


# -------------------------------------------------------------------
# define boundary condition by setting ghost zones
# -------------------------------------------------------------------
def BoundaryCondition(U):
    #  outflow
    U[0:nghost] = U[nghost]
    U[N - nghost:N] = U[N - nghost - 1]


# -------------------------------------------------------------------
# compute pressure
# -------------------------------------------------------------------
# @jit
def ComputePressure(d, px, py, pz, e):
    P = (gamma - 1.0) * (e - 0.5 * (px ** 2.0 + py ** 2.0 + pz ** 2.0) / d)
    # assert np.all(P > 0), "negative pressure !!"
    return P


# -------------------------------------------------------------------
# compute time-step by the CFL condition
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# compute limited slope
# -------------------------------------------------------------------
def ComputeLimitedSlope(L, C, R):
    #  compute the left and right slopes
    slope_L = C - L
    slope_R = R - C

    #  apply the van-Leer limiter
    # slope_LR = slope_L * slope_R
    # slope_limited = np.where(slope_LR > 0.0, 2.0 * slope_LR / (slope_L + slope_R), 0.0)

    #  apply the minmod limiter
    slope_limited = np.where(slope_L * slope_R > 0.0, np.where(np.abs(slope_L) < np.abs(slope_R), slope_L, slope_R),
                             0.0)

    return slope_limited


# -------------------------------------------------------------------
# convert conserved variables to primitive variables
# -------------------------------------------------------------------
def Conserved2Primitive(U):
    W = np.empty(5)

    W[0] = U[0]
    W[1] = U[1] / U[0]
    W[2] = U[2] / U[0]
    W[3] = U[3] / U[0]
    W[4] = ComputePressure(U[0], U[1], U[2], U[3], U[4])

    return W


# -------------------------------------------------------------------
# convert primitive variables to conserved variables
# -------------------------------------------------------------------
def Primitive2Conserved(W):
    U = np.empty(5)

    U[0] = W[0]
    U[1] = W[0] * W[1]
    U[2] = W[0] * W[2]
    U[3] = W[0] * W[3]
    U[4] = W[4] / (gamma - 1.0) + 0.5 * W[0] * (W[1] ** 2.0 + W[2] ** 2.0 + W[3] ** 2.0)

    return U


# -------------------------------------------------------------------
# piecewise-linear data reconstruction
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# convert conserved variables to fluxes
# -------------------------------------------------------------------
# @jit
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


@cuda.jit(device=True)
def Conserved2Flux_d(U, flux):
    # P = ComputePressure_d(U[0], U[1], U[2], U[3], U[4])
    P = (gamma - 1.0) * (U[4] - 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / U[0])
    u = U[1] / U[0]

    flux[0] = U[1]
    flux[1] = u * U[1] + P
    flux[2] = u * U[2]
    flux[3] = u * U[3]
    flux[4] = u * (U[4] + P)


@cuda.jit(device=True)
def exactFlux_d(L, R, flux_L, flux_R, flux, amp, EigenValue, EigenVector_R):
    P_L = (gamma - 1.0) * (L[4] - 0.5 * (L[1] ** 2 + L[2] ** 2 + L[3] ** 2) / L[0])
    P_R = (gamma - 1.0) * (R[4] - 0.5 * (R[1] ** 2 + R[2] ** 2 + R[3] ** 2) / R[0])
    H_L = (L[4] + P_L) / L[0]
    H_R = (R[4] + P_R) / R[0]

    rhoL_sqrt = math.sqrt(L[0])
    rhoR_sqrt = math.sqrt(R[0])

    u = (L[1] / rhoL_sqrt + R[1] / rhoR_sqrt) / (rhoL_sqrt + rhoR_sqrt)
    v = (L[2] / rhoL_sqrt + R[2] / rhoR_sqrt) / (rhoL_sqrt + rhoR_sqrt)
    w = (L[3] / rhoL_sqrt + R[3] / rhoR_sqrt) / (rhoL_sqrt + rhoR_sqrt)
    H = (rhoL_sqrt * H_L + rhoR_sqrt * H_R) / (rhoL_sqrt + rhoR_sqrt)
    V2 = u * u + v * v + w * w

    a = math.sqrt((gamma - 1.0) * (H - 0.5 * V2))
    # dU = R - L
    dU_0 = R[0] - L[0]
    dU_1 = R[1] - L[1]
    dU_2 = R[2] - L[2]
    dU_3 = R[3] - L[3]
    dU_4 = R[4] - L[4]

    # amp[2] = dU[2] - v * dU[0]
    # amp[3] = dU[3] - w * dU[0]
    # amp[1] = (gamma - 1.0) / a ** 2.0 \
    #          * (dU[0] * (H - u ** 2.0) + u * dU[1] - dU[4] + v * amp[2] + w * amp[3])
    # amp[0] = 0.5 / a * (dU[0] * (u + a) - dU[1] - a * amp[1])
    # amp[4] = dU[0] - amp[0] - amp[1]

    amp[2] = dU_2 - v * dU_0
    amp[3] = dU_3 - w * dU_0
    amp[1] = (gamma - 1.0) / a ** 2.0 \
             * (dU_0 * (H - u ** 2.0) + u * dU_1 - dU_4 + v * amp[2] + w * amp[3])
    amp[0] = 0.5 / a * (dU_0 * (u + a) - dU_1 - a * amp[1])
    amp[4] = dU_0 - amp[0] - amp[1]

    EigenValue[0] = u - a
    EigenValue[1] = u
    EigenValue[2] = u
    EigenValue[3] = u
    EigenValue[4] = u + a

    EigenVector_R[0, 0] = 1.0
    EigenVector_R[0, 1] = u - a
    EigenVector_R[0, 2] = v
    EigenVector_R[0, 3] = w
    EigenVector_R[0, 4] = H - u * a

    EigenVector_R[1, 0] = 1.0
    EigenVector_R[1, 1] = u
    EigenVector_R[1, 2] = v
    EigenVector_R[1, 3] = w
    EigenVector_R[1, 4] = 0.5 * V2

    EigenVector_R[2, 0] = 0.0
    EigenVector_R[2, 1] = 0.0
    EigenVector_R[2, 2] = 1.0
    EigenVector_R[2, 3] = 0.0
    EigenVector_R[2, 4] = v

    EigenVector_R[3, 0] = 0.0
    EigenVector_R[3, 1] = 0.0
    EigenVector_R[3, 2] = 0.0
    EigenVector_R[3, 3] = 1.0
    EigenVector_R[3, 4] = w

    EigenVector_R[4, 0] = 1.0
    EigenVector_R[4, 1] = u + a
    EigenVector_R[4, 2] = v
    EigenVector_R[4, 3] = w
    EigenVector_R[4, 4] = H + u * a

    Conserved2Flux_d(L, flux_L)
    Conserved2Flux_d(R, flux_R)

    # amp *= np.abs( EigenValue )
    for i in range(5):
        amp[i] *= math.fabs(EigenValue[i])

    for i in range(5):
        flux[i] = 0.5 * (flux_L[i] + flux_R[i])
        for j in range(5):
            flux[i] -= 0.5 * amp[j] * EigenVector_R[j, i]


@cuda.jit
def calculate_flux(nghost, N, L, R, flux_L, flux_R, flux, amp, EigenValue, EigenVector_R):
    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if j < N:
        if nghost <= j < N - nghost + 1:
            exactFlux_d(R[j - 1], L[j], flux_L[j], flux_R[j], flux[j], amp[j], EigenValue[j], EigenVector_R[j])


# @jit
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


# @njit(parallel=True)
def compute_flux(N, L, R, nghost):
    flux = np.empty((N, 5))
    for j in prange(nghost, N - nghost + 1):
        flux[j] = exactFlux(R[j - 1], L[j])

    return flux


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

while (t < end_time):
    # set boundary condition
    BoundaryCondition(U)
    dt = ComputeTimestep(U)
    # print("t = %13.7e --> %13.7e, dt = %13.7e" % (t, t + dt, dt))
    L, R = DataReconstruction_PLM(U)

    # update the face-centered variables by 0.5*dt
    for j in range(1, N - 1):
        flux_L = Conserved2Flux(L[j])
        flux_R = Conserved2Flux(R[j])
        dflux = 0.5 * dt / dx * (flux_R - flux_L)
        L[j] -= dflux
        R[j] -= dflux

    # Decide the number of CUDA threads
    threads_per_block = 128
    blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block
    Lflux = np.empty((N, 5))
    Rflux = np.empty((N, 5))
    flux = np.empty((N, 5))
    amps = np.empty((N, 5))
    EigenValue = np.empty((N, 5))
    EigenVector_R = np.empty((N, 5, 5))
    calculate_flux[blocks_per_grid, threads_per_block](nghost, N, L, R, Lflux, Rflux, flux, amps, EigenValue,
                                                       EigenVector_R)

    U[nghost:N - nghost] -= dt / dx * (flux[nghost + 1:N - nghost + 1] - flux[nghost:N - nghost])

    t = t + dt

d = U[nghost:N - nghost, 0]
u = U[nghost:N - nghost, 1] / U[nghost:N - nghost, 0]
P = ComputePressure(U[nghost:N - nghost, 0], U[nghost:N - nghost, 1], U[nghost:N - nghost, 2], U[nghost:N - nghost, 3],
                    U[nghost:N - nghost, 4])

# plot in scatter plot
# plt.scatter(x, d, c='b', marker='o')
# plt.scatter(x, u, c='r', marker='o')
# plt.scatter(x, P, c='g', marker='o')
plt.plot(x, d, 'b-')
plt.plot(x, u, 'r-')
plt.plot(x, P, 'g-')
plt.show()
