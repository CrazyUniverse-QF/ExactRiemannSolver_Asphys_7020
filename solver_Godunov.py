import sys

import numpy as np
import matplotlib.pyplot as plt
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
end_time = 0.1  # simulation time

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
    slope_limited = np.where(slope_L * slope_R > 0.0, np.where(np.abs(slope_L) < np.abs(slope_R), slope_L, slope_R), 0.0)

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


def exactFlux(L, R, s):
    L = Conserved2Primitive(L)
    # print(L)
    R = Conserved2Primitive(R)
    # trim L to stateL by removing index 2 and 3, remaining 0, 1, 4
    stateL = [L[0], L[1], L[4]]
    stateR = [R[0], R[1], R[4]]

    # stateL = [1.0, 0.75, 1.0]
    # stateR = [0.125, 0.0, 0.1]
    # gamma = 1.4

    rp = exactRP.exactRP(gamma, stateL, stateR)
    success = rp.solve()
    if (not success):
        sys.stdout.write('[FAILURE] Unable to solve problem {:s}')

    pstar = rp.presS
    # print(pstar)
    vstar = rp.velxS
    # print(vstar)

    dOut, pOut, uOut = rp.samplePt(s)

    primitive = np.empty(5)
    primitive[0] = dOut
    primitive[1] = uOut
    primitive[2] = 0
    primitive[3] = 0
    primitive[4] = pOut

    # flux = np.sqrt(((gamma + 1.0) * pstar * stateL[2] + (gamma - 1.0) * stateL[2] * stateL[0]) / 2)
    conserved = Primitive2Conserved(primitive)
    flux = Conserved2Flux(conserved)
    # print(flux[0])
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

while (t < 0.02):
    # set boundary condition
    BoundaryCondition(U)
    dt = ComputeTimestep(U)
    print("t = %13.7e --> %13.7e, dt = %13.7e" % (t, t + dt, dt))
    L, R = DataReconstruction_PLM(U)
    for j in range(1, N - 1):
        flux_L = Conserved2Flux(L[j])
        flux_R = Conserved2Flux(R[j])
        dflux = 0.5 * dt / dx * (flux_R - flux_L)
        L[j] -= dflux
        R[j] -= dflux

    flux = np.empty((N, 5))
    for j in range(nghost, N - nghost + 1):
        flux[j] = exactFlux(R[j - 1], L[j], j-1)
        # print(flux[j])

    U[nghost:N - nghost] -= dt / dx * (flux[nghost + 1:N - nghost + 1] - flux[nghost:N - nghost])
    t = t + dt


d = U[nghost:N - nghost, 0]
u = U[nghost:N - nghost, 1] / U[nghost:N - nghost, 0]
P = ComputePressure(U[nghost:N - nghost, 0], U[nghost:N - nghost, 1], U[nghost:N - nghost, 2], U[nghost:N - nghost, 3],
                    U[nghost:N - nghost, 4])

# plot in scatter plot
plt.scatter(x, d, c='b', marker='o')
plt.scatter(x, u, c='r', marker='o')
plt.scatter(x, P, c='g', marker='o')
# plt.plot(x, d, 'b-')
# plt.plot(x, u, 'r-')
# plt.plot(x, P, 'g-')
plt.show()