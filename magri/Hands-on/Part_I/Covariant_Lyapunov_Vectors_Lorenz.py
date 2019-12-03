"""
@author: Luca Magri
"""
#!/usr/bin/env python
# coding: utf-8

# ## CLV analysis of the Lorenz system

from __future__ import print_function

from numpy import *
from scipy.linalg import solve_triangular
from scipy.integrate import odeint

from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D

Ndim = 3
sigma, beta, rho = 10.0, 8.0/3, 28.0
q0 = array([-8.67139571762, 4.98065219709, 25.0])


class Solution:
    def __init__(self, d):
        for i in d.keys():
            setattr(self, i, d[i])


def gsQR(M):
    ''' QR decomposition based on Gram-Schmidt '''
    Q = zeros(M.shape)
    R = zeros(M.shape)

    for j in range(M.shape[0]):
        v = M[:, j].copy()
        for i in range(j):
            R[i, j] = dot(Q[:, i], M[:, j])
            v -= R[i, j]*Q[:, i]
        R[j, j] = linalg.norm(v)
        Q[:, j] = v/R[j, j]

    return Q, R


def F(q, t):
    ''' dq/dt and dM/dt '''

    x, y, z = q[:Ndim]
    M = q[Ndim:(Ndim+1)*Ndim].reshape((Ndim, Ndim))

    J = array([[-sigma, sigma, 0.0], [rho-z, -1.0, -x], [y, x, -beta]])

    dqdt = [sigma*(y-x), x*(rho-z) - y, x*y - beta*z]
    dMdt = dot(J, M)

    return concatenate([dqdt, dMdt.flatten()])


def normalize(M):
    ''' Normalizes columns of M individually '''
    nM = zeros(M.shape)  # normalized matrix
    nV = zeros(M.shape)  # norms of columns
    for i in range(M.shape[1]):
        nV[i, i] = linalg.norm(M[:, i])
        nM[:, i] = M[:, i] / nV[i, i]
    return nM, nV


def average(y, x):
    ''' Return average of y(x) '''
    return trapz(y, x)/(x[-1] - x[0])


def lorenz(N=10000, su=200, sd=200, dt=0.01, q0=q0):
    s = su        # index of spinup time
    e = N+1 - sd  # index of spindown time
    T = arange(N+1) * dt  # times of sample points

    # time series of state vector
    # index 0 is time step
    # index 1:
    #    0,1,2 = x,y,z of Lorenz system
    #    3->12 = matrix M of dM/dt = J*M
    q = zeros((N+1, Ndim+Ndim*Ndim))

    # time series of M (result of odeint)
    M = zeros((N+1, Ndim, Ndim))

    # time series of Q, R from QR decomposition
    Q = zeros((N+1, Ndim, Ndim))
    R = zeros((N+1, Ndim, Ndim))

    # Lyapunov exponents history (to check for convergence)
    hl = zeros((N+1, Ndim))

    # initialization of state vector
    q[0, :Ndim] = q0
    q[0, Ndim:(Ndim+1)*Ndim] = eye(Ndim).flatten()

    M[0] = q[0, Ndim:(Ndim+1)*Ndim].reshape((Ndim, Ndim))
    Q[0], R[0] = gsQR(M[0])

    N_output = N//10
    for i in 1+arange(N):
        if i % N_output == 0:
            print("=> Iteration #%05d Time %.3f -> %.3f" % (i, T[i-1], T[i]))

        # evolve from T_i-1 -> T_i
        q[i] = odeint(F, q[i-1], [T[i-1], T[i]])[-1, :]

        # extract and reshape M matrices from simulation state vector q
        M[i] = q[i, Ndim:(Ndim+1)*Ndim].copy().reshape((Ndim, Ndim))

        # run QR decomposition
        Q[i], R[i] = gsQR(M[i])

        # replace M for Q in state vector
        q[i, Ndim:(Ndim+1)*Ndim] = Q[i].flatten()

        # sum diagonal of R for LE computation if time > spinup
        if s < i and i <= e:
            hl[i] = hl[i-1] + log(abs(diag(R[i])))

    # average l in time
    for i in 1+arange(s, e):
        hl[i] /= (i-s)*dt

    # lyapunov exponents
    l = hl[e].copy()

    # coordinates of CLVs in local GS vector basis
    C = zeros((N+1, Ndim, Ndim))
    D = zeros((N+1, Ndim, Ndim))  # diagonal matrix
    V = zeros((N+1, Ndim, Ndim))  # coordinates of CLVs in physical space

    # initialise components to I
    C[N] = eye(Ndim)
    D[N] = eye(Ndim)
    V[N] = dot(Q[N], C[N])

    # R_i C_i = C_i+1
    for i in reversed(range(N)):
        C[i], D[i] = normalize(solve_triangular(R[i], C[i+1]))
        V[i] = dot(Q[i], C[i])

    d = {'N': N, 'T': T, 's': s, 'e': e, 'q': q, 'M': M, 'Q': Q, 'R': R, 'l': l,
         'hl': hl, 'C': C, 'V': V, 'D': D, 's': s, 'e': e, 'dt': dt}

    return Solution(d)


if __name__ == '__main__':
    dt = 0.005
    N = int(60/dt)
    su = int(20/dt)
    sd = int(20/dt)
    q0 = random.random((Ndim))

    sol = lorenz(N=N, su=su, sd=sd, dt=dt, q0=q0)
    print("Lyapunov exponents:", sol.l)

    n = 2
    m = 2

    subplots(n, m, figsize=(10.0, 10.0))
    subplots_adjust(hspace=0.25, wspace=0.25)

    # plot x, y, z vs time
    subplot(n, m, 1)
    title('q')
    plot(sol.T, sol.q[:, :Ndim])
    ax = gca()
    grid()

    # plot convergence of LEs
    subplot(n, m, 2)
    title('LEs')
    plot(sol.T[sol.s:sol.e], sol.hl[sol.s:sol.e])
    grid()

    # 3d plot of trajectory
    ax = subplot(n, m, 3, projection='3d')
    title('trajectory')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plot(sol.q[:, 0], sol.q[:, 1], sol.q[:, 2], color='b')
    s = sol.s
    e = sol.e

    # 3d plot of part of trajectory with CLVs superimposed
    ax = subplot(n, m, 4, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    t1, t2 = sol.N//2, sol.N//2+100
    plot(sol.q[t1:t2, 0], sol.q[t1:t2, 1], sol.q[t1:t2, 2], color='r')
    for i in range(3):
        quiver(sol.q[t1:t2, 0], sol.q[t1:t2, 1], sol.q[t1:t2, 2], sol.V[t1:t2, 0, i],
               sol.V[t1:t2, 1, i], sol.V[t1:t2, 2, i], color=('tab:'+['blue', 'orange', 'green'][i]))

    # plot divergence of perturbed trajectories along the CLVs
    figure(2)
    eps = 1e-3
    N = int(10/dt)
    for i in range(Ndim):
        q0_pert = sol.q[sol.s, :Ndim] + eps*sol.V[sol.s, :, i]
        pert_sol = lorenz(N=N, q0=q0_pert, dt=dt)
        diff = sol.q[sol.s:sol.s+N+1, :Ndim] - pert_sol.q[:, :Ndim]
        dist = sqrt((diff**2).sum(axis=1))
        semilogy(pert_sol.T, dist/eps, label='$\\lambda %c 0$' % '>=<'[i])
        legend()

    show()
