import os
import numpy as np
import matplotlib.pyplot as plt
from math import *

# Simulation length
t, t_final = 0.0, 0.8  # [-]

# Physical params
n = 5  # [-]
gamma = 1.4  # [-]
P0 = 101.325e3  # Pa
rho0 = 1.225  # kg / m (1D)
c0 = sqrt(gamma * P0 / rho0)  # m/s
K0 = P0 / (rho0**gamma)  # Pa * (m/kg)^1.4
m0 = 2.7e-3  # kg, mass of projectile (Ping Pong ball)
T0 = (m0 * c0) / (P0)
L0 = c0 * T0  # m


# Computational params
NCELLS = 1500
dx0 = L0 / NCELLS  # m, Initial cell spacing
CFL = 0.1  # Coefficient to maintain numerical stability
FREQ = 1000  # How often to print out system

# Key quantities
x = -(NCELLS - 1 - np.arange(NCELLS)) * dx0 / L0  # [-]
v = np.zeros(NCELLS)
P = np.zeros(NCELLS)
rho = np.zeros(NCELLS)
c = np.zeros(NCELLS)
m = rho0 * dx0 / m0  # [-]
K = K0 / (P0 * ((L0 / m0) ** 1.4))  # [-]


# Exact solutions for projectile and gas
def xproj(time):

    return n * ((1 + time) - pow(1 + (time * (n + 1) / n), (n / (n + 1))))


def vproj(time):

    return n * (1 - pow(1 + (time * (n + 1) / n), -(1 / (n + 1))))


def vgas(pos, time):

    if(time + pos > 0):

        return (time + pos) / (1 + ((6/5) * time ))
        
    else:
    
        return 0

# Just to print out projected dt
dx = np.diff(x)
rho = m / dx
P = K * rho**gamma
c = np.sqrt(gamma * P / rho)
dt = min(CFL * np.min(dx / c), t_final - t)

# Make directory for given cell size
print("NCELLS = {}".format(NCELLS))
dir = "NCELLS_{}".format(NCELLS)
os.mkdir(dir)


print("Starging dt: {}".format(dt))
print("Approximate number of time steps: {}\n".format(int(t_final / dt)))

while t < t_final:

    # Recalculate cell sizes and resulting thermodynamic quantities
    dx = np.diff(x)  # Cell widths
    assert np.all(dx > 0), f"Nodes crossed at t={t}"  # Sanity check
    rho = m / dx
    P = K * rho**gamma
    c = np.sqrt(gamma * P / rho)

    # Adjust time step if necessary so information from cells only travels to immediate neighbors
    dt = min(CFL * np.min(dx / c), t_final - t)

    # Accelerate interior nodes (left node is fixed at rest, right moves at vproj)
    v[1:-1] -= dt * (P[1:] - P[:-1]) / m

    # Move interior nodes with new velocities
    x[1:-1] += dt * v[1:-1]

    # Update projectile
    t += dt
    x[-1], v[-1] = xproj(t), vproj(t)

    # Compare computational and analytic results
    step = int(t / dt)

    if step % FREQ == 0:

        print("step = {}".format(step))

        # Print out v vs x
        file = "{}/data_{}_{:.2e}".format(dir, step, t)

        with open(file, "w") as outstream:

            for j in range(NCELLS):
                outstream.write("{} {} {}\n".format(x[j], v[j], vgas(x[j], t)))
