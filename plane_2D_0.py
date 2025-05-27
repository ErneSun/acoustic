import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt

# Parameters
R = 0.01            # Pipe radius [m]
dr = 0.00001         # Spatial resolution [m]
inner_r = R - 0.002 # Detonation band inner radius
outer_r = R         # Detonation band outer radius
wave_pressure = 10  # Peak pressure [Pa]
sigma = 0.001       # Gaussian tail width [m]
omega = 1400 / R    # Angular speed [rad/s]
t = 1e-4             # Initial time

# Meshgrid
x = np.arange(-R, R + dr, dr)
y = np.arange(-R, R + dr, dr)
X, Y = np.meshgrid(x, y)

# Source term function
def source_term(X, Y, t):
    theta = np.arctan2(Y, X) % (2*np.pi)
    r = np.sqrt(X**2 + Y**2)
    theta0 = (omega * t) % (2*np.pi)
    dtheta = (theta - theta0 + np.pi) % (2*np.pi) - np.pi
    mask_ring = (r >= inner_r) & (r <= outer_r) & (dtheta >= 0) & (dtheta*R < 3*sigma)
    S = np.zeros_like(X)+1
    S[mask_ring] = wave_pressure * np.exp( (dtheta[mask_ring] * R)**2 / (2 * sigma**2)) #现在这个maskring应该束缚在环带中，而且没有加入时间衰减项
    return S

# Evaluate and plot
S0 = source_term(X, Y, t)

plt.figure(figsize=(6,5))
plt.contourf(X*1000, Y*1000, S0, levels=100, cmap='inferno')
plt.colorbar(label='Source Pressure (Pa)')
plt.title("Initial Detonation Source Distribution (t = 0)")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig("source_term_initial.png", dpi=300)
plt.show()