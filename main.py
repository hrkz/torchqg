import sys
import math

import torch
import torch.nn as nn

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from qg import _s, _p, QgModel
from sgs import Constant, Hyperviscosity, Leith, Smagorinsky, Gradient

plt.rcParams.update({'mathtext.fontset':'cm'})

# A framework for the evaluation of turbulence closures used in mesoscale ocean large-eddy simulations
# Graham and Ringler (2013)

def t_unit():
  return 1.2e6

def l_unit():
  return (504e4 / math.pi)

Lx = 2*math.pi
Ly = 2*math.pi
Nx = 2048
Ny = 2048

dt = 600 / t_unit() # 600s
mu = 1.57e-8 / l_unit()**(-1) # 1.57e-8m^-1
#nu = 98 / l_unit()**2 / t_unit()**(-1) # 98m^2s^-1 for startup (1024^2)
nu = 24.5 / l_unit()**2 / t_unit()**(-1) # 24.5m^2s^-1 for the simulation (2048^2)

eta = torch.zeros([Ny, Nx], dtype=torch.float64, requires_grad=True)

# High res model
h = QgModel(
  name='\\mathcal{F}',
  Nx=Nx,
  Ny=Ny,
  Lx=Lx,
  Ly=Ly,
  dt=dt,
  B=0.0,   # Planetary vorticity y-gradient
  mu=mu,   # Linear drag
  nu=nu,   # Viscosity coefficient
  nv=1,    # Hyperviscous order (nv=1 is viscosity)
  eta=eta  # Topographic PV
)

# Initial conditions
h.ic_spectral(0.01, [3.0, 5.0])
# Loading fields
#h.load('qg_models/data/geo/q_startup.h5') # (1024^2) startup (190 000 iterations)
#h.load('qg_models/data/geo/q_begin.h5')   # (2048^2) begin from established (39 000 iterations, 10 eddy turnover)
# Set up spectral filter
h.fil_ = h.g_.cutoff

print(h)

# Wind stress forcing
def Fs(i, sol, dt, t, grid):
  phi_x = math.pi * math.sin(1.2e-6 / t_unit()**(-1) * t)
  phi_y = math.pi * math.sin(1.2e-6 * math.pi / t_unit()**(-1) * t / 3)
  y = torch.cos(4 * grid.y + phi_y).view(grid.Ny, 1) - torch.cos(4 * grid.x + phi_x).view(1, grid.Nx)

  yh = _s(y)
  K = torch.sqrt(grid.krsq)
  yh[K < 3.0] = 0
  yh[K > 5.0] = 0

  e0 = 1.75e-18 / t_unit()**(-3)
  ei = 0.5 * grid.int_sq(yh) / (grid.Lx * grid.Ly)
  yh *= torch.sqrt(e0 / ei)
  return yh

# Stochastic forcing
h.f_ = Fs

# Low res model(s)
scale = 8

Nxl = int(Nx / scale)
Nyl = int(Ny / scale)

eta_m = torch.zeros([Nyl, Nxl], dtype=torch.float64, requires_grad=True)

# No model
m1 = QgModel(
  name='\\mathcal{R}',
  Nx=Nxl,
  Ny=Nyl,
  Lx=Lx,
  Ly=Ly,
  dt=dt,
  B=0.0,    # Planetary vorticity y-gradient
  mu=mu,    # Linear drag
  nu=nu,    # Viscosity coefficient
  nv=1,     # Hyperviscous order (nv=1 is viscosity)
  eta=eta_m # Topographic PV
)

# Hyperviscosity model
m2 = QgModel(
  name='\\mathcal{R}^{\\mathrm{Hyperviscosity}}',
  Nx=Nxl,
  Ny=Nyl,
  Lx=Lx,
  Ly=Ly,
  dt=dt,
  B=0.0,    # Planetary vorticity y-gradient
  mu=mu,    # Linear drag
  nu=nu,    # Viscosity coefficient
  nv=1,     # Hyperviscous order (nv=1 is viscosity)
  eta=eta_m # Topographic PV
)

# Leith model
m3 = QgModel(
  name='\\mathcal{R}^{\\mathrm{Leith}}',
  Nx=Nxl,
  Ny=Nyl,
  Lx=Lx,
  Ly=Ly,
  dt=dt,
  B=0.0,    # Planetary vorticity y-gradient
  mu=mu,    # Linear drag
  nu=nu,    # Viscosity coefficient
  nv=1,     # Hyperviscous order (nv=1 is viscosity)
  eta=eta_m # Topographic PV
)

# Smagorinsky model
m4 = QgModel(
  name='\\mathcal{R}^{\\mathrm{Smagorinsky}}',
  Nx=Nxl,
  Ny=Nyl,
  Lx=Lx,
  Ly=Ly,
  dt=dt,
  B=0.0,    # Planetary vorticity y-gradient
  mu=mu,    # Linear drag
  nu=nu,    # Viscosity coefficient
  nv=1,     # Hyperviscous order (nv=1 is viscosity)
  eta=eta_m # Topographic PV
)

# Gradient model
m5 = QgModel(
  name='\\mathcal{R}^{\\mathrm{Gradient}}',
  Nx=Nxl,
  Ny=Nyl,
  Lx=Lx,
  Ly=Ly,
  dt=dt,
  B=0.0,    # Planetary vorticity y-gradient
  mu=mu,    # Linear drag
  nu=nu,    # Viscosity coefficient
  nv=1,     # Hyperviscous order (nv=1 is viscosity)
  eta=eta_m # Topographic PV
)

# Subgrid-scale models
m1.sgs_ = Constant(c=0.0)
m2.sgs_ = Hyperviscosity(v=1e-8)
m3.sgs_ = Leith(c=0.5)
m4.sgs_ = Smagorinsky(c=0.3)
m5.sgs_ = Gradient(scale, c=-0.08)

# Initialize from DNS vorticity field
m1.p_.sol = h.filter(m1.g_, scale, h.p_.sol)
m2.p_.sol = h.filter(m2.g_, scale, h.p_.sol)
m3.p_.sol = h.filter(m3.g_, scale, h.p_.sol)
m4.p_.sol = h.filter(m4.g_, scale, h.p_.sol)
m5.p_.sol = h.filter(m5.g_, scale, h.p_.sol)

# Use forcing from DNS simulation
m1.f_ = Fs
m2.f_ = Fs
m3.f_ = Fs
m4.f_ = Fs
m5.f_ = Fs

ms = [
 m1, 
 m2, 
 m3, 
 m4, 
 m5
]

def visitor_run(m, cur, it):
  if it % 100 == 0:
    print('model:',m.name)
    print('eddy turnover time:', m.eddy_time(L=4))
    print('cfl:', m.cfl())
    print('vrms:', m.vrms())
    print('re:', m.re(L=4))
    print('----')
  return None

iters = 100000
with torch.no_grad():
  for it in tqdm.tqdm(range(iters)):
    for m in ms:
      visitor_run(m, m.p_.cur, it)
      m.p_.step(m)
    visitor_run(h, h.p_.cur, it)
    h.p_.step(sys)
