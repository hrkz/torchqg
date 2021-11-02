import sys
import math

import torch
import torch.nn as nn

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from qg import _s, _p, QgModel
from sgs import MLdiv, Constant, Leith, Smagorinsky, Gradient

import workflow

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
#Nx = 3072
#Ny = 3072

dt = 600 / t_unit() # 600s
mu = 1.25e-8 / l_unit()**(-1) # 1.25e-8m^-1
#nu = 98 / l_unit()**2 / t_unit()**(-1) # 98m^2s^-1 for startup (1024^2)
nu = 24.5 / l_unit()**2 / t_unit()**(-1) # 24.5m^2s^-1 for the simulation (2048^2)
#nu = 12.25 / l_unit()**2 / t_unit()**(-1) # 12.25m^2s^-1 for the generalization (3072^2)

t0 = 0.0

eta = torch.zeros([Ny, Nx], dtype=torch.float64, requires_grad=True)

# High res model
h = QgModel(
  name='\\mathcal{F}',
  Nx=Nx,
  Ny=Ny,
  Lx=Lx,
  Ly=Ly,
  dt=dt,
  t0=t0,
  B=0.0,   # Planetary vorticity y-gradient
  mu=mu,   # Linear drag
  nu=nu,   # Viscosity coefficient
  nv=1,    # Hyperviscous order (nv=1 is viscosity)
  eta=eta  # Topographic PV
)

# Initial conditions
h.ic_spectral(0.01, [3.0, 5.0])
# Loading fields
#h.load('qg_models/data/geo/q_start_00.h5')  # (1024^2) startup (190 000 iterations)
#h.load('qg_models/data/geo/q_end_00.h5')    # (2048^2) begin from established (39 000 iterations, 10 eddy turnover)
# Set up spectral filter
h.fil_ = h.g_.cutoff

print(h)

# Low res model(s)
scale = 16

Nxl = int(Nx / scale)
Nyl = int(Ny / scale)

# Wind stress forcing
def Fs(i, sol, dt, t, grid):
  phi_x = math.pi * math.sin(1.2e-6 / t_unit()**(-1) * t)
  phi_y = math.pi * math.sin(1.2e-6 * math.pi / t_unit()**(-1) * t / 3)
  y = torch.cos(4 * grid.y + phi_y).view(grid.Ny, 1) - torch.cos(4 * grid.x + phi_x).view(1, grid.Nx)

  yh = _s(y)
  K = torch.sqrt(grid.krsq)
  yh[K < 3.0] = 0
  yh[K > 5.0] = 0
  yh[0, 0] = 0

  e0 = 1.75e-18 / t_unit()**(-3)
  ei = 0.5 * grid.int_sq(yh) / (grid.Lx * grid.Ly)
  yh *= torch.sqrt(e0 / ei)
  return yh

eta_m = torch.zeros([Nyl, Nxl], dtype=torch.float64, requires_grad=True)

# No model
m1 = QgModel(
  name='',
  Nx=Nxl,
  Ny=Nyl,
  Lx=Lx,
  Ly=Ly,
  dt=dt,
  t0=t0,
  B=0.0,     # Planetary vorticity y-gradient
  mu=mu,     # Linear drag
  nu=nu,     # Viscosity coefficient
  nv=1,      # Hyperviscous order (nv=1 is viscosity)
  eta=eta_m, # Topographic PV
  sgs=Constant(c=0.0)
)

# Gradient model
m2 = QgModel(
  name='_{\\mathrm{Gradient}}',
  Nx=Nxl,
  Ny=Nyl,
  Lx=Lx,
  Ly=Ly,
  dt=dt,
  t0=t0,
  B=0.0,     # Planetary vorticity y-gradient
  mu=mu,     # Linear drag
  nu=nu,     # Viscosity coefficient
  nv=1,      # Hyperviscous order (nv=1 is viscosity)
  eta=eta_m, # Topographic PV
  sgs=Gradient(scale, c=0.08)
)

# Leith model
m3 = QgModel(
  name='_{\\mathrm{DynLeith}}',
  Nx=Nxl,
  Ny=Nyl,
  Lx=Lx,
  Ly=Ly,
  dt=dt,
  t0=t0,
  B=0.0,     # Planetary vorticity y-gradient
  mu=mu,     # Linear drag
  nu=nu,     # Viscosity coefficient
  nv=1,      # Hyperviscous order (nv=1 is viscosity)
  eta=eta_m, # Topographic PV
  sgs=Leith(c=None)
)

# Smagorinsky model
m4 = QgModel(
  name='_{\\mathrm{DynSmagorinsky}}',
  Nx=Nxl,
  Ny=Nyl,
  Lx=Lx,
  Ly=Ly,
  dt=dt,
  t0=t0,
  B=0.0,     # Planetary vorticity y-gradient
  mu=mu,     # Linear drag
  nu=nu,     # Viscosity coefficient
  nv=1,      # Hyperviscous order (nv=1 is viscosity)
  eta=eta_m, # Topographic PV
  sgs=Smagorinsky(c=None)
)

# CNN div model
m5 = QgModel(
  name='_{\\mathrm{CNN}}',
  Nx=Nxl,
  Ny=Nyl,
  Lx=Lx,
  Ly=Ly,
  dt=dt,
  t0=t0,
  B=0.0,     # Planetary vorticity y-gradient
  mu=mu,     # Linear drag
  nu=nu,     # Viscosity coefficient
  nv=1,      # Hyperviscous order (nv=1 is viscosity)
  eta=eta_m, # Topographic PV
  #sgs=MLdiv(model=torch.load('?', map_location=h.g_.device))
)

# Initialize from DNS vorticity field
m1.p_.sol = h.filter(m1.g_, scale, h.p_.sol)
m2.p_.sol = h.filter(m2.g_, scale, h.p_.sol)
m3.p_.sol = h.filter(m3.g_, scale, h.p_.sol)
m4.p_.sol = h.filter(m4.g_, scale, h.p_.sol)
m5.p_.sol = h.filter(m5.g_, scale, h.p_.sol)

# Stochastic forcing
h.f_ = Fs

# Use forcing from DNS simulation
m1.f_ = Fs
m2.f_ = Fs
m3.f_ = Fs
m4.f_ = Fs
m5.f_ = Fs

workflow.workflow(
  name='geo',
  iters=3000,   # Model iterations
  steps=100,    # Discrete steps
  scale=scale,  # Kernel scale
  diags=[       # Diagnostics
    workflow.diag_show,
    workflow.diag_sgs_metrics,
    workflow.diag_spatial_stats,
    workflow.diag_spatial_metrics,
    workflow.diag_temporal_stats,
    workflow.diag_integrals,
    workflow.diag_spectra,
    workflow.diag_transfers_spectral,
    workflow.diag_transfers_physical,
  ],
  qtes={},
  sys=h,       # Dns system
  les=[        # Sgs systems
    m1,
    m2,
    m3,
    m4,
    m5,
  ],
)


