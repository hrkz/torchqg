import math

import torch
import torch.nn as nn

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from qg import _s, _p, QgModel
from sgs import Learned, Constant, Leith, Smagorinsky

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
Nx = 512
Ny = 512

dt = 600 / t_unit() # 600s
mu = 1.25e-8 / l_unit()**-1 # 1.25e-8m^-1
#nu = 88  / l_unit()**2 / t_unit()**-1 # 88m^2s^-1 for startup
nu = 1.375 / l_unit()**2 / t_unit()**-1 # 1.375m^2s^-1 for the simulation

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
#h.ic_spectral(0.01, [3.0, 5.0])
# Loading fields
h.load('qg_workflows/geo/q_end.h5')

# Wind stress forcing
def Fs(i, sol, dt, t, grid):
  phi_x = math.pi * math.sin(1.2e-6 / t_unit()**-1 * t)
  phi_y = math.pi * math.sin(1.2e-6 * math.pi / t_unit()**-1 * t / 3)
  y = torch.cos(4 * grid.y + phi_y).view(grid.Ny, 1) - torch.cos(4 * grid.x + phi_x).view(1, grid.Nx)

  yh = _s(y)
  K = torch.sqrt(grid.krsq)
  yh[K < 3.0] = 0
  yh[K > 5.0] = 0

  e0 = 1.75e-18 / t_unit()**-3
  ei = 0.5 * grid.int_sq(yh) / (grid.Lx * grid.Ly)
  yh *= torch.sqrt(e0 / ei)
  return yh

# Stochastic forcing
h.f_ = Fs

# Low res model(s)
scale = 4

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

# Leith model
m2 = QgModel(
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
m3 = QgModel(
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

# CNN model
m4 = QgModel(
  name='\\mathcal{R}^{\\mathrm{CNN}}',
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
m1.sgs_ = Constant(constant=0.0)
m2.sgs_ = Leith(c=0.22)
m3.sgs_ = Smagorinsky(c=0.11)
m4.sgs_ = Learned(model=torch.load('qg_models/cnn/weights.pyt'))

# Initialize from DNS vorticity field
m1.p_.sol = h.cutoff(m1.g_, scale, h.p_.sol)
m2.p_.sol = h.cutoff(m2.g_, scale, h.p_.sol)
m3.p_.sol = h.cutoff(m3.g_, scale, h.p_.sol)
m4.p_.sol = h.cutoff(m4.g_, scale, h.p_.sol)

# Use forcing from DNS simulation
m1.f_ = Fs
m2.f_ = Fs
m3.f_ = Fs
m4.f_ = Fs

workflow.workflow(
  name='geo',
  iters=10000,  # Model iterations
  steps=100,    # Discrete steps
  scale=scale,  # Kernel scale
  diags=[       # Diagnostics
    workflow.diag_show,
    workflow.diag_sgs_metrics,
    workflow.diag_spatial_stats,
    workflow.diag_temporal_stats,
    workflow.diag_integrals,
    workflow.diag_spectrum,
    workflow.diag_fluxes,
  ],
  qtes={},
  sys=h,       # Dns system
  les=[        # Sgs systems
    m1,
    m2,
    m3,
    m4,
  ],
)

