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

Lx = 2*math.pi
Ly = 2*math.pi
Nx = 512
Ny = 512

eta = torch.zeros([Ny, Nx], dtype=torch.float64, requires_grad=True)

# High res model
h = QgModel(
  name='\\mathcal{F}',
  Nx=Nx,
  Ny=Ny,
  Lx=Lx,
  Ly=Ly,
  dt=0.005,
  B=10.0,   # Planetary vorticity y-gradient
  mu=0.01,  # Linear drag
  nu=0,     # Viscosity coefficient
  nv=1,     # Hyperviscous order (nv=1 is viscosity)
  eta=eta   # Topographic PV
)

# Initial conditions
#h.ic_spectral(0.01, [3.0, 4.0])
# Loading fields
h.load('qg_workflows/dev/q_inviscid.h5')

# Forcing
E0 = 5e-6

# Delta correlated in time and homogeneous-isotropic correlated in space
f_wn = 8.0
f_bd = 1.0

K = torch.sqrt(h.g_.krsq)
k = h.g_.kr.repeat(h.g_.Ny, 1)

qih = torch.exp(-(K - f_wn)**2 / (2 * f_bd**2))
qih[K < 7.0] = 0
qih[K > 9.0] = 0
qih[k < 1.0] = 0
qih[0, 0] = 0
Ei = 0.5 * (h.g_.int_sq(h.g_.kr * h.g_.irsq * qih) + h.g_.int_sq(h.g_.ky * h.g_.irsq * qih)) / (h.g_.Lx * h.g_.Ly)
qih *= torch.sqrt(E0 / Ei)

sto = torch.zeros((h.s_.n,) + h.p_.sol.size(), dtype=torch.complex128).to(h.g_.device)
def Fs(i, sol, dt, t, grid):
  global sto
  sto[i] = torch.sqrt(qih) * torch.exp(2*math.pi * 1j * torch.rand(sol.size(), dtype=torch.float64).to(grid.device)) / math.sqrt(dt)
  return sto[i]

# Stochastic forcing
h.f_ = Fs

# Low res model(s)
delta = 4

Nxl = int(Nx / delta)
Nyl = int(Ny / delta)

eta_m = torch.zeros([Nyl, Nxl], dtype=torch.float64, requires_grad=True)

# No model
m1 = QgModel(
  name='\\mathcal{R}',
  Nx=Nxl,
  Ny=Nyl,
  Lx=Lx,
  Ly=Ly,
  dt=0.005,
  B=10.0,   # Planetary vorticity y-gradient
  mu=0.01,  # Linear drag
  nu=0,     # Viscosity coefficient
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
  dt=0.005,
  B=10.0,   # Planetary vorticity y-gradient
  mu=0.01,  # Linear drag
  nu=0,     # Viscosity coefficient
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
  dt=0.005,
  B=10.0,   # Planetary vorticity y-gradient
  mu=0.01,  # Linear drag
  nu=0,     # Viscosity coefficient
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
  dt=0.005,
  B=10.0,   # Planetary vorticity y-gradient
  mu=0.01,  # Linear drag
  nu=0,     # Viscosity coefficient
  nv=1,     # Hyperviscous order (nv=1 is viscosity)
  eta=eta_m # Topographic PV
)

# Subgrid-scale models
m1.sgs_ = Constant(constant=0.0)
m2.sgs_ = Leith(c=0.22)
m3.sgs_ = Smagorinsky(c=0.11)
#m4.sgs_ = Learned(model=torch.load('qg_models/cnn/weights.pyt'))

# Initialize from DNS vorticity field
m1.p_.sol = h.cutoff(delta, m1.g_, h.p_.sol)
m2.p_.sol = h.cutoff(delta, m2.g_, h.p_.sol)
m3.p_.sol = h.cutoff(delta, m3.g_, h.p_.sol)
m4.p_.sol = h.cutoff(delta, m4.g_, h.p_.sol)

def Fsl(i, sol, dt, t, grid):
  stol = h.cutoff(delta, grid, sto[i])
  return stol

# Use forcing from DNS simulation
m1.f_ = Fsl
m2.f_ = Fsl
m3.f_ = Fsl
m4.f_ = Fsl

workflow.workflow(
  name='dev',
  iters=5000, # Model iterations
  steps=50,   # Discrete steps
  delta=delta,  # Kernel scale
  diags=[       # Diagnostics
    workflow.diag_show,
    workflow.diag_sgs_metrics,
    workflow.diag_spatial_stats,
    workflow.diag_temporal_stats,
    workflow.diag_integrals,
    workflow.diag_spectra,
  ],
  qtes={},
  sys=h,       # Dns system
  les=[        # Sgs systems
    m1,
    m2,
    m3,
    #m4,
  ],
)

