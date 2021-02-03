import math
import tqdm

import h5py

import torch
import torch.fft

import matplotlib
import matplotlib.pyplot as plt

from src.grid import TwoGrid
from src.explicit import ForwardEuler, RungeKutta4
from src.pde import Pde, Eq

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device =', device)

def _s(y): return torch.fft. rfftn(y, norm='forward')
def _p(y): return torch.fft.irfftn(y, norm='forward')

class QgModel:
  def __init__(self, name, Nx, Ny, Lx, Ly, dt, B, mu, nu, nv, eta, forcing=None, sgs=None):
    self.name = name
    
    self.B = B
    self.mu = mu
    self.nu = nu
    self.nv = nv
    self.eta = eta.to(device)
    
    self.g_ = TwoGrid(device, Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)

    self.e_ = Eq(grid=self.g_, Lc=self.qg_Lin(self.g_), NL=self.qg_NL)
    self.s_ = RungeKutta4(eq=self.e_)

    self.p_ = Pde(dt=dt, eq=self.e_, stepper=self.s_)
    self.f_ = forcing
    self.sgs_ = sgs

  def qg_NL(self, i, S, sol, dt, t, grid):
    qh = sol.clone()
    ph = -qh * grid.irsq
    uh = -1j * grid.ky * ph
    vh =  1j * grid.kr * ph

    q = _p(qh)
    u = _p(uh)
    v = _p(vh)

    qe = q + self.eta
    uq = u * qe
    vq = v * qe

    uqh = _s(uq)
    vqh = _s(vq)
    S[:] = -1j * grid.kr * uqh - 1j * grid.ky * vqh
  
    if (self.f_):
      S[:] += self.f_(i, sol, dt, t, grid)
    if (self.sgs_):
      S[:] += self.sgs_.predict(self, sol, dt, t, grid)

    grid.dealias(S[:])

  def qg_Lin(self, grid):
    Lc = -self.mu - self.nu * grid.krsq**self.nv + 1j * self.B * grid.kr * grid.irsq
    Lc[0, 0] = 0
    return Lc
    
  # Flow with energy only in the wavenumbers range
  def ic_spectral(self, energy, wavenumbers):
    K = torch.sqrt(self.g_.krsq)
    k = self.g_.kr.repeat(self.g_.Ny, 1)

    qih = torch.randn(self.p_.sol.size(), dtype=torch.complex128).to(device)
    qih[K < wavenumbers[0]] = 0.0
    qih[K > wavenumbers[1]] = 0.0
    qih[k == 0.0] = 0.0
    qih[0, 0] = 0

    E0 = energy
    Ei = 0.5 * (self.g_.int_sq(self.g_.kr * self.g_.irsq * qih) + self.g_.int_sq(self.g_.ky * self.g_.irsq * qih)) / (self.g_.Lx * self.g_.Ly)
    qih *= torch.sqrt(E0 / Ei)
    self.p_.sol = qih
    
  def update(self):
    qh = self.p_.sol
    ph = -qh * self.g_.irsq
    uh = -1j * self.g_.ky * ph
    vh =  1j * self.g_.kr * ph

    # Potential vorticity
    q = _p(qh)
    # Streamfunction
    p = _p(ph)
    # x-axis velocity
    u = _p(uh)
    # y-axis velocity
    v = _p(vh)
    return q, p, u, v

  def Jac(self, grid, a, b):
    bx = _p(1j * grid.kr * b)
    by = _p(1j * grid.ky * b)
    return 1j * grid.kr * _s(_p(a) * by) - 1j * grid.ky * _s(_p(a) * bx)

  def J(self, grid, qh):
    ph = -qh * grid.irsq
    uh = -1j * grid.ky * ph
    vh =  1j * grid.kr * ph

    q = _p(qh)
    u = _p(uh)
    v = _p(vh)

    uq = u * q
    vq = v * q

    uqh = _s(uq)
    vqh = _s(vq)

    return 1j * grid.kr * uqh + 1j * grid.ky * vqh

  def R(self, grid, width):
    qh = self.p_.sol
    ph = -qh * self.g_.irsq

    # J(q, p)_
    Jh_ = self.cutoff(width, grid, self.Jac(self.g_, qh, ph))
    # J(q_, p_)
    J_h = self.Jac(grid, self.cutoff(width, grid, qh), self.cutoff(width, grid, ph))
    return J_h - Jh_
  
  # Filters
  def cutoff(self, width, grid, y):
    yh = y.clone()
    return grid.reduce(self.g_.cutoff(width * self.g_.delta(), yh))

  def symm(self, y):
    y[:, 1:-1] *= 2.0

  def cutoff_physical(self, width, grid, y):
    yh = _s(y)
    yl = grid.reduce(self.g_.cutoff(width * self.g_.delta(), yh))
    yl = _p(yl)
    return yl

  def run(self, N, visit, update=False):
    for it in tqdm.tqdm(range(N)):
      self.p_.step(self)
      visit(self, self.p_.cur, it)
    if update:
      return self.update()

  # Diagnostics (outside of the Model)
  def energy(self, u, v):
    return 0.5 * torch.mean(u**2 + v**2)

  def enstrophy(self, q):
    return 0.5 * torch.mean(q**2)

  def fluxes(self, width, grid):
    qh  = self.p_.sol
    qh_ = self.cutoff(width, grid, qh)

    s_ = -torch.imag(torch.conj(qh_) * self.J(grid, qh_))
    self.symm(s_)
    l_ =  torch.imag(torch.conj(qh_) * self.R(grid, width))
    self.symm(l_)

    K = torch.sqrt(grid.krsq)
    d = 0.5
    k = torch.arange(1, grid.Nx // 2 - 1)
    m = torch.zeros(k.size())
    A = torch.zeros(k.size())

    es = torch.zeros(k.size())
    el = torch.zeros(k.size())

    for ik in range(len(k)):
      n = k[ik]
      i = torch.nonzero((K < (k[ik] + d)) & (K > (k[ik] - d)), as_tuple=True)
      m[ik] = i[0].numel()
      es[ik] = torch.sum(s_[i]) * k[ik] * math.pi / (m[ik] - d)
      el[ik] = torch.sum(l_[i]) * k[ik] * math.pi / (m[ik] - d)

    return k, es, el

  def spectrum(self):
    qh = self.p_.sol
    ph = -qh * self.g_.irsq
    uh = -1j * self.g_.ky * ph
    vh =  1j * self.g_.kr * ph

    u = _p(uh)
    v = _p(vh)
    kin = 0.5 * (torch.abs(uh)**2 + torch.abs(vh)**2)
    self.symm(kin)
    ens = 0.5 * (torch.abs(qh)**2)
    self.symm(ens)

    K = torch.sqrt(self.g_.krsq)
    d = 0.5
    k = torch.arange(1, self.g_.Nx // 2 - 1)
    m = torch.zeros(k.size())
    A = torch.zeros(k.size())
    
    ek = torch.zeros(k.size())
    en = torch.zeros(k.size())

    for ik in range(len(k)):
      n = k[ik]
      i = torch.nonzero((K < (n + d)) & (K > (n - d)), as_tuple=True)
      m[ik] = i[0].numel()
      ek[ik] = torch.sum(kin[i]) * k[ik] * math.pi / (m[ik] - d)
      en[ik] = torch.sum(ens[i]) * k[ik] * math.pi / (m[ik] - d)

    return k, ek, en

  # Data
  def save(self, name):
    hf = h5py.File(name, 'w')
    hf.create_dataset('q', data=_p(self.p_.sol.cpu().detach()))
    hf.close()

  def load(self, name):
    hf = h5py.File(name, 'r')
    fq = hf.get('q')
    self.p_.sol = _s(torch.from_numpy(fq[:]).to(device))
    hf.close()

  def zero_grad(self):
    self.s_.zero_grad()
