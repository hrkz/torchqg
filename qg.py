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
  def __init__(self, name, Nx, Ny, Lx, Ly, dt, B, mu, nu, nv, eta, forcing=None, filter=None, sgs=None):
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
    self.fil_ = filter
    self.sgs_ = sgs

  def __str__(self):
    return """Qg model
       Grid: [{nx},{ny}] in [{lx},{ly}]
       μ: {mu}
       ν: {nu}
       β: {beta}
       dt: {dt}
       """.format(
      nx=self.g_.Nx, 
      ny=self.g_.Ny, 
      lx=self.g_.Lx,
      ly=self.g_.Ly,
      mu=self.mu,
      nu=self.nu,
      beta=self.B,
      dt=self.p_.cur.dt)

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
 
    grid.dealias(S[:])

    if (self.f_):
      S[:] += self.f_(
        i, 
        sol, 
        dt, 
        t, 
        grid)
    if (self.sgs_):
      S[:] += self.sgs_.predict(
        self, 
        sol, 
        grid)

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

    J = 1j * grid.kr * uqh + 1j * grid.ky * vqh
    grid.dealias(J[:])
    return J

  def R(self, grid, scale):
    qh = self.p_.sol.clone()
    # J(p, q)_
    Jh_ = self.filter(
      grid, 
      scale, 
      self.J(self.g_, qh))
    #Jh_ = self.fil_(self.g_.delta() * scale, self.J(self.g_, qh))
    # J(p_, q_)
    J_h = self.J(
      grid, 
      self.filter(grid, scale, qh))
    #J_h = self.J(self.g_, self.fil_(self.g_.delta() * scale, qh))
    return J_h - Jh_

  # Filters
  def filter(self, grid, scale, y):
    yh = y.clone()
    return grid.reduce(self.fil_(scale * self.g_.delta(), yh))

  def symm(self, y):
    y[:, 1:-1] *= 2.0

  def filter_physical(self, grid, scale, y):
    yh = _s(y)
    yl = grid.reduce(self.fil_(scale * self.g_.delta(), yh))
    yl = _p(yl)
    return yl

  def run(self, iters, visit, update=False):
    for it in tqdm.tqdm(range(iters)):
      visit(self, self.p_.cur, it)
      self.p_.step(self)
    if update:
      return self.update()

  # Diagnostics
  def energy(self, u, v):
    return 0.5 * torch.mean(u**2 + v**2)

  def enstrophy(self, q):
    return 0.5 * torch.mean(q**2)

  def cfl(self):
    _, _, u, v = self.update()
    return torch.stack((u, v), dim=0).abs().max() * self.p_.cur.dt / min(self.g_.dx, self.g_.dy)

  def vrms(self):
    _, _, u, v = self.update()
    return torch.stack((u, v), dim=0).pow(2).mean().sqrt()

  def re(self, L=1):
    return self.vrms() * min(self.g_.Lx, self.g_.Ly) / L / self.nu

  def eddy_time(self, L=1):
    e = 0.5 * self.g_.int_sq(self.p_.sol) / (self.g_.Lx * self.g_.Ly)
    return min(self.g_.Lx, self.g_.Ly) / L * math.sqrt(1.0 / e)

  def spectrum(self, y):
    K = torch.sqrt(self.g_.krsq)
    d = 0.5
    k = torch.arange(1, self.g_.Nx // 2)
    m = torch.zeros(k.size())
    A = torch.zeros(k.size())

    e = [torch.zeros(k.size()) for _ in range(len(y))]
    for ik in range(len(k)):
      n = k[ik]
      i = torch.nonzero((K < (n + d)) & (K > (n - d)), as_tuple=True)
      m[ik] = i[0].numel()
      for j, yj in enumerate(y):
        e[j][ik] = torch.sum(yj[i]) * k[ik] * math.pi / (m[ik] - d)
    return k, e

  def invariants(self):
    qh = self.p_.sol
    ph = -qh * self.g_.irsq
    uh = -1j * self.g_.ky * ph
    vh =  1j * self.g_.kr * ph

    # kinetic energy
    e = 0.5 * (torch.abs(uh)**2 + torch.abs(vh)**2)
    self.g_.dealias(e)
    self.symm(e)

    # enstrophy
    z = 0.5 * (torch.abs(qh)**2)
    self.symm(z)
    self.g_.dealias(z)

    k, [ek, zk] = self.spectrum([e, z])
    return k, ek, zk

  def fluxes(self, R, qh):
    # resolved rate
    sh = -torch.conj(qh) * self.J(self.g_, qh)
    # modelled rate
    lh = torch.conj(qh) * R

    k, [sk, lk] = self.spectrum([torch.real(sh), torch.real(lh)])
    return k, sk, lk

  # Data
  def save(self, name):
    hf = h5py.File(name, 'w')
    hf.create_dataset('q', data=_p(self.p_.sol).cpu().detach())
    hf.close()

  def load(self, name):
    hf = h5py.File(name, 'r')
    fq = hf.get('q')
    sq = _s(torch.from_numpy(fq[:]).to(device))

    # Copy first wavenumbers
    self.p_.sol = self.g_.increase(sq)
    hf.close()

  def zero_grad(self):
    self.s_.zero_grad()
