import math
import tqdm

import h5py

import torch
import torch.fft

import matplotlib
import matplotlib.pyplot as plt

from src.grid import TwoGrid
from src.timestepper import ForwardEuler, RungeKutta2, RungeKutta4
from src.pde import Pde, Eq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device =', device)

def to_spectral(y): return torch.fft. rfftn(y, norm='forward')
def to_physical(y): return torch.fft.irfftn(y, norm='forward')

class QgModel:
  def __init__(self, name, Nx, Ny, Lx, Ly, dt, t0, B, mu, nu, nv, eta, source=None, kernel=None, sgs=None):
    self.name = name
    
    self.B = B
    self.mu = mu
    self.nu = nu
    self.nv = nv
    self.eta = eta.to(device)
    
    self.grid = TwoGrid(device, Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)

    if sgs:
      # use 3/2 rule
      self.eq = Eq(grid=self.grid, linear_term=self.linear_term(self.grid), nonlinear_term=self.nonlinear_les)
      self.da = TwoGrid(device, Nx=int((3./2.)*Nx), Ny=int((3./2.)*Ny), Lx=Lx, Ly=Ly, dealias=1/3)
    else:
      # use 2/3 rule
      self.eq = Eq(grid=self.grid, linear_term=self.linear_term(self.grid), nonlinear_term=self.nonlinear_dns)
    self.stepper = RungeKutta4(eq=self.eq)

    self.pde = Pde(dt=dt, t0=t0, eq=self.eq, stepper=self.stepper)
    self.source = source
    self.kernel = kernel
    self.sgs = sgs

  def __str__(self):
    return """Qg model
       Grid: [{nx},{ny}] in [{lx},{ly}]
       μ: {mu}
       ν: {nu}
       β: {beta}
       dt: {dt}
       """.format(
      nx=self.grid.Nx, 
      ny=self.grid.Ny, 
      lx=self.grid.Lx,
      ly=self.grid.Ly,
      mu=self.mu,
      nu=self.nu,
      beta=self.B,
      dt=self.pde.cur.dt)

  def nonlinear_dns(self, i, S, sol, dt, t, grid):
    qh = sol.clone()
    ph = -qh * grid.irsq
    uh = -1j * grid.ky * ph
    vh =  1j * grid.kr * ph

    q = to_physical(qh)
    u = to_physical(uh)
    v = to_physical(vh)

    qe = q + self.eta
    uq = u * qe
    vq = v * qe

    uqh = to_spectral(uq)
    vqh = to_spectral(vq)
    S[:] = -1j * grid.kr * uqh - 1j * grid.ky * vqh

    grid.dealias(S[:])

    if (self.source):
      S[:] += self.source(
        i,
        sol,
        dt,
        t,
        grid)

  def nonlinear_les(self, i, S, sol, dt, t, grid):
    qh = sol.clone()
    ph = -qh * grid.irsq
    uh = -1j * grid.ky * ph
    vh =  1j * grid.kr * ph
    eh = to_spectral(self.eta)

    qhh = self.da.increase(qh)
    uhh = self.da.increase(uh)
    vhh = self.da.increase(vh)
    ehh = self.da.increase(eh)

    q = to_physical(qhh)
    u = to_physical(uhh)
    v = to_physical(vhh)
    e = to_physical(ehh)

    qe = q + e
    uq = u * qe
    vq = v * qe

    uqhh = to_spectral(uq)
    vqhh = to_spectral(vq)

    uqh = grid.reduce(uqhh)
    vqh = grid.reduce(vqhh)

    S[:] = -1j * grid.kr * uqh - 1j * grid.ky * vqh

    if (self.sgs):
      S[:] += self.sgs.predict(
        self,
        i,
        sol,
        grid)

    if (self.source):
      S[:] += self.source(
        i, 
        sol, 
        dt, 
        t, 
        grid)

  def linear_term(self, grid):
    Lc = -self.mu - self.nu * grid.krsq**self.nv - 1j * self.B * grid.kr * grid.irsq
    Lc[0, 0] = 0
    return Lc
    
  # Flow with random gaussian energy only in the wavenumbers range
  def init_randn(self, energy, wavenumbers):
    K = torch.sqrt(self.grid.krsq)
    k = self.grid.kr.repeat(self.grid.Ny, 1)

    qih = torch.randn(self.pde.sol.size(), dtype=torch.complex128).to(device)
    qih[K < wavenumbers[0]] = 0.0
    qih[K > wavenumbers[1]] = 0.0
    qih[k == 0.0] = 0.0

    E0 = energy
    Ei = 0.5 * (self.grid.int_sq(self.grid.kr * self.grid.irsq * qih) + self.grid.int_sq(self.grid.ky * self.grid.irsq * qih)) / (self.grid.Lx * self.grid.Ly)
    qih *= torch.sqrt(E0 / Ei)
    self.pde.sol = qih
    
  def update(self):
    qh = self.pde.sol.clone()
    ph = -qh * self.grid.irsq
    uh = -1j * self.grid.ky * ph
    vh =  1j * self.grid.kr * ph

    # Potential vorticity
    q = to_physical(qh)
    # Streamfunction
    p = to_physical(ph)
    # x-axis velocity
    u = to_physical(uh)
    # y-axis velocity
    v = to_physical(vh)
    return q, p, u, v

  def J(self, grid, qh):
    ph = -qh * grid.irsq
    uh = -1j * grid.ky * ph
    vh =  1j * grid.kr * ph

    q = to_physical(qh)
    u = to_physical(uh)
    v = to_physical(vh)

    uq = u * q
    vq = v * q

    uqh = to_spectral(uq)
    vqh = to_spectral(vq)

    J = 1j * grid.kr * uqh + 1j * grid.ky * vqh
    return J

  def R(self, grid, scale):
    return self.R_field(grid, scale, self.pde.sol)

  def R_field(self, grid, scale, yh):
    return grid.div(torch.stack(self.R_flux(grid, scale, yh), dim=0))

  def R_flux(self, grid, scale, yh):
    qh = yh.clone()
    ph = -qh * self.grid.irsq
    uh = -1j * self.grid.ky * ph
    vh =  1j * self.grid.kr * ph

    q = to_physical(qh)
    u = to_physical(uh)
    v = to_physical(vh)

    uq = u * q
    vq = v * q

    uqh = to_spectral(uq)
    vqh = to_spectral(vq)

    uqh_ = self.kernel(scale * self.grid.delta(), uqh)
    vqh_ = self.kernel(scale * self.grid.delta(), vqh)
    uh_  = self.kernel(scale * self.grid.delta(), uh)
    vh_  = self.kernel(scale * self.grid.delta(), vh)
    qh_  = self.kernel(scale * self.grid.delta(), qh)

    u_ = to_physical(uh_)
    v_ = to_physical(vh_)
    q_ = to_physical(qh_)

    u_q_ = u_ * q_
    v_q_ = v_ * q_

    u_q_h = to_spectral(u_q_)
    v_q_h = to_spectral(v_q_)

    tu = u_q_h - uqh_
    tv = v_q_h - vqh_
    return grid.reduce(tu), grid.reduce(tv)

  # Filters
  def filter(self, grid, scale, y):
    yh = y.clone()
    return grid.reduce(self.kernel(scale * self.grid.delta(), yh))

  def filter_physical(self, grid, scale, y):
    yh = to_spectral(y)
    yl = grid.reduce(self.kernel(scale * self.grid.delta(), yh))
    yl = to_physical(yl)
    return yl

  def run(self, iters, visit, update=False, invisible=False):
    for it in tqdm.tqdm(range(iters), disable=invisible):
      self.pde.step(self)
      visit(self, self.pde.cur, it)
    if update:
      return self.update()

  # Diagnostics
  def energy(self, u, v):
    return 0.5 * torch.mean(u**2 + v**2)

  def enstrophy(self, q):
    return 0.5 * torch.mean(q**2)

  def cfl(self):
    _, _, u, v = self.update()
    return (u.abs().max() * self.pde.cur.dt) / self.grid.dx + (v.abs().max() * self.pde.cur.dt) / self.grid.dy

  def spectrum(self, y):
    K = torch.sqrt(self.grid.krsq)
    d = 0.5
    k = torch.arange(1, int(self.grid.kcut + 1))
    m = torch.zeros(k.size())

    e = [torch.zeros(k.size()) for _ in range(len(y))]
    for ik in range(len(k)):
      n = k[ik]
      i = torch.nonzero((K < (n + d)) & (K > (n - d)), as_tuple=True)
      m[ik] = i[0].numel()
      for j, yj in enumerate(y):
        e[j][ik] = torch.sum(yj[i]) * k[ik] * math.pi / (m[ik] - d)
    return k, e

  def invariants(self, qh):
    ph = -qh * self.grid.irsq
    uh = -1j * self.grid.ky * ph
    vh =  1j * self.grid.kr * ph

    # kinetic energy
    e = torch.abs(uh)**2 + torch.abs(vh)**2
    # enstrophy
    z = torch.abs(qh)**2

    k, [ek, zk] = self.spectrum([e, z])
    return k, ek, zk

  def fluxes(self, R, qh):
    # resolved rate
    sh = -torch.conj(qh) * self.J(self.grid, qh)
    # modeled rate
    lh =  torch.conj(qh) * R

    k, [sk, lk] = self.spectrum([torch.real(sh), torch.real(lh)])
    return k, sk, lk

  # Data
  def save(self, name):
    hf = h5py.File(name, 'w')
    hf.create_dataset('q', data=to_physical(self.p_.sol).cpu().detach())
    hf.close()

  def load(self, name):
    hf = h5py.File(name, 'r')
    fq = hf.get('q')
    sq = to_spectral(torch.from_numpy(fq[:]).to(device))

    # Copy first wavenumbers
    self.pde.sol = self.grid.increase(sq)
    hf.close()

  def zero_grad(self):
    self.stepper.zero_grad()

