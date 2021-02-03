import math
import torch

def filter(K, order=4, inner=0.65, outer=1):
  decay = 15.0 * math.log(10) / (outer - inner)**order
  filt  = torch.exp(-decay * torch.pow(K - inner, order))
  filt[K < inner] = 1.0
  return filt

class ForwardEuler:
  def __init__(self, eq):
    self.n = 1
    self.S = torch.zeros(eq.dim, dtype=torch.complex128, requires_grad=True).to(eq.device)

  def step(self, sol, cur, eq, grid):
    eq.NL(
      0,
      self.S, 
      sol,
      cur.dt, 
      cur.t, 
      grid,
    )

    sol += cur.dt*(eq.Lc*sol.clone() + self.S)
    cur.step()

class RungeKutta4:
  def __init__(self, eq):
    self.n    = 4
    self.f    = filter(eq.grid.decay())
    self.S    = torch.zeros(eq.dim, dtype=torch.complex128, requires_grad=True).to(eq.device)
    self.rhs1 = torch.zeros(eq.dim, dtype=torch.complex128, requires_grad=True).to(eq.device)
    self.rhs2 = torch.zeros(eq.dim, dtype=torch.complex128, requires_grad=True).to(eq.device)
    self.rhs3 = torch.zeros(eq.dim, dtype=torch.complex128, requires_grad=True).to(eq.device)
    self.rhs4 = torch.zeros(eq.dim, dtype=torch.complex128, requires_grad=True).to(eq.device)

  def zero_grad(self):
    self.f.detach_()
    self.S.detach_()
    self.rhs1.detach_()
    self.rhs2.detach_()
    self.rhs3.detach_()

  def step(self, m, sol, cur, eq, grid):
    dt = cur.dt
    t  = cur.t

    # substep 1
    eq.NL(0, self.rhs1, sol, dt, t, grid)
    self.rhs1 += eq.Lc*sol

    # substep 2
    self.S = sol + self.rhs1 * dt*0.5
    eq.NL(1, self.rhs2, self.S, dt*0.5, t + dt*0.5, grid)
    self.rhs2 += eq.Lc*self.S

    # substep 3
    self.S = sol + self.rhs2 * dt*0.5
    eq.NL(2, self.rhs3, self.S, dt*0.5, t + dt*0.5, grid)
    self.rhs3 += eq.Lc*self.S

    # substep 4
    self.S = sol + self.rhs3 * dt
    eq.NL(3, self.rhs4, self.S, dt, t + dt, grid)
    self.rhs4 += eq.Lc*self.S

    sol += dt*(self.rhs1/6.0 + self.rhs2/3.0 + self.rhs3/3.0 + self.rhs4/6.0)
    sol *= self.f
    cur.step()

