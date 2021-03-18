import math
import torch

class Cursor:
  def __init__(self, dt):
    self.dt = dt

    self.t = 0.0
    self.n = 0

  def step(self):
    self.t += self.dt
    self.n += 1

class Eq:
  def __init__(self, grid, Lc, NL):
    self.device = grid.device

    self.grid = grid
    self.Lc = Lc
    self.NL = NL
    self.dim = Lc.size()

class Pde:
  def __init__(self, dt, eq, stepper):
    self.device = eq.device

    self.eq = eq
    self.grid = eq.grid
    self.sol = torch.zeros(eq.dim, dtype=torch.complex128, requires_grad=True).to(self.device)
    self.cur = Cursor(dt)
    self.stepper = stepper

  def step(self, m):
    self.stepper.step(m, self.sol, self.cur, self.eq, self.grid)

  def reset(self):
    self.cur.t = 0.0
    self.cur.n = 0
