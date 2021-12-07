import math
import torch

class Cursor:
  def __init__(self, dt, t0):
    self.dt = dt

    self.t = t0
    self.n = 0

  def step(self):
    self.t += self.dt
    self.n += 1

class Eq:
  def __init__(self, grid, linear_term, nonlinear_term):
    self.device = grid.device

    self.grid = grid
    self.linear_term = linear_term
    self.nonlinear_term = nonlinear_term
    self.dim = linear_term.size()

class Pde:
  def __init__(self, dt, t0, eq, stepper):
    self.device = eq.device

    self.eq = eq
    self.grid = eq.grid
    self.sol = torch.zeros(eq.dim, dtype=torch.complex128, requires_grad=True).to(self.device)
    self.cur = Cursor(dt, t0)
    self.stepper = stepper

  def step(self, m):
    self.stepper.step(m, self.sol, self.cur, self.eq, self.grid)

