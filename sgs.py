import math
import torch

import qg

class Constant:
  def __init__(self, constant=0.0):
    self.constant = constant

  def predict(self, m, sol, grid):
    div = torch.full_like(sol, self.constant)
    return div

class Leith:
  def __init__(self, c=1.0):
    self.c = c

  def predict(self, m, sol, grid):
    qh = sol.clone()
    cte = (self.c * grid.delta())**3
    gh = grid.grad(qh)
    gx = qg._p(gh[0])
    gy = qg._p(gh[1])
    r = cte * torch.sqrt(gx**2 + gy**2) * torch.stack((gx, gy), dim=0)
    rx = qg._s(r[0])
    ry = qg._s(r[1])
    div = grid.div(torch.stack((rx, ry), dim=0))
    return div

class Smagorinsky:
  def __init__(self, c=0.5):
    self.c = c

  def predict(self, m, sol, grid):
    qh = sol.clone()
    ph = -qh * grid.irsq
    cte = (self.c * grid.delta())**2
    gh = grid.grad(qh)
    gx = qg._p(gh[0])
    gy = qg._p(gh[1])
    
    shxy = qg._p(1j * grid.kr * ph + 1j * grid.ky * ph)
    shxx = qg._p(1j * grid.kr * ph + 1j * grid.kr * ph)
    shyy = qg._p(1j * grid.ky * ph + 1j * grid.ky * ph)
    
    r = cte * torch.sqrt(4 * shxy**2 + (shxx - shyy)**2) * torch.stack((gx, gy), dim=0)
    rx = qg._s(r[0])
    ry = qg._s(r[1])
    div = grid.div(torch.stack((rx, ry), dim=0))
    return div

class Learned:
  def __init__(self, model):
    self.model = model
    self.model.eval()
    print(self.model)

  def predict(self, m, sol, grid):
    q, p, u, v = m.update()

    i = p
    i = (i - i.mean()) / i.std()
    
    r = self.model(i.unsqueeze(0).unsqueeze(0)).view(grid.Ny, grid.Nx)
    return qg._s(r)
