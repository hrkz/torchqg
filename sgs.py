import math
import torch

import qg

class Constant:
  def __init__(self, c=0.0):
    self.c = c

  def predict(self, m, sol, grid):
    div = torch.full_like(sol, self.c)
    return div

class Hyperviscosity:
  def __init__(self, v):
    self.v = v

  def predict(self, m, sol, grid):
    qh = sol.clone()
    div = -self.v * grid.krsq**2 * qh
    return div

class Leith:
  def __init__(self, c=0.25):
    self.c = c

  def predict(self, m, sol, grid):
    qh = sol.clone()
    cte = self.c * grid.delta()**3
    gh = grid.grad(qh)
    gx = qg._p(gh[0])
    gy = qg._p(gh[1])
    r = cte * torch.sqrt(gx**2 + gy**2) * torch.stack((gx, gy), dim=0)
    rx = qg._s(r[0])
    ry = qg._s(r[1])
    div = grid.div(torch.stack((rx, ry), dim=0))
    return div

class Smagorinsky:
  def __init__(self, c=0.18):
    self.c = c

  def predict(self, m, sol, grid):
    qh = sol.clone()
    ph = -qh * grid.irsq
    cte = self.c * grid.delta()**2
    gh = grid.grad(qh)
    gx = qg._p(gh[0])
    gy = qg._p(gh[1])
   
    shx = 1j * grid.kr * ph
    shy = 1j * grid.ky * ph
    shxy = qg._p(1j * grid.ky * shx)
    shxx = qg._p(1j * grid.kr * shx)
    shyy = qg._p(1j * grid.ky * shy)
    
    r = cte * torch.sqrt(4 * shxy**2 + (shxx - shyy)**2) * torch.stack((gx, gy), dim=0)
    rx = qg._s(r[0])
    ry = qg._s(r[1])
    div = grid.div(torch.stack((rx, ry), dim=0))
    return div

class Gradient:
  def __init__(self, scale, c=-0.08):
    self.c = c
    self.s = scale

  def predict(self, m, sol, grid):
    qh = sol.clone()
    ph = -qh * grid.irsq
    uh = -1j * grid.ky * ph
    vh =  1j * grid.kr * ph

    cte = self.c * (self.s * grid.delta())**2

    qgh = grid.grad(qh)
    ugh = grid.grad(uh)
    vgh = grid.grad(vh)
    qgx, qgy = qg._p(qgh[0]), qg._p(qgh[1])
    ugx, ugy = qg._p(ugh[0]), qg._p(ugh[1])
    vgx, vgy = qg._p(vgh[0]), qg._p(vgh[1])

    r1 = cte * (ugx*qgx + ugy*qgy)
    r2 = cte * (vgx*qgx + vgy*qgy)
    rx = qg._s(r1)
    ry = qg._s(r2)
    div = grid.div(torch.stack((rx, ry), dim=0))
    return div

class Perfect:
  def __init__(self, dns, scale):
    self.dns = dns
    self.scale = scale

  def predict(self, m, sol, grid):
    return self.dns.R(grid, self.scale)
