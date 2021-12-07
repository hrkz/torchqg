import math
import torch

import qg

class Constant:
  def __init__(self, c=0.0):
    self.c = c

  def predict(self, m, it, sol, grid):
    div = torch.full_like(sol, self.c)
    return div

class MLdiv:
  def __init__(self, model):
    self.model = model
    self.model.eval()
    #print(self.model)

  def predict(self, m, it, sol, grid):
    qh = sol.clone()
    ph = -qh * grid.irsq

    q = qg.to_physical(qh)
    p = qg.to_physical(ph)

    # M(q, p) = M({i})
    i = torch.stack((q, p), dim=0) 
    # M({i}) ~ r
    r = self.model(i.unsqueeze(0).to(torch.float32)).view(grid.Ny, grid.Nx)
    return qg.to_spectral(r)

