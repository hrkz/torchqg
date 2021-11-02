import math
import torch

import numpy as np

class TwoGrid:
  def __init__(self, device, Nx, Ny, Lx, Ly, dealias=1/3):
    self.device = device

    self.Nx = Nx
    self.Ny = Ny
    self.Lx = Lx
    self.Ly = Ly
    self.size = Nx*Ny

    self.dx = Lx/Nx
    self.dy = Ly/Ny
    self.x = torch.arange(start=-Lx/2, end=Lx/2, step=self.dx, dtype=torch.float64).to(device)
    self.y = torch.arange(start=-Ly/2, end=Ly/2, step=self.dy, dtype=torch.float64).to(device)

    self.dk = int(Nx/2 + 1)
    self.kx = torch.reshape(torch.from_numpy(np.fft. fftfreq(Nx, Lx/(Nx*2*math.pi))), (1, self.Nx)).to(device)
    self.ky = torch.reshape(torch.from_numpy(np.fft. fftfreq(Ny, Ly/(Ny*2*math.pi))), (self.Ny, 1)).to(device)
    self.kr = torch.reshape(torch.from_numpy(np.fft.rfftfreq(Nx, Lx/(Nx*2*math.pi))), (1, self.dk)).to(device)

    self.kcut = math.sqrt(2) * (1 - dealias) * min(self.ky.max(), self.kr.max())

    self.krsq = self.kr**2 + self.ky**2
    self.irsq = 1.0 / self.krsq
    self.irsq[0, 0] = 0.0

  def grad(self, y):
    diffx = 1j * self.kr * y
    diffy = 1j * self.ky * y
    return torch.stack((diffx, diffy), dim=0)

  def div(self, y):
    return 1j * self.kr * y[0] + 1j * self.ky * y[1]
  
  def laplacian(self, y):
    return self.div(self.grad(y))

  def norm(self, y):
    return torch.linalg.norm(y, dim=0)

  def int_sq(self, y):
    Y = torch.sum(torch.abs(y[:, 0])**2) + 2*torch.sum(torch.abs(y[:, 1:])**2)
    n = self.Lx * self.Ly
    return Y * n

  def int(self, y):
    Y = torch.sum(y)
    n = self.Lx * self.Ly
    return Y * n

  def decay(self):
    return torch.sqrt(torch.pow(self.kr * self.dx / math.pi, 2) + torch.pow(self.ky * self.dy / math.pi, 2))

  def grid_points(self):
    return torch.meshgrid(self.x, self.y)

  def delta(self):
    d = (self.Lx * self.Ly) / (self.Nx * self.Ny)
    d = d**0.5
    return d

  # Apply cutoff filter on y
  def cutoff(self, delta, y):
    c = math.pi / delta
    y[torch.sqrt(self.krsq) >= c] = 0
    return y

  # Apply gaussian filter on y
  def gaussian(self, delta, y):
    return y * torch.exp(-delta**2 * self.krsq / 24)

  # Discretize y on grid
  def reduce(self, y):
    y_r = y.size()
    z = torch.zeros([self.Ny, self.dk], dtype=torch.complex128, requires_grad=True).to(self.device)
    z[:int(self.Ny / 2),         :self.dk] = y[                         :int(self.Ny / 2), :self.dk]
    z[ int(self.Ny / 2):self.Ny, :self.dk] = y[y_r[0] - int(self.Ny / 2):y_r[0],           :self.dk]
    return z

  # Discretize y on grid
  def increase(self, y):
    y_r = y.size()
    z = torch.zeros([self.Ny, self.dk], dtype=torch.complex128, requires_grad=True).to(self.device)
    z[                         :int(y_r[0] / 2), :y_r[1]] = y[:int(y_r[0] / 2),        :y_r[1]]
    z[self.Ny - int(y_r[0] / 2):self.Ny,         :y_r[1]] = y[ int(y_r[0] / 2):y_r[0], :y_r[1]]
    return z

  # Apply de-aliasing (isotropic, homogeneous)
  def dealias(self, y):
    y[torch.sqrt(self.krsq) > self.kcut] = 0

def aliased_wavenumbers(Nk, dk, dealias):
  L = (1 - dealias)/2
  R = (1 + dealias)/2
  il = math.floor(L*Nk) + 1
  ir = math.ceil (R*Nk)

  p = (il, ir)
  r = (il, dk)
  return p, r
