import math
import torch

import qg

class Constant:
  def __init__(self, c=0.0):
    self.c = c

  def predict(self, m, it, sol, grid):
    div = torch.full_like(sol, self.c)
    return div

class Hyperviscosity:
  def __init__(self, v):
    self.v = v

  def predict(self, m, it, sol, grid):
    qh = sol.clone()
    div = -self.v * grid.krsq**2 * qh
    return div

def coef_leith(m, sol, grid, sx, sy):
  test_filter = 2.0
  test_delta = grid.delta() * test_filter

  qh = sol.clone()
  ph = -qh * grid.irsq
  uh = -1j * grid.ky * ph
  vh =  1j * grid.kr * ph

  q = qg._p(qh)
  u = qg._p(uh)
  v = qg._p(vh)

  uq = u * q
  vq = v * q

  uqh = qg._s(uq)
  vqh = qg._s(vq)

  uqh_ = m.fil_(test_delta, uqh)
  vqh_ = m.fil_(test_delta, vqh)
  uh_  = m.fil_(test_delta, uh)
  vh_  = m.fil_(test_delta, vh)
  qh_  = m.fil_(test_delta, qh)

  u_ = qg._p(uh_)
  v_ = qg._p(vh_)
  q_ = qg._p(qh_)

  u_q_ = u_ * q_
  v_q_ = v_ * q_

  u_q_h = qg._s(u_q_)
  v_q_h = qg._s(v_q_)

  Lu = qg._p(u_q_h - uqh_)
  Lv = qg._p(v_q_h - vqh_)

  g_q = grid.grad(qh_)
  q_x = qg._p(g_q[0])
  q_y = qg._p(g_q[1])

  s_n = torch.sqrt(q_x**2 + q_y**2)
  s_x = s_n * q_x
  s_y = s_n * q_y

  sxh = qg._s(sx)
  syh = qg._s(sy)

  sx_ = qg._p(m.fil_(test_delta, sxh))
  sy_ = qg._p(m.fil_(test_delta, syh))

  Mu = test_delta**3 * s_x - grid.delta()**3 * sx_
  Mv = test_delta**3 * s_y - grid.delta()**3 * sy_

  L = Lu * Mu + Lv * Mv
  M = Mu * Mu + Mv * Mv
  #c = torch.sum(L) / torch.sum(M)
  c = torch.sum(0.5 * (L + L.abs())) / torch.sum(M)
  return c

class Leith:
  def __init__(self, c=0.25):
    self.c = c

  def predict(self, m, it, sol, grid):
    qh = sol.clone()
    
    gq = grid.grad(qh)
    qx = qg._p(gq[0])
    qy = qg._p(gq[1])

    sn = torch.sqrt(qx**2 + qy**2)
    sx = sn * qx
    sy = sn * qy

    tst = coef_leith(m, sol, grid, sx, sy) if self.c is None else self.c**3
    cte = tst * grid.delta()**3

    rx = qg._s(cte * sx)
    ry = qg._s(cte * sy)

    div = grid.div(torch.stack((rx, ry), dim=0))
    return div

def coef_smag(m, sol, grid, sx, sy):
  test_filter = 2.0
  test_delta = grid.delta() * test_filter

  qh = sol.clone()
  ph = -qh * grid.irsq
  uh = -1j * grid.ky * ph
  vh =  1j * grid.kr * ph

  q = qg._p(qh)
  u = qg._p(uh)
  v = qg._p(vh)

  uq = u * q
  vq = v * q

  uqh = qg._s(uq)
  vqh = qg._s(vq)

  uqh_ = m.fil_(test_delta, uqh)
  vqh_ = m.fil_(test_delta, vqh)
  uh_  = m.fil_(test_delta, uh)
  vh_  = m.fil_(test_delta, vh)
  qh_  = m.fil_(test_delta, qh)

  u_ = qg._p(uh_)
  v_ = qg._p(vh_)
  q_ = qg._p(qh_)

  u_q_ = u_ * q_
  v_q_ = v_ * q_

  u_q_h = qg._s(u_q_)
  v_q_h = qg._s(v_q_)

  Lu = qg._p(u_q_h - uqh_)
  Lv = qg._p(v_q_h - vqh_)

  g_u = grid.grad(uh_)
  g_v = grid.grad(vh_)
  u_x = qg._p(g_u[0])
  u_y = qg._p(g_u[1])
  v_x = qg._p(g_v[0])
  v_y = qg._p(g_v[1])

  g_q = grid.grad(qh_)
  q_x = qg._p(g_q[0])
  q_y = qg._p(g_q[1])

  s_n = torch.sqrt(2 * u_x**2 + 2 * v_y**2 + (v_x + u_y)**2)
  s_x = s_n * q_x
  s_y = s_n * q_y

  sxh = qg._s(sx)
  syh = qg._s(sy)

  sx_ = qg._p(m.fil_(test_delta, sxh))
  sy_ = qg._p(m.fil_(test_delta, syh))

  Mu = test_delta**2 * s_x - grid.delta()**2 * sx_
  Mv = test_delta**2 * s_y - grid.delta()**2 * sy_
  
  L = Lu * Mu + Lv * Mv
  M = Mu * Mu + Mv * Mv
  #c = torch.sum(L) / torch.sum(M)
  c = torch.sum(0.5 * (L + L.abs())) / torch.sum(M)
  return c

class Smagorinsky:
  def __init__(self, c=0.18):
    self.c = c

  def predict(self, m, it, sol, grid):
    qh = sol.clone()
    ph = -qh * grid.irsq
    uh = -1j * grid.ky * ph
    vh =  1j * grid.kr * ph

    gu = grid.grad(uh)
    gv = grid.grad(vh)

    ux = qg._p(gu[0])
    uy = qg._p(gu[1])
    vx = qg._p(gv[0])
    vy = qg._p(gv[1])

    gq = grid.grad(qh)
    qx = qg._p(gq[0])
    qy = qg._p(gq[1])

    sn = torch.sqrt(2 * ux**2 + 2 * vy**2 + (vx + uy)**2)
    sx = sn * qx
    sy = sn * qy

    tst = coef_smag(m, sol, grid, sx, sy) if self.c is None else self.c**2
    cte = tst * grid.delta()**2

    rx = qg._s(cte * sx)
    ry = qg._s(cte * sy)

    div = grid.div(torch.stack((rx, ry), dim=0))
    return div

class Gradient:
  def __init__(self, scale, c=0.08):
    self.c = c

  def predict(self, m, it, sol, grid):
    qh = sol.clone()
    ph = -qh * grid.irsq
    uh = -1j * grid.ky * ph
    vh =  1j * grid.kr * ph

    cte = -self.c * grid.delta()**2

    gq = grid.grad(qh)
    gu = grid.grad(uh)
    gv = grid.grad(vh)

    qx = qg._p(gq[0])
    qy = qg._p(gq[1])
    ux = qg._p(gu[0])
    uy = qg._p(gu[1])
    vx = qg._p(gv[0])
    vy = qg._p(gv[1])

    r1 = cte * (ux*qx + uy*qy)
    r2 = cte * (vx*qx + vy*qy)
    rx = qg._s(r1)
    ry = qg._s(r2)
    div = grid.div(torch.stack((rx, ry), dim=0))
    return div

class MLdiv:
  def __init__(self, model):
    self.model = model
    self.model.eval()
    #print(self.model)

  def predict(self, m, it, sol, grid):
    qh = sol.clone()
    ph = -qh * grid.irsq

    q = qg._p(qh)
    p = qg._p(ph)

    # M(q, p) = M({i})
    i = torch.stack((q, p), dim=0) 
    # M({i}) ~ r
    r = self.model(i.unsqueeze(0).to(torch.float32)).view(grid.Ny, grid.Nx)
    return qg._s(r)

