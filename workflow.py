import math
import os
import tqdm

import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import qg

plt.rcParams.update({'mathtext.fontset':'cm'})

def workflow(
  name,
  iters, 
  steps,
  delta,
  diags,
  qtes,
  sys,
  les,
): 
  store = int(iters / steps)

  Nx = sys.g_.Nx
  Ny = sys.g_.Ny
  Nxl = int(Nx / delta)
  Nyl = int(Ny / delta)

  sgs_grid = les[0].g_

  qts = {}
  for qte in qtes:
    qts[qte] = {}
    qts[qte][sys.name] = []
    for model in les:
      qts[qte][model.name] = []

  dns = torch.zeros([steps, 5, Nyl, Nxl], dtype=torch.float64)
  hr_ = torch.zeros([steps, 4, Ny,  Nx ], dtype=torch.float64)

  sgs = {}
  lf_ = {}
  lr_ = {}
  for m in les:
    sgs[m.name] = torch.zeros([steps, 1, Nyl, Nxl], dtype=torch.float64)
    lr_[m.name] = torch.zeros([steps, 4, Nyl, Nxl], dtype=torch.float64)
  time = torch.zeros([steps])

  def run_hr(m, cur, it):
    # high res
    if it % store == 0:
      i = int(it / store)
      q, p, u, v = m.update()

      for qte, fn in qtes.items():
        qts[qte][m.name].append(fn(m))

      # exact sgs
      r = m.R(sgs_grid, delta)

      dns[i, 0] = qg._p(r)
      dns[i, 1] = m.cutoff_physical(delta, sgs_grid, q).view(1, Nyl, Nxl)
      dns[i, 2] = m.cutoff_physical(delta, sgs_grid, p).view(1, Nyl, Nxl)
      dns[i, 3] = m.cutoff_physical(delta, sgs_grid, u).view(1, Nyl, Nxl)
      dns[i, 4] = m.cutoff_physical(delta, sgs_grid, v).view(1, Nyl, Nxl)
      hr_[i] = torch.stack((q, p, u, v))
      # step time
      time[i] = cur.t
    return None

  def run_lr(m, cur, it):
    # low res
    if it % store == 0:
      i = int(it / store)
      q, p, u, v = m.update()

      for qte, fn in qtes.items():
        qts[qte][m.name].append(fn(m))

      # predicted sgs
      if m.sgs_:
        r = m.sgs_.predict(
          m, 
          m.p_.sol, 
          cur.dt, 
          cur.t, 
          m.g_
        )
        sgs[m.name][i, 0] = qg._p(r)
      # model fields
      lr_[m.name][i] = torch.stack((q, p, u, v))
    return None
 
  dir = 'qg_workflows/'
  if not os.path.exists(dir + name):
    os.mkdir(dir + name)

  with torch.no_grad():
    for it in tqdm.tqdm(range(iters)):
      sys.p_.step(sys)
      run_hr(sys, sys.p_.cur, it)
      for m in les:
        m.p_.step(m)
        run_lr(m, m.p_.cur, it)

    for diag in diags:
      diag(
        dir,
        name,
        time,
        qts,
        sys,
        les,
        dns=dns,
        hr=hr_, 
        sgs=sgs,
        lr=lr_,
      )

    #sys.save(
    #  dir  + '/' + 
    #  name + '/' + 
    #  'q_end.h5'
    #)

def diag_show(dir, name, time, qts, sys, les, dns, hr, sgs, lr):
  # Plotting
  cols = 1
  rows = 4
  m_fig, m_axs = plt.subplots(
    nrows=rows,
    ncols=cols + 1,
    figsize=(cols * 2.5 + 0.5, rows * 2.5),
    constrained_layout=True,
    gridspec_kw={"width_ratios": np.append(np.repeat(rows, cols), 0.1)}
  )

  # DNS
  m_fig.colorbar(m_axs[0, 0].contourf(sys.g_.x.cpu().detach(), sys.g_.y.cpu().detach(), hr[-1, 0], cmap='bwr', levels=100), cax=m_axs[0, 1])
  m_fig.colorbar(m_axs[1, 0].contourf(sys.g_.x.cpu().detach(), sys.g_.y.cpu().detach(), hr[-1, 1], cmap='bwr', levels=100), cax=m_axs[1, 1])
  m_fig.colorbar(m_axs[2, 0].contourf(sys.g_.x.cpu().detach(), sys.g_.y.cpu().detach(), hr[-1, 2], cmap='bwr', levels=100), cax=m_axs[2, 1])
  m_fig.colorbar(m_axs[3, 0].contourf(sys.g_.x.cpu().detach(), sys.g_.y.cpu().detach(), hr[-1, 3], cmap='bwr', levels=100), cax=m_axs[3, 1])
  
  m_axs[0, 0].set_ylabel(r'$q$', fontsize=20)
  m_axs[1, 0].set_ylabel(r'$\psi$', fontsize=20)
  m_axs[2, 0].set_ylabel(r'$u_{x}$', fontsize=20)
  m_axs[3, 0].set_ylabel(r'$u_{y}$', fontsize=20)
  m_axs[3, 0].set_xlabel(r'$\mathcal{M}_{\mathrm{' + sys.name + '}}$', fontsize=20)

  m_fig.savefig(dir + name + '/' + name + '_dns.png')
  plt.close(m_fig)

  cols = len(lr) + 1
  rows = 5
  m_fig, m_axs = plt.subplots(
    nrows=rows,
    ncols=cols + 1,
    figsize=(cols * 2.5 + 0.5, rows * 2.5),
    constrained_layout=True,
    gridspec_kw={"width_ratios": np.append(np.repeat(rows, cols), 0.1)}
  )

  def plot_q(i, label, grid, data):
    c0 = m_axs[0, i].contourf(grid.x.cpu().detach(), grid.y.cpu().detach(), data[-1, 0], cmap='bwr', levels=100)
    c1 = m_axs[1, i].contourf(grid.x.cpu().detach(), grid.y.cpu().detach(), data[-1, 1], cmap='bwr', levels=100)
    c2 = m_axs[2, i].contourf(grid.x.cpu().detach(), grid.y.cpu().detach(), data[-1, 2], cmap='bwr', levels=100)
    c3 = m_axs[3, i].contourf(grid.x.cpu().detach(), grid.y.cpu().detach(), data[-1, 3], cmap='bwr', levels=100)
    if i == 0:
      m_fig.colorbar(c0, cax=m_axs[0, cols])
      m_fig.colorbar(c1, cax=m_axs[1, cols])
      m_fig.colorbar(c2, cax=m_axs[2, cols])
      m_fig.colorbar(c3, cax=m_axs[3, cols])

  def plot_s(i, label, grid, data):
    c4 = m_axs[4, i].contourf(grid.x.cpu().detach(), grid.y.cpu().detach(), data[-1], cmap='bwr', levels=100)
    if i == 0:
      m_fig.colorbar(c4, cax=m_axs[4, cols])

    m_axs[4, i].set_xlabel(label, fontsize=20)

  # Projected DNS
  plot_q(0, r'$\overline{\mathcal{M}_{\mathrm{' + sys.name + '}}}$', les[0].g_, dns[:, 1:])
  plot_s(0, r'$\overline{\mathcal{M}_{\mathrm{' + sys.name + '}}}$', les[0].g_, dns[:, 0])
  # LES
  for i, (model, mr) in enumerate(lr.items()):
    plot_q(
      1 + i,
      r'$\mathcal{M}_{' + model + '}$',
      les[i].g_,
      mr
    )
  for i, (model, sm) in enumerate(sgs.items()):
    plot_s(
      1 + i,
      r'$\mathcal{M}_{' + model + '}$',
      les[i].g_,
      sm[:, 0]
    )

  m_axs[0, 0].set_ylabel(r'$q$', fontsize=20)
  m_axs[1, 0].set_ylabel(r'$\psi$', fontsize=20)
  m_axs[2, 0].set_ylabel(r'$u_{x}$', fontsize=20)
  m_axs[3, 0].set_ylabel(r'$u_{y}$', fontsize=20)
  m_axs[4, 0].set_ylabel(r'$R(q)$', fontsize=20)

  m_fig.savefig(dir + name + '/' + name + '_quantities.png')
  plt.close(m_fig)

def mse(y, yhat, dim):
  return torch.pow(y - yhat, 2).mean(dim)

def cor(y, yhat, dim):
  return torch.abs(y - yhat).mean(dim)

def diag_sgs_metrics(dir, name, time, qts, sys, les, dns, hr, sgs, lr):
  # Plotting
  m_fig, m_axs = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(10,5),
    constrained_layout=True
  )

  def plot(i, label, grid, dns_data, les_data):
    m_axs[0].plot(time, mse(dns_data, les_data, (1, 2)), label=label)
    m_axs[1].plot(time, cor(dns_data, les_data, (1, 2)), label=label)

  # models
  for i, (model, sgm) in enumerate(sgs.items()):
    r_dns = dns[:, 0]
    r_sgs = sgm[:, 0]
    plot(
      1 + i,
      r'$\mathcal{M}_{' + model + '}$',
      les[i].g_,
      r_dns,
      r_sgs
    )

  m_axs[0].grid()
  m_axs[0].set_xlabel(r'$t$', fontsize=20)
  m_axs[0].set_ylabel(r'$\mathcal{L}_{\mathrm{rms}}(q)$', fontsize=20)
  m_axs[0].legend(fontsize=15)
  m_axs[1].grid()
  m_axs[1].set_xlabel(r'$t$', fontsize=20)
  m_axs[1].set_ylabel(r'$\mathcal{P}(q)$', fontsize=20)
  m_axs[1].legend(fontsize=15)

  m_fig.savefig(dir + name + '/' + name + '_sgs_metrics.pdf')
  plt.close(m_fig)

def moment_order(q, m, dim):
  return torch.mean(torch.pow(q - torch.mean(q, dim), m), dim)

def variance(q, dim):
  return moment_order(q, 2, dim)

def skewness(q, dim):
  return moment_order(q, 3, dim) / torch.pow(moment_order(q, 2, dim), 3.0/2.0)

def kurtosis(q, dim):
  return moment_order(q, 4, dim) / torch.pow(moment_order(q, 2, dim), 2.0) - 3.0

def diag_spatial_stats(dir, name, time, qts, sys, les, dns, hr, sgs, lr):
  # Plotting
  m_fig, m_axs = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(15, 5),
    constrained_layout=True
  )

  def plot(label, grid, data):
    m_axs[0].plot(time, variance(data.permute(1, 2, 0), (0, 1)), label=label)
    m_axs[1].plot(time, skewness(data.permute(1, 2, 0), (0, 1)), label=label)
    m_axs[2].plot(time, kurtosis(data.permute(1, 2, 0), (0, 1)), label=label)

  # DNS
  q = hr[:, 0]
  plot(r'$\mathcal{M}_{\mathrm{' + sys.name + '}}$', sys.g_, q)
  # Projected DNS
  q = dns[:, 1]
  plot(r'$\overline{\mathcal{M}_{\mathrm{' + sys.name + '}}}$', les[0].g_, q)
  # LES
  for i, (model, mr) in enumerate(lr.items()):
    q = mr[:, 0]
    plot(
      r'$\mathcal{M}_{' + model + '}$',
      les[i].g_,
      q
    )

  m_axs[0].grid()
  m_axs[0].set_xlabel(r'$t$', fontsize=20)
  m_axs[0].set_ylabel(r'$v\langle q \rangle_{\Omega}$', fontsize=20)
  m_axs[0].legend(fontsize=15)
  m_axs[1].grid()
  m_axs[1].set_xlabel(r'$t$', fontsize=20)
  m_axs[1].set_ylabel(r'$s\langle q \rangle_{\Omega}$', fontsize=20)
  m_axs[1].legend(fontsize=15)
  m_axs[2].grid()
  m_axs[2].set_xlabel(r'$t$', fontsize=20)
  m_axs[2].set_ylabel(r'$k\langle q \rangle_{\Omega}$', fontsize=20)
  m_axs[2].legend(fontsize=15)

  m_fig.savefig(dir + name + '/' + name + '_spatial_stats.pdf')
  plt.close(m_fig)

def diag_temporal_stats(dir, name, time, qts, sys, les, dns, hr, sgs, lr):
  # Plotting
  cols = len(lr) + 2
  rows = 3
  m_fig, m_axs = plt.subplots(
    nrows=rows,
    ncols=cols + 1,
    figsize=(cols * 2.5 + 0.5, rows * 2.5),
    constrained_layout=True,
    gridspec_kw={"width_ratios": np.append(np.repeat(rows, cols), 0.1)}
  )

  def plot(i, label, grid, data):
    c0 = m_axs[0, i].contourf(grid.x.cpu().detach(), grid.y.cpu().detach(), variance(data, 0), cmap='bwr', levels=100)
    c1 = m_axs[1, i].contourf(grid.x.cpu().detach(), grid.y.cpu().detach(), skewness(data, 0), cmap='bwr', levels=100)
    c2 = m_axs[2, i].contourf(grid.x.cpu().detach(), grid.y.cpu().detach(), kurtosis(data, 0), cmap='bwr', levels=100)
    
    if i == 0:
      m_fig.colorbar(c0, cax=m_axs[0, cols])
      m_fig.colorbar(c1, cax=m_axs[1, cols])
      m_fig.colorbar(c2, cax=m_axs[2, cols])

    m_axs[2, i].set_xlabel(label, fontsize=20)

  # DNS
  q = hr[:, 0]
  plot(0, r'$\mathcal{M}_{\mathrm{' + sys.name + '}}$', sys.g_, q)
  # Projected DNS
  q = dns[:, 1]
  plot(1, r'$\overline{\mathcal{M}_{\mathrm{' + sys.name + '}}}$', les[0].g_, q)
  # LES
  for i, (model, mr) in enumerate(lr.items()):
    q = mr[:, 0]
    plot(
      2 + i,
      r'$\mathcal{M}_{' + model + '}$',
      les[i].g_,
      q
    )

  m_axs[0, 0].set_ylabel(r'$v\langle q \rangle_{t}$', fontsize=20)
  m_axs[1, 0].set_ylabel(r'$s\langle q \rangle_{t}$', fontsize=20)
  m_axs[2, 0].set_ylabel(r'$k\langle q \rangle_{t}$', fontsize=20)

  m_fig.savefig(dir + name + '/' + name + '_temporal_stats.png')
  plt.close(m_fig)

def diag_integrals(dir, name, time, qts, sys, les, dns, hr, sgs, lr):
  m_fig, m_axs = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(10,5),
    constrained_layout=True
  )

  samples = len(time)

  # DNS
  q = hr[:, 0]
  p = hr[:, 1]
  u = hr[:, 2]
  v = hr[:, 3]
  m_axs[0].plot(time, [sys.energy   (u[i], v[i]).item() for i in range(samples)], label='$\mathcal{M}_{' + sys.name + '}$')
  m_axs[1].plot(time, [sys.enstrophy(q[i]      ).item() for i in range(samples)], label='$\mathcal{M}_{' + sys.name + '}$')
  # Projected DNS
  q = dns[:, 1]
  p = dns[:, 2]
  u = dns[:, 3]
  v = dns[:, 4]
  m_axs[0].plot(time, [les[0].energy   (u[i], v[i]).item() for i in range(samples)], label='$\overline{\mathcal{M}_{' + sys.name + '}}$')
  m_axs[1].plot(time, [les[0].enstrophy(q[i]      ).item() for i in range(samples)], label='$\overline{\mathcal{M}_{' + sys.name + '}}$')
  # LES
  for m in les:
    l = lr[m.name]
    q = l[:, 0]
    p = l[:, 1]
    u = l[:, 2]
    v = l[:, 3]
    ene = [m.energy   (u[i], v[i]).item() for i in range(samples)]
    enk = [m.enstrophy(q[i]      ).item() for i in range(samples)]
    m_axs[0].plot(time, ene, label=r'$\mathcal{M}_{' + m.name + '}$')
    m_axs[1].plot(time, enk, label=r'$\mathcal{M}_{' + m.name + '}$')

  m_axs[0].grid()
  m_axs[0].set_xlabel(r'$t$', fontsize=20)
  m_axs[0].set_ylabel(r'$\frac{1}{2} \int \, \mathbf{u}^2 \,\, \mathrm{d}r$', fontsize=20)
  m_axs[0].legend(fontsize=15)
  m_axs[1].grid()
  m_axs[1].set_xlabel(r'$t$', fontsize=20)
  m_axs[1].set_ylabel(r'$\frac{1}{2} \int \, q^{2} \,\, \mathrm{d}r$', fontsize=20)
  m_axs[1].legend(fontsize=15)

  m_fig.savefig(dir + name + '/' + name + '_integrals.pdf')
  plt.close(m_fig)

def diag_spectra(dir, name, time, qts, sys, les, dns, hr, sgs, lr):
  m_fig, m_axs = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(10,5),
    constrained_layout=True
  )

  k, ek, en = sys.spectrum()

  # Theory
  m_axs[0].loglog(k[:-1], k[:-1].to(dtype=torch.float64)**(-3), label=r'$\sim k^{-3}$', linestyle='-.', color='k')
  m_axs[1].loglog(k[:-1], k[:-1].to(dtype=torch.float64)**(-1), label=r'$\sim k^{-1}$', linestyle='-.', color='k')
  # DNS
  m_axs[0].loglog(k[:-1], ek[:-1], label=r'$\mathcal{M}_{' + sys.name + '}$', linestyle='--')
  m_axs[1].loglog(k[:-1], en[:-1], label=r'$\mathcal{M}_{' + sys.name + '}$', linestyle='--')
  # LES
  for m in les:
    k, ek, en = m.spectrum()
    m_axs[0].loglog(k[:-1], ek[:-1], label=r'$\mathcal{M}_{' + m.name + '}$')
    m_axs[1].loglog(k[:-1], en[:-1], label=r'$\mathcal{M}_{' + m.name + '}$')

  m_axs[0].grid()
  m_axs[0].set_xlabel(r'$k$', fontsize=20)
  m_axs[0].set_ylabel(r'$E(k)$', fontsize=20)
  m_axs[0].set_ylim(bottom=1e-10)
  m_axs[0].legend(fontsize=15)
  m_axs[1].grid()
  m_axs[1].set_xlabel(r'$k$', fontsize=20)
  m_axs[1].set_ylabel(r'$Z(k)$', fontsize=20)
  m_axs[1].set_ylim(bottom=1e-5)
  m_axs[1].legend(fontsize=15)

  m_fig.savefig(dir + name + '/' + name + '_spectra.pdf')
  plt.close(m_fig)
