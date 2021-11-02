import math
import os
import tqdm

import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import qg

plt.rcParams.update({'mathtext.fontset':'cm'})
plt.rcParams.update({'xtick.minor.visible':True})
plt.rcParams.update({'ytick.minor.visible':True})

def workflow(
  dir,
  name,
  iters, 
  steps,
  scale,
  diags,
  qtes,
  sys,
  les,
  verbose=False,
  save=False,
  dump=False,
):
  t0 = sys.p_.cur.t
  store = int(iters / steps)
  truth = store * scale

  Nx = sys.g_.Nx
  Ny = sys.g_.Ny
  Nxl = int(Nx / scale)
  Nyl = int(Ny / scale)

  sgs_grid = les[-1].g_

  qts = {}
  for qte in qtes:
    qts[qte] = {}
    qts[qte][sys.name] = []
    for model in les:
      qts[qte][model.name] = []

  dns = torch.zeros([steps, 5, Nyl, Nxl], dtype=torch.float64)
  hr_ = torch.zeros([steps, 4, Ny,  Nx ], dtype=torch.float64)

  sgs = {}
  lr_ = {}
  for m in les:
    sgs[m.name] = torch.zeros([steps, 1, Nyl, Nxl], dtype=torch.float64)
    lr_[m.name] = torch.zeros([steps, 4, Nyl, Nxl], dtype=torch.float64)
  time = torch.zeros([steps])

  def run_hr(m, cur, it):
    # high res
    if it % truth == 0:
      i = int(it / truth)
      q, p, u, v = m.update()

      for qte, fn in qtes.items():
        qts[qte][m.name].append(fn(m))

      # exact sgs
      r = m.R(sgs_grid, scale)

      if verbose:
        print('eddy turnover time:', m.eddy_time(L=1))
        print('cfl:', m.cfl())
        print('vrms:', m.vrms())
        print('re:', m.re(L=1))
      if dump:
        hf = h5py.File(dir + name + '/' + name + '_dump_{}.h5'.format(str(it).zfill(8)), 'w')
        hf.create_dataset('q', data=q.cpu().detach())
        hf.close() 
 
      dns[i, 0] = qg._p(r)
      dns[i, 1] = m.filter_physical(sgs_grid, scale, q).view(1, Nyl, Nxl)
      dns[i, 2] = m.filter_physical(sgs_grid, scale, p).view(1, Nyl, Nxl)
      dns[i, 3] = m.filter_physical(sgs_grid, scale, u).view(1, Nyl, Nxl)
      dns[i, 4] = m.filter_physical(sgs_grid, scale, v).view(1, Nyl, Nxl)
      hr_[i] = torch.stack((q, p, u, v))

      # step time
      time[i] = cur.t - t0
    return None

  def run_lr(m, cur, it):
    # low res
    if it % store == 0:
      i = int(it / store)
      q, p, u, v = m.update()

      for qte, fn in qtes.items():
        qts[qte][m.name].append(fn(m))

      if verbose:
        print('eddy turnover time:', m.eddy_time(L=1))
        print('cfl:', m.cfl())
        print('vrms:', m.vrms())
        print('re:', m.re(L=1))

      # predicted sgs
      if m.sgs_:
        r = m.sgs_.predict(
          m, 
          0,
          m.p_.sol, 
          m.g_
        )
        sgs[m.name][i, 0] = qg._p(r)
        # model fields
        lr_[m.name][i] = torch.stack((q, p, u, v))
      else:
        r = m.R(sgs_grid, scale)
        sgs[m.name][i, 0] = qg._p(r)
        # filter fields
        lr_[m.name][i, 0] = m.filter_physical(sgs_grid, scale, q).view(1, Nyl, Nxl)
        lr_[m.name][i, 1] = m.filter_physical(sgs_grid, scale, p).view(1, Nyl, Nxl)
        lr_[m.name][i, 2] = m.filter_physical(sgs_grid, scale, u).view(1, Nyl, Nxl)
        lr_[m.name][i, 3] = m.filter_physical(sgs_grid, scale, v).view(1, Nyl, Nxl)
    return None
 
  if not os.path.exists(dir + name):
    os.mkdir(dir + name)

  with torch.no_grad():
    for it in tqdm.tqdm(range(iters * scale)):
      sys.p_.step(sys)
      run_hr(sys, sys.p_.cur, it)
      for m in les:
        if it % scale == 0:
          m.p_.step(m)
          run_lr(m, m.p_.cur, it / scale)

    for diag in diags:
      diag(
        dir,
        name,
        scale,
        time,
        qts,
        sys,
        les,
        dns=dns,
        hr=hr_, 
        sgs=sgs,
        lr=lr_,
      )

    if save:
      sys.save(
        dir  + '/' + 
        name + '/' + 
        'q_end.h5'
      )

def diag_show(dir, name, scale, time, qts, sys, les, dns, hr, sgs, lr):
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
  
  m_axs[0, 0].set_ylabel(r'$\omega$', fontsize=20)
  m_axs[1, 0].set_ylabel(r'$\psi$', fontsize=20)
  m_axs[2, 0].set_ylabel(r'$u_{x}$', fontsize=20)
  m_axs[3, 0].set_ylabel(r'$u_{y}$', fontsize=20)
  m_axs[3, 0].set_xlabel(r'$\mathcal{M}' + sys.name + '$', fontsize=20)

  m_fig.savefig(dir + name + '/' + name + '_dns.png')
  plt.show()
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

  span_r = max(dns[-1, 0].max(), abs(dns[-1, 0].min()))
  span_q = max(dns[-1, 1].max(), abs(dns[-1, 1].min()))
  span_p = max(dns[-1, 2].max(), abs(dns[-1, 2].min()))
  span_u = max(dns[-1, 3].max(), abs(dns[-1, 3].min()))
  span_v = max(dns[-1, 4].max(), abs(dns[-1, 4].min()))

  def plot_q(i, label, grid, data):
    c0 = m_axs[0, i].contourf(grid.x.cpu().detach(), grid.y.cpu().detach(), data[-1, 0], vmax=span_q, vmin=-span_q, cmap='bwr', levels=100)
    c1 = m_axs[1, i].contourf(grid.x.cpu().detach(), grid.y.cpu().detach(), data[-1, 1], vmax=span_p, vmin=-span_p, cmap='bwr', levels=100)
    c2 = m_axs[2, i].contourf(grid.x.cpu().detach(), grid.y.cpu().detach(), data[-1, 2], vmax=span_u, vmin=-span_u, cmap='bwr', levels=100)
    c3 = m_axs[3, i].contourf(grid.x.cpu().detach(), grid.y.cpu().detach(), data[-1, 3], vmax=span_v, vmin=-span_v, cmap='bwr', levels=100)
    if i == 0:
      m_fig.colorbar(c0, cax=m_axs[0, cols])
      m_fig.colorbar(c1, cax=m_axs[1, cols])
      m_fig.colorbar(c2, cax=m_axs[2, cols])
      m_fig.colorbar(c3, cax=m_axs[3, cols])

  def plot_s(i, label, grid, data):
    c4 = m_axs[4, i].contourf(grid.x.cpu().detach(), grid.y.cpu().detach(), data[-1], vmax=span_r, vmin=-span_r, cmap='bwr', levels=100)
    if i == 0:
      m_fig.colorbar(c4, cax=m_axs[4, cols])

    m_axs[4, i].set_xlabel(label, fontsize=20)

  # Projected DNS
  plot_q(0, r'$\overline{\mathcal{M}' + sys.name + '}$', les[-1].g_, dns[:, 1:])
  plot_s(0, r'$\overline{\mathcal{M}' + sys.name + '}$', les[-1].g_, dns[:, 0])
  # LES
  for i, (model, mr) in enumerate(lr.items()):
    plot_q(
      1 + i,
      r'$\mathcal{M}_{' + model + '}$',
      les[-1].g_,
      mr
    )
  for i, (model, sm) in enumerate(sgs.items()):
    plot_s(
      1 + i,
      r'$\mathcal{M}_{' + model + '}$',
      les[-1].g_,
      sm[:, 0]
    )

  m_axs[0, 0].set_ylabel(r'$\omega$', fontsize=20)
  m_axs[1, 0].set_ylabel(r'$\psi$', fontsize=20)
  m_axs[2, 0].set_ylabel(r'$u_{x}$', fontsize=20)
  m_axs[3, 0].set_ylabel(r'$u_{y}$', fontsize=20)
  m_axs[4, 0].set_ylabel(r'$R(q)$', fontsize=20)

  m_fig.savefig(dir + name + '/' + name + '_quantities.png')
  plt.show()
  plt.close(m_fig)

