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
  system,
  models,
  dump=False,
):
  t0 = system.pde.cur.t
  store_les = int(iters / steps)
  store_dns = store_les * scale

  Nx = system.grid.Nx
  Ny = system.grid.Ny
  Nxl = int(Nx / scale)
  Nyl = int(Ny / scale)

  if models:
    sgs_grid = models[-1].grid

  # Filtered DNS
  fdns = torch.zeros([steps, 5, Nyl, Nxl], dtype=torch.float64)
  # DNS
  dns  = torch.zeros([steps, 4, Ny,  Nx ], dtype=torch.float64)

  # LES
  les = {}
  for m in models:
    les[m.name] = torch.zeros([steps, 5, Nyl, Nxl], dtype=torch.float64)
  time = torch.zeros([steps])

  def visitor_dns(m, cur, it):
    # High res
    if it % store_dns == 0:
      i = int(it / store_dns)
      q, p, u, v = m.update()

      # Exact sgs
      if models:
        r = m.R(sgs_grid, scale)
        fdns[i, 0] = qg.to_physical(r)
        fdns[i, 1] = m.filter_physical(sgs_grid, scale, q).view(1, Nyl, Nxl)
        fdns[i, 2] = m.filter_physical(sgs_grid, scale, p).view(1, Nyl, Nxl)
        fdns[i, 3] = m.filter_physical(sgs_grid, scale, u).view(1, Nyl, Nxl)
        fdns[i, 4] = m.filter_physical(sgs_grid, scale, v).view(1, Nyl, Nxl)
      dns[i] = torch.stack((q, p, u, v))

      # step time
      time[i] = cur.t - t0
    return None

  def visitor_les(m, cur, it):
    # Low res
    if it % store_les == 0:
      i = int(it / store_les)
      q, p, u, v = m.update()

      # Predicted sgs
      if m.sgs:
        r = m.sgs.predict(m, 0, m.pde.sol, m.grid)
      else:
        r = torch.zeros([Nyl, Nxl], dtype=torch.float64)
      les[m.name][i] = torch.stack((qg.to_physical(r), q, p, u, v))
    return None
 
  if not os.path.exists(dir):
    os.mkdir(dir)

  with torch.no_grad():
    for it in tqdm.tqdm(range(iters * scale)):
      system.pde.step(system)
      visitor_dns(system, system.pde.cur, it)
      for m in models:
        if it % scale == 0:
          m.pde.step(m)
          visitor_les(m, m.pde.cur, it / scale)

    for diag in diags:
      diag(
        dir, 
        name, 
        scale, 
        time, 
        system, 
        models, 
        dns=dns, 
        fdns=fdns, 
        les=les
      )
    if dump:
      hf = h5py.File(os.path.join(dir, name + '_dump.h5'), 'w')
      hf.create_dataset('time', data=time.detach().numpy())
      hf.create_dataset(system.name + '_r', data=fdns[:, 0].detach().numpy())
      hf.create_dataset(system.name + '_q', data=fdns[:, 1].detach().numpy())
      hf.create_dataset(system.name + '_p', data=fdns[:, 2].detach().numpy())
      hf.create_dataset(system.name + '_u', data=fdns[:, 3].detach().numpy())
      hf.create_dataset(system.name + '_v', data=fdns[:, 4].detach().numpy())
      for m in models:
        hf.create_dataset(m.name + '_r', data=les[m.name][:, 0].detach().numpy())
        hf.create_dataset(m.name + '_q', data=les[m.name][:, 1].detach().numpy())
        hf.create_dataset(m.name + '_p', data=les[m.name][:, 2].detach().numpy())
        hf.create_dataset(m.name + '_u', data=les[m.name][:, 3].detach().numpy())
        hf.create_dataset(m.name + '_v', data=les[m.name][:, 4].detach().numpy())
      hf.close()

def diag_fields(dir, name, scale, time, system, models, dns, fdns, les):
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
  m_fig.colorbar(m_axs[0, 0].contourf(system.grid.x.cpu().detach(), system.grid.y.cpu().detach(), dns[-1, 0], cmap='bwr', levels=100), cax=m_axs[0, 1])
  m_fig.colorbar(m_axs[1, 0].contourf(system.grid.x.cpu().detach(), system.grid.y.cpu().detach(), dns[-1, 1], cmap='bwr', levels=100), cax=m_axs[1, 1])
  m_fig.colorbar(m_axs[2, 0].contourf(system.grid.x.cpu().detach(), system.grid.y.cpu().detach(), dns[-1, 2], cmap='bwr', levels=100), cax=m_axs[2, 1])
  m_fig.colorbar(m_axs[3, 0].contourf(system.grid.x.cpu().detach(), system.grid.y.cpu().detach(), dns[-1, 3], cmap='bwr', levels=100), cax=m_axs[3, 1])
  
  m_axs[0, 0].set_ylabel(r'$\omega$', fontsize=20)
  m_axs[1, 0].set_ylabel(r'$\psi$', fontsize=20)
  m_axs[2, 0].set_ylabel(r'$u_{x}$', fontsize=20)
  m_axs[3, 0].set_ylabel(r'$u_{y}$', fontsize=20)
  m_axs[3, 0].set_xlabel(r'$\mathcal{M}' + system.name + '$', fontsize=20)

  m_fig.savefig(os.path.join(dir, name + '_dns.png'), dpi=300)
  plt.show()
  plt.close(m_fig)

  if not models:
    return

  cols = len(models) + 1
  rows = 5
  m_fig, m_axs = plt.subplots(
    nrows=rows,
    ncols=cols + 1,
    figsize=(cols * 2.5 + 0.5, rows * 2.5),
    constrained_layout=True,
    gridspec_kw={"width_ratios": np.append(np.repeat(rows, cols), 0.1)}
  )

  span_r = max(fdns[-1, 0].max(), abs(fdns[-1, 0].min()))
  span_q = max(fdns[-1, 1].max(), abs(fdns[-1, 1].min()))
  span_p = max(fdns[-1, 2].max(), abs(fdns[-1, 2].min()))
  span_u = max(fdns[-1, 3].max(), abs(fdns[-1, 3].min()))
  span_v = max(fdns[-1, 4].max(), abs(fdns[-1, 4].min()))

  def plot_fields(i, label, grid, data):
    c0 = m_axs[0, i].contourf(grid.x.cpu().detach(), grid.y.cpu().detach(), data[-1, 1], vmax=span_q, vmin=-span_q, cmap='bwr', levels=100)
    c1 = m_axs[1, i].contourf(grid.x.cpu().detach(), grid.y.cpu().detach(), data[-1, 2], vmax=span_p, vmin=-span_p, cmap='bwr', levels=100)
    c2 = m_axs[2, i].contourf(grid.x.cpu().detach(), grid.y.cpu().detach(), data[-1, 3], vmax=span_u, vmin=-span_u, cmap='bwr', levels=100)
    c3 = m_axs[3, i].contourf(grid.x.cpu().detach(), grid.y.cpu().detach(), data[-1, 4], vmax=span_v, vmin=-span_v, cmap='bwr', levels=100)
    c4 = m_axs[4, i].contourf(grid.x.cpu().detach(), grid.y.cpu().detach(), data[-1, 0], vmax=span_r, vmin=-span_r, cmap='bwr', levels=100)
    if i == 0:
      m_fig.colorbar(c0, cax=m_axs[0, cols])
      m_fig.colorbar(c1, cax=m_axs[1, cols])
      m_fig.colorbar(c2, cax=m_axs[2, cols])
      m_fig.colorbar(c3, cax=m_axs[3, cols])
      m_fig.colorbar(c4, cax=m_axs[4, cols])
    m_axs[4, i].set_xlabel(label, fontsize=20)

  # Projected DNS
  plot_fields(0, r'$\overline{\mathcal{M}' + system.name + '}$', models[-1].grid, fdns)
  # LES
  for i, m in enumerate(models):
    data = les[m.name]
    plot_fields(i + 1, r'$\mathcal{M}_{' + m.name + '}$', m.grid, data)

  m_axs[0, 0].set_ylabel(r'$\omega$', fontsize=20)
  m_axs[1, 0].set_ylabel(r'$\psi$', fontsize=20)
  m_axs[2, 0].set_ylabel(r'$u_{x}$', fontsize=20)
  m_axs[3, 0].set_ylabel(r'$u_{y}$', fontsize=20)
  m_axs[4, 0].set_ylabel(r'$R(q)$', fontsize=20)

  m_fig.savefig(os.path.join(dir, name + '_fields.png'), dpi=300)
  plt.show()
  plt.close(m_fig)

