import os

import torch
import numpy as np

import qg

# Useful for a posteriori learning.
class DynamicalDataset(torch.utils.data.Dataset):
  def __init__(self, inputs, labels, steps, iters, dt, t0):
    self.inputs = inputs
    self.labels = labels
    self.iters = iters
    self.dt = dt
    self.t0 = t0
    self.adapt(steps)

  def __len__(self):
    return int(self.inputs.shape[0] * self.samples)

  def __getitem__(self, idx):
    tra = int(idx / self.samples)
    idx = int(idx % self.samples)

    it0 = idx * self.steps
    itn = (idx + 1) * self.steps
    t = it0 * self.dt

    inputs = self.inputs[tra, it0:itn + 1]
    labels = self.labels[tra, it0:itn + 1]
    return (self.t0 + t, inputs, labels)

  def adapt(self, steps):
    self.steps = steps
    self.samples = int(self.iters / self.steps) - 1

def training(device, net, dataloader, loss, opti, rate, stat):
  net.train()
  cost = 0.0
  for step, batch in enumerate(dataloader):
    opti.zero_grad()
    data, labs = batch[0].to(device), batch[1].to(device)

    q = data[:,0]
    p = data[:,1]

    pred = net(torch.stack((q, p), dim=1))
    grad = loss(data, data, pred, labs)

    grad.backward()
    opti.step()
    rate.step()

    cost  += grad.item()

  cost /= len(dataloader)
  stat.append(cost)

def valididation(device, net, dataloader, loss, stat):
  net.eval()
  cost = 0.0
  with torch.no_grad():
    for step, batch in enumerate(dataloader):
      data, labs = batch[0].to(device), batch[1].to(device)

      q = data[:,0]
      p = data[:,1]

      pred = net(torch.stack((q, p), dim=1))
      grad = loss(data, data, pred, labs)
      cost += grad.item()

    cost /= len(dataloader)
    stat.append(cost)

# A priori learning strategy
def apriori(device, dir, net, train_loader, valid_loader, loss, opti, rate, epochs=1000):
  if not os.path.exists(dir + net.name):
    os.mkdir(dir + net.name)

  train_loss = []
  valid_loss = []

  for epoch in range(1, epochs + 1):
    train(device, net, train_loader, loss, opti, rate, train_loss)
    valid(device, net, valid_loader, loss, valid_loss)
    
    if epoch % 1 == 0:
      print('Epoch {} (training loss = {}, validation loss = {})'.format(epoch, train_loss[-1], valid_loss[-1]), flush=True)
    if epoch % 10 == 0:
      np.savetxt(dir + net.name + '/losses.csv', np.column_stack((train_loss, valid_loss)), delimiter=",", fmt='%s')
      torch.save(net, dir + net.name + '/weights.pyt')
  
  print('Finished training, with last progress loss = {}'.format(train_loss[-1]))

# A posteriori learning strategy
def aposteriori(device, dir, net, dyn, iters, dataloader, loss, opti, rate, epochs=5, epochs_full=2):
  if not os.path.exists(dir + net.name):
    os.mkdir(dir + net.name)

  notify_freq = int(len(dataloader) / 10)

  time_loss = []
  temp_loss = 0
  temp_cnt  = 0

  def timestep(m, cur, it):
    q, p, u, v = m.update()

    states_i[it, 0] = q
    states_i[it, 1] = p
    states_i[it, 2] = u
    states_i[it, 3] = v

    # Predict SGS from NN
    r = net(torch.stack((q, p), dim=0).unsqueeze(0).to(torch.float32)).squeeze(0)
    states_o[it, 0] = r[0]
    return None

  ck = int(iters / epochs)
  net.train()
  for epoch in range(1, epochs + epochs_full + 1):
    it = max(1, min(iters, ck * epoch))
    dataloader.dataset.adapt(it)

    states_i = torch.zeros([it, 4, dyn.grid.Ny, dyn.grid.Nx], requires_grad=True).to(device)
    states_o = torch.zeros([it, 2, dyn.grid.Ny, dyn.grid.Nx], requires_grad=True).to(device)

    for step, batch in enumerate(dataloader):

      states_i.detach_()
      states_o.detach_()
      opti.zero_grad()
      dyn.zero_grad()

      t, data, labs = batch[0], batch[1].squeeze(0).to(device), batch[2].squeeze(0).to(device)

      # Start from DNS
      # bar(q)(t) = bar(q(t))
      dyn.pde.sol = qg.to_spectral(data[0, 0])
      dyn.pde.cur.t = t
      # Run dynamical model
      # bar(q)(t + ndt)
      dyn.run(it, timestep, invisible=True)

      if dyn.cfl() < 1:
        # Compute loss
        grad = loss(states_i, data[1:it+1], states_o, labs[1:it+1])

        grad.backward()
        opti.step()
        rate.step()

        temp_loss += grad.item() / it
        temp_cnt  += 1

      # No validation yet
 
      if step % notify_freq == 0:
        time_loss.append(temp_loss / temp_cnt)
        temp_loss = 0
        temp_cnt  = 0
        print('Epoch {} with {} iters (step {}, loss = {})'.format(epoch, it, step, time_loss[-1]), flush=True)

    if epoch % 1 == 0:
      np.savetxt(dir + net.name + '/losses.csv', time_loss, delimiter=",", fmt='%s')
      torch.save(net, dir + net.name + '/weights.pyt')

  print('Finished training, with last progress loss = {}'.format(time_loss[-1]))
