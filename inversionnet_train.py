from func.datasets_reader import batch_read_npyfile
from net.InversionNet import InversionNet
from func.utils import model_reader
from path_config import *
from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import time

train_or_test = "train"
device_ids = [0]
device = torch.device("cuda")
LearnRate = 0.0001
Epochs = 120
TrainSize = 48000
BatchSize = 128

external_model_src = r""
InvNet = InversionNet()               # 申请网络

if external_model_src is not "":
    InvNet = model_reader(net=InvNet, device=device, save_src=external_model_src)

if torch.cuda.is_available():
    InvNet = torch.nn.DataParallel(InvNet, device_ids=device_ids).cuda()

optimizer = torch.optim.Adam(InvNet.parameters(), lr = LearnRate)
optimizer.zero_grad()

print("---------------------------------")
print("· Loading the datasets...")

data_set, label_sets = batch_read_npyfile(data_dir, 1, ceil(TrainSize / 500), "train")

print("Normalization in progress...")

for i in range(data_set.shape[0]):
    vm = label_sets[0][i][0]
    label_sets[0][i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

seis_and_vm = data_utils.TensorDataset(torch.from_numpy(data_set).float(),
                                       torch.from_numpy(label_sets[0]).float())
seis_and_vm_loader = data_utils.DataLoader(seis_and_vm, batch_size=BatchSize, shuffle=True)

print('Epoch: {}'.format(Epochs))
print('Lr: {}'.format(LearnRate))

prefix= '(InversionNet)'
tagM1 = '_TrainSize' + str(TrainSize)
tagM2 = '_Epoch' + str(Epochs)
tagM3 = '_BatchSize' + str(BatchSize)
modelname = prefix + dataset_name + tagM1 + tagM2 + tagM3

loss_of_stage = 0.0
step = int(TrainSize / BatchSize)
start = time.time()
save_times = 12
save_epoch = Epochs // save_times

for epoch in range(Epochs):
    cur_epoch_loss = 0.0
    cur_node_time = time.time()
    for i, (images, labels) in enumerate(seis_and_vm_loader):
        iteration = epoch * step + i + 1

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        optimizer.zero_grad()
        InvNet.train()
        outputs = InvNet(images)

        criterion = nn.MSELoss()
        loss = criterion(outputs, labels)

        if np.isnan(float(loss.item())):
            raise ValueError('loss is nan while training')

        cur_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        if iteration % 2 == 0:
            print('Epochs: {}/{}, Iteration: {}/{} --- Training Loss:{:.6f} --- Learning Rate:{:.12f}'
                  .format(epoch + 1, Epochs, iteration, step * Epochs, loss.item(), optimizer.param_groups[0]['lr']))

    if (epoch + 1) % 1 == 0:

        time_elapsed = time.time() - cur_node_time
        print('[InversionNet] Epochs consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if (epoch + 1) % save_epoch == 0:
        last_model_save_path = models_dir + 'InversionNet' + '_CurEpo' + str(epoch + 1) + '.pkl'
        torch.save(InvNet.state_dict(), last_model_save_path)
        print('[InversionNet] Trained model saved: %d percent completed' % int((epoch + 1) * 100 / Epochs))

print('Finish!')