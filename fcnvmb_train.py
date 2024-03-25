from func.datasets_reader import batch_read_matfile
from net.FCNVMB import FCNVMB
from func.utils import model_reader
from path_config import *

import numpy as np
import torch
import torch.utils.data as data_utils
import torch.nn.functional as F
import time

train_or_test = "train"
device_ids = [0]
device = torch.device("cuda")
LearnRate = 0.001
Epochs = 100
TrainSize = 1600
BatchSize = 10

external_model_src = r""
fcnNet = FCNVMB(n_classes=classes, in_channels=inchannels, is_deconv=True, is_batchnorm=True)

if external_model_src is not "":
    fcnNet = model_reader(net=fcnNet, device=device, save_src=external_model_src)

if torch.cuda.is_available():
    fcnNet = torch.nn.DataParallel(fcnNet, device_ids=device_ids).cuda()

optimizer = torch.optim.Adam(fcnNet.parameters(), lr = LearnRate)
optimizer.zero_grad()

print("---------------------------------")
print("· Loading the datasets...")

data_set, label_sets = batch_read_matfile(data_dir, 1, TrainSize, "train")


seis_and_vm = data_utils.TensorDataset(torch.from_numpy(data_set).float(),
                                       torch.from_numpy(label_sets[0]).float())
seis_and_vm_loader = data_utils.DataLoader(seis_and_vm, batch_size=BatchSize, shuffle=True)

print('Epoch: {}'.format(Epochs))
print('Lr: {}'.format(LearnRate))

prefix= '(FCNVMB)'
tagM1 = '_TrainSize' + str(TrainSize)
tagM2 = '_Epoch' + str(Epochs)
tagM3 = '_BatchSize' + str(BatchSize)
modelname = prefix + dataset_name + tagM1 + tagM2 + tagM3

loss_of_stage = 0.0
step = int(TrainSize / BatchSize)
start = time.time()
save_times = 2
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

        fcnNet.train()
        outputs = fcnNet(images)

        loss = F.mse_loss(outputs, labels, reduction='sum') / (model_dim[0] * model_dim[1] * BatchSize)

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
        print('[FCNVMB] Epochs consuming time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if (epoch + 1) % save_epoch == 0:
        last_model_save_path = models_dir + 'FCNVMB' + '_CurEpo' + str(epoch + 1) + '.pkl'
        torch.save(fcnNet.state_dict(), last_model_save_path)
        print('[FCNVMB] Trained model saved: %d percent completed' % int((epoch + 1) * 100 / Epochs))

print('Finish!')