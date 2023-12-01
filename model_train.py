# -*- coding: utf-8 -*-
"""
Curriculum Learning

Created on Feb 2023

@author: Xing-Yi Zhang (Zhangzxy20004182@163.com)

"""
from path_config import *
from func.utils import model_reader, add_gasuss_noise, magnify_amplitude_fornumpy
from func.datasets_reader import batch_read_matfile, batch_read_npyfile
from net.InversionNet import InversionNet
from net.FCNVMB import FCNVMB
from net.DDNet70 import DDNet70Model, SDNet70Model, LossDDNet
from net.DDNet import DDNetModel, SDNetModel
from math import ceil

import time
import numpy as np
import torch
import torch.utils.data as data_utils
import gc
import torch.nn.functional as F

def determine_network(external_model_src="", model_type="DDNet"):
    '''
    Request a network object and import an external network, or create an initialized network

    :param external_model_src:  External pkl file path
    :param model_type:          The main model used, this model is differentiated based on different papers.
                                The available key model keywords are
                                [DDNet | DDNet70 | InversionNet | FCNVMB | SDNet | SDNet70]
    :return:                    A triplet: model object, GPU environment object and optimizer
    '''

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    gpus = [0]

    # Network initialization
    if model_type == "DDNet":
        net_model = DDNetModel(n_classes=classes,
                               in_channels=inchannels,
                               is_deconv=True,
                               is_batchnorm=True)
    elif model_type == "DDNet70":
        net_model = DDNet70Model(n_classes=classes,
                                 in_channels=inchannels,
                                 is_deconv=True,
                                 is_batchnorm=True)
    elif model_type == "SDNet":
        net_model = SDNetModel(n_classes=classes,
                               in_channels=inchannels,
                               is_deconv=True,
                               is_batchnorm=True)
    elif model_type == "SDNet70":
        net_model = SDNet70Model(n_classes=classes,
                                 in_channels=inchannels,
                                 is_deconv=True,
                                 is_batchnorm=True)
    elif model_type == "InversionNet":
        net_model = InversionNet()
    elif model_type == "FCNVMB":
        net_model = FCNVMB(n_classes=classes,
                           in_channels=inchannels,
                           is_deconv=True,
                           is_batchnorm=True)
    else:
        net_model = None
        print(
            'The "model_type" parameter selected in the determine_network(...)'
            ' is the undefined network model keyword! Please check!')
        exit(0)

    # Inherit the previous network structure
    if external_model_src is not "":
        net_model = model_reader(net=net_model, device=device, save_src=external_model_src)

    # Allocate GPUs and set optimizers
    if torch.cuda.is_available():
        net_model = torch.nn.DataParallel(net_model.cuda(), device_ids=gpus)

    optimizer = torch.optim.Adam(net_model.parameters(), lr=learning_rate)

    return net_model, device, optimizer

def load_dataset(stage = 3):
    '''
    Load the training data according to the parameters in "param_config"

    :return:
    '''

    print("---------------------------------")
    print("· Loading the datasets...")

    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        data_set, label_sets = batch_read_matfile(data_dir, 1, train_size, "train")
    else:
        data_set, label_sets = batch_read_npyfile(data_dir, 1, ceil(train_size / 500), "train")
        for i in range(data_set.shape[0]):
            vm = label_sets[0][i][0]
            label_sets[0][i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        middle_shot_id = 15
        first_p = 9
        second_p = 18
    else:
        middle_shot_id = 2
        first_p = 2
        second_p = 4

    if stage == 1:
        for eachData in range(train_size):
            middle_shot = data_set[eachData, middle_shot_id, :, :].copy()
            middle_shot_with_noise = add_gasuss_noise(middle_shot.copy())
            middle_shot_magnified = magnify_amplitude_fornumpy(middle_shot.copy())
            for j in range(second_p, inchannels):
                data_set[eachData, j, :, :] = middle_shot
            for j in range(first_p, second_p):
                data_set[eachData, j, :, :] = middle_shot_magnified
            for j in range(0, first_p):
                data_set[eachData, j, :, :] = middle_shot_with_noise
    elif stage == 2:
        for eachBatch in range(train_size):
            middle_shot = data_set[eachBatch, middle_shot_id, :, :].copy()
            for eachChannel in range(inchannels):
                data_set[eachBatch, eachChannel, :, :] = middle_shot
    else:
        pass

    # Training set
    seis_and_vm = data_utils.TensorDataset(
        torch.from_numpy(data_set).float(),
        torch.from_numpy(label_sets[0]).float(),
        torch.from_numpy(label_sets[1]).long())
    seis_and_vm_loader = data_utils.DataLoader(
        seis_and_vm,
        batch_size=train_batch_size,
        pin_memory=True,
        shuffle=True)

    print("· Number of seismic gathers included in the training set: {}".format(train_size))
    print("· Dimensions of seismic data: ({},{},{},{})".format(train_size, inchannels, data_dim[0], data_dim[1]))
    print("· Dimensions of velocity model: ({},{},{},{})".format(train_size, classes, model_dim[0], model_dim[1]))
    print("---------------------------------")

    return seis_and_vm_loader, data_set, label_sets

def train_for_one_stage(cur_epochs, model, training_loader, optimizer, save_times = 1, key_word = "CLstage1", model_type = "DDNet"):
    '''
    Training for designated epochs

    :param cur_epochs:      Designated epochs
    :param model:           Network model objects to be used for training
    :param training_loader: Trainin dataset loader to be fed into the network
    :param optimizer:       Optimizer
    :param key_word:        After the training, the keywords will be saved to the model
    :param stage_keyword:   The selected difficulty keyword (set "no settings" to ignore CL)
    :param model_type:      The main model used, this model is differentiated based on different papers.
                            The available key model keywords are [DDNet | DDNet70 | InversionNet | FCNVMB]
    :return:                Model save path
    '''

    loss_of_stage = []
    last_model_save_path = ""
    step = int(train_size / train_batch_size)       # Total number of batches to train
    save_epoch = cur_epochs // save_times
    training_time = 0

    model_save_name = "{}_{}_TrSize{}_AllEpo{}".format(dataset_name, key_word, train_size, cur_epochs)

    for epoch in range(cur_epochs):
        # Training for the current epoch
        loss_of_epoch = 0.0
        cur_node_time = time.time()
        ############
        # training #
        ############
        for i, (images, labels, contours_labels) in enumerate(training_loader):

            iteration = epoch * step + i + 1
            model.train()

            # Load to GPU
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                contours_labels = contours_labels.cuda(non_blocking=True)

            # Gradient cache clearing
            optimizer.zero_grad()
            criterion = LossDDNet(weights=loss_weight)

            if model_type in ["DDNet", "DDNet70"]:
                outputs = model(images, model_dim)
                loss = criterion(outputs[0], outputs[1], labels, contours_labels)
            elif model_type in ["SDNet", "SDNet70"]:
                output = model(images, model_dim)
                loss = F.mse_loss(output, labels, reduction='sum') / (model_dim[0] * model_dim[1] * train_batch_size)
            else:
                print(
                    'The "model_type" parameter selected in the train_for_one_stage(...)'
                    ' is the undefined network model keyword! Please check!')
                exit(0)

            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while training')

            # Loss backward propagation
            loss.backward()

            # Optimize
            optimizer.step()

            loss_of_epoch += loss.item()

            if iteration % display_step == 0:
                print('[{}] Epochs: {}/{}, Iteration: {}/{} --- Training Loss:{:.6f}'
                      .format(key_word, epoch + 1, cur_epochs, iteration, step * cur_epochs, loss.item()))

        ################################
        # The end of the current epoch #
        ################################
        if (epoch + 1) % 1 == 0:

            # Calculate the average loss of the current epoch
            print('[{}] Epochs: {:d} finished ! Training loss: {:.5f}'
                  .format(key_word, epoch + 1, loss_of_epoch / i))

            # Include the average loss in the array belonging to the current stage
            loss_of_stage.append(loss_of_epoch / i)

            # Statistics of the time spent in a epoch
            time_elapsed = time.time() - cur_node_time
            print('[{}] Epochs consuming time: {:.0f}m {:.0f}s'
                  .format(key_word, time_elapsed // 60, time_elapsed % 60))
            training_time += time_elapsed
        #########################################################################
        # When it reaches the point where intermediate results can be stored... #
        #########################################################################
        if (epoch + 1) % save_epoch == 0:
            last_model_save_path = models_dir + model_save_name + '_CurEpo' + str(epoch + 1) + '.pkl'
            torch.save(model.state_dict(), last_model_save_path)
            print('[' + key_word + '] Trained model saved: %d percent completed' % int((epoch + 1) * 100 / cur_epochs))

        np.save(results_dir + "[Loss]" + model_save_name + ".npy", np.array(loss_of_stage))

    return last_model_save_path, training_time

def curriculum_learning_training(model_type):
    '''
    Curriculum learning

    :param model_type:              The main model used, this model is differentiated based on different papers.
                                    The available key model keywords are
                                    [DDNet70 | DDNet | InversionNet | FCNVMB| SDNet70 | SDNet]
    '''
    all_training_time = 0

    init_model_src = ""
    stage1_net_src = ""
    stage2_net_src = ""


    ###########
    # Stage 1 #
    ###########
    if firststage_epochs != 0:
        print("read path: {}".format(init_model_src))
        net_model, device, optimizer = determine_network(external_model_src=init_model_src, model_type=model_type)
        training_loader, seismic_gathers, velocity_models = load_dataset(stage=1)
        stage1_net_src, training_time = train_for_one_stage(firststage_epochs, net_model, training_loader,
                                                            optimizer, key_word="CLStage1", model_type=model_type,
                                                            save_times=2)
        all_training_time += training_time
        del training_loader
        del seismic_gathers
        del velocity_models
        del net_model
        del optimizer
        del device
        gc.collect()

    ###########
    # Stage 2 #
    ###########
    if secondstage_epochs != 0:
        print("read path: {}".format(stage1_net_src))
        net_model, device, optimizer = determine_network(external_model_src=stage1_net_src, model_type=model_type)
        training_loader, seismic_gathers, velocity_models = load_dataset(stage=2)
        stage2_net_src, training_time = train_for_one_stage(secondstage_epochs, net_model, training_loader,
                                                            optimizer, key_word="CLStage2", model_type=model_type,
                                                            save_times=2)
        all_training_time += training_time
        del training_loader
        del seismic_gathers
        del velocity_models
        del net_model
        del optimizer
        del device
        gc.collect()

    ###########
    # Stage 3 #
    ###########
    if thirdstage_epochs != 0:
        print("read path: {}".format(stage2_net_src))
        net_model, device, optimizer = determine_network(external_model_src=stage2_net_src, model_type=model_type)
        training_loader, seismic_gathers, velocity_models = load_dataset(stage=3)
        stage3_net_src, training_time = train_for_one_stage(thirdstage_epochs, net_model, training_loader,
                                                            optimizer, key_word="CLStage3", model_type=model_type,
                                                            save_times=2)
        all_training_time += training_time
        del training_loader
        del seismic_gathers
        del velocity_models
        del net_model
        del optimizer
        del device
        gc.collect()

    print("training runtime: {}s".format(all_training_time))

if __name__ == "__main__":
    curriculum_learning_training("DDNet70")
