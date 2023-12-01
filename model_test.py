# -*- coding: utf-8 -*-
"""
Test the model effect after training.

Created on Sep 2023

@author: Xing-Yi Zhang (Zhangzxy20004182@163.com)

"""
from path_config import *
from func.utils import run_mse, run_mae, run_lpips, run_uqi, pain_seg_seismic_data, pain_seg_velocity_model,\
    pain_openfwi_velocity_model, pain_openfwi_seismic_data
from func.datasets_reader import batch_read_matfile, batch_read_npyfile, single_read_matfile, single_read_npyfile
from model_train import determine_network

import time
import lpips
import numpy as np
import torch
import torch.utils.data as data_utils

import matplotlib
matplotlib.use('TkAgg')


def load_dataset():
    '''
    Load the testing data according to the parameters in "param_config"

    :return:    A triplet: datasets loader, seismic gathers and velocity models
    '''

    print("---------------------------------")
    print("· Loading the datasets...")
    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        data_set, label_sets = batch_read_matfile(data_dir, 1 if dataset_name == 'SEGSalt'
                                                              else 1601, test_size, "test")
    else:
        data_set, label_sets = batch_read_npyfile(data_dir, 1, test_size // 500, "test")
        for i in range(data_set.shape[0]):
            vm = label_sets[0][i][0]
            max_velocity, min_velocity = np.max(vm), np.min(vm)
            label_sets[0][i][0] = (vm - min_velocity) / (max_velocity - min_velocity)

    print("· Number of seismic gathers included in the testing set: {}.".format(test_size))
    print("· Dimensions of seismic data: ({},{},{},{}).".format(test_size, inchannels, data_dim[0], data_dim[1]))
    print("· Dimensions of velocity model: ({},{},{},{}).".format(test_size, classes, model_dim[0], model_dim[1]))
    print("---------------------------------")

    seis_and_vm = data_utils.TensorDataset(torch.from_numpy(data_set).float(),
                                           torch.from_numpy(label_sets[0]).float())
    seis_and_vm_loader = data_utils.DataLoader(seis_and_vm, batch_size=test_batch_size, shuffle=True)

    return seis_and_vm_loader, data_set, label_sets

def batch_test(model_path, model_type = "DDNet"):
    '''
    Batch testing for multiple seismic data

    :param model_path:              Model path
    :param model_type:              The main model used, this model is differentiated based on different papers.
    :return:
    '''

    loader, seismic_gathers, velocity_models = load_dataset()

    print("Loading test model:{}".format(model_path))
    model_net, device, optimizer = determine_network(model_path, model_type=model_type)

    mse_record = np.zeros((1, test_size), dtype=float)
    mae_record = np.zeros((1, test_size), dtype=float)
    uqi_record = np.zeros((1, test_size), dtype=float)
    lpips_record = np.zeros((1, test_size), dtype=float)

    counter = 0

    lpips_object = lpips.LPIPS(net='alex', version="0.1")

    cur_node_time = time.time()
    for i, (seis_image, gt_vmodel) in enumerate(loader):

        if torch.cuda.is_available():
            seis_image = seis_image.cuda(non_blocking=True)
            gt_vmodel = gt_vmodel.cuda(non_blocking=True)

        # Prediction
        model_net.eval()
        if model_type in ["DDNet", "DDNet70"]:
            [outputs, _] = model_net(seis_image, model_dim)
        elif model_type in ["SDNet", "SDNet70"]:
            outputs = model_net(seis_image, model_dim)
        elif model_type == "InversionNet":
            outputs = model_net(seis_image)
        elif model_type == "FCNVMB":
            outputs = model_net(seis_image, model_dim)
        else:
            print('The "model_type" parameter selected in the batch_test(...) '
                  'is the undefined network model keyword! Please check!')
            exit(0)

        # # Both target labels and prediction tags return to "numpy"
        pd_vmodel = outputs.cpu().detach().numpy()
        pd_vmodel = np.where(pd_vmodel > 0.0, pd_vmodel, 0.0)   # Delete bad points
        gt_vmodel = gt_vmodel.cpu().detach().numpy()

        # Calculate MSE, MAE, UQI and LPIPS of the current batch
        for k in range(test_batch_size):

            pd_vmodel_of_k = pd_vmodel[k, 0, :, :]
            gt_vmodel_of_k = gt_vmodel[k, 0, :, :]

            mse_record[0, counter]   = run_mse(pd_vmodel_of_k, gt_vmodel_of_k)
            mae_record[0, counter]   = run_mae(pd_vmodel_of_k, gt_vmodel_of_k)
            uqi_record[0, counter]   = run_uqi(gt_vmodel_of_k, pd_vmodel_of_k)
            lpips_record[0, counter] = run_lpips(gt_vmodel_of_k, pd_vmodel_of_k, lpips_object)

            print('The %d testing MSE: %.4f\tMAE: %.4f\tUQI: %.4f\tLPIPS: %.4f' %
                  (counter, mse_record[0, counter], mae_record[0, counter],
                   uqi_record[0, counter], lpips_record[0, counter]))
            counter = counter + 1
    time_elapsed = time.time() - cur_node_time

    print("The average of MSE: {:.6f}".format(mse_record.mean()))
    print("The average of MAE: {:.6f}".format(mae_record.mean()))
    print("The average of UQI: {:.6f}".format(uqi_record.mean()))
    print("The average of LIPIS: {:.6f}".format(lpips_record.mean()))
    print("-----------------")
    print("Time-consuming testing of batch samples: {:.6f}".format(time_elapsed))
    print("Average test-consuming per sample: {:.6f}".format(time_elapsed / test_size))

def single_test(model_path, select_id, train_or_test = "test", model_type = "DDNet"):
    '''
    Batch testing for single seismic data

    :param model_path:              Model path
    :param select_id:               The ID of the selected data. if it is openfwi, here is a pair,
                                    e.g. [11, 100], otherwise it is just a single number, e.g. 56.
    :param train_or_test:           Whether the data set belongs to the training set or the testing set
    :param model_type:              The main model used, this model is differentiated based on different papers.
                                    The available key model keywords are [DDNet70 | DDNet | InversionNet | FCNVMB]
    :return:
    '''

    print("Loading test model:{}".format(model_path))
    model_net, device, optimizer = determine_network(model_path, model_type=model_type)

    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        seismic_data, velocity_model, _ = single_read_matfile(data_dir, data_dim, model_dim, select_id, train_or_test = train_or_test)
        max_velocity, min_velocity = np.max(velocity_model), np.min(velocity_model)
    else:
        seismic_data, velocity_model, _ = single_read_npyfile(data_dir, select_id, train_or_test = train_or_test)
        max_velocity, min_velocity = np.max(velocity_model), np.min(velocity_model)
        velocity_model = (velocity_model - np.min(velocity_model)) / (np.max(velocity_model) - np.min(velocity_model))

    lpips_object = lpips.LPIPS(net='alex', version="0.1")


    # Convert numpy to tensor and load it to GPU
    seismic_data_tensor = torch.from_numpy(np.array([seismic_data])).float()
    if torch.cuda.is_available():
        seismic_data_tensor = seismic_data_tensor.cuda(non_blocking=True)

    # Prediction
    model_net.eval()
    cur_node_time = time.time()
    if model_type in ["DDNet", "DDNet70"]:
        [predicted_vmod_tensor, _] = model_net(seismic_data_tensor, model_dim)
    elif model_type in ["SDNet", "SDNet70"]:
        predicted_vmod_tensor = model_net(seismic_data_tensor, model_dim)
    elif model_type == "InversionNet":
        predicted_vmod_tensor = model_net(seismic_data_tensor)
    elif model_type == "FCNVMB":
        predicted_vmod_tensor = model_net(seismic_data_tensor, model_dim)
    else:
        print('The "model_type" parameter selected in the single_test(...) '
              'is the undefined network model keyword! Please check!')
        exit(0)
    time_elapsed = time.time() - cur_node_time

    predicted_vmod = predicted_vmod_tensor.cpu().detach().numpy()[0][0]     # (1, 1, X, X)
    predicted_vmod = np.where(predicted_vmod > 0.0, predicted_vmod, 0.0)    # Delete bad points

    mse   = run_mse(predicted_vmod, velocity_model)
    mae   = run_mae(predicted_vmod, velocity_model)
    uqi   = run_uqi(velocity_model, predicted_vmod)
    lpi = run_lpips(velocity_model, predicted_vmod, lpips_object)

    print('MSE: %.6f\nMAE: %.6f\nUQI: %.6f\nLPIPS: %.6f' % (mse, mae, uqi, lpi))
    print("-----------------")
    print("Time-consuming testing of a sample: {:.6f}".format(time_elapsed))

    # Show
    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        pain_seg_seismic_data(seismic_data[15])
        pain_seg_velocity_model(velocity_model, min_velocity, max_velocity)
        pain_seg_velocity_model(predicted_vmod, min_velocity, max_velocity)
    else:
        pain_openfwi_seismic_data(seismic_data[2])
        minV = np.min(min_velocity + velocity_model * (max_velocity - min_velocity))
        maxV = np.max(min_velocity + velocity_model * (max_velocity - min_velocity))
        pain_openfwi_velocity_model(min_velocity + velocity_model * (max_velocity - min_velocity), minV, maxV)
        pain_openfwi_velocity_model(min_velocity + predicted_vmod * (max_velocity - min_velocity), minV, maxV)

if __name__ == "__main__":
    batch_of_single = 1
    # |DDNet|DDNet70|InversionNet|FCNVMB|SDNet|SDNet70|
    model_type = "DDNet"

    if batch_of_single == 1:
        # Batch test #
        batch_test("...", model_type=model_type)
    else:
        # Single test #
        if dataset_name in ["SEGSalt", "SEGSimulation"]:
            # 1~10      :SEGSalt
            # 1601~1700 :SEGSimulation
            select_id = 1615
        else:
            # [1~2, 0~499]
            select_id = [11, 104]
        single_test("...", select_id=select_id, model_type=model_type)
