# -*- coding: utf-8 -*-
"""
Test the model effect after training.

Created on Feb 2023

@author: Xing-Yi Zhang (Zhangzxy20004182@163.com)

"""

from func.utils import *
from func.datasets_reader import *
from model_train import determine_network

def load_dataset():
    '''
    Load the testing data according to the parameters in "param_config"

    :return:    A triplet: datasets loader, seismic gathers and velocity models
    '''

    print("---------------------------------")
    print("· Loading the datasets...")
    if dataset_name in ['SEGReal', 'SEGSimulation']:
        data_set, label_sets = batch_read_matfile(data_dir, data_dim, model_dim,
                                                  1 if dataset_name == 'SEGReal' else 1601, test_size, "test")
        data_set = data_set.reshape(test_size, inchannels, data_dim[0], data_dim[1])
        label_sets[0] = label_sets[0].reshape(test_size, classes, model_dim[0], model_dim[1])
        label_sets[1] = label_sets[1].reshape(test_size, classes, model_dim[0], model_dim[1])
    else:
        data_set, label_sets = batch_read_npyfile(data_dir, 11, test_size // 500, "test")

    print("· Number of seismic gathers included in the testing set: {}.".format(test_size))
    print("· Dimensions of seismic data: ({},{},{},{}).".format(test_size, inchannels, data_dim[0], data_dim[1]))
    print("· Dimensions of velocity model: ({},{},{},{}).".format(test_size, classes, model_dim[0], model_dim[1]))
    print("---------------------------------")

    seis_and_vm = data_utils.TensorDataset(torch.from_numpy(data_set).float(),
                                           torch.from_numpy(label_sets[0]).float(),
                                           torch.from_numpy(label_sets[1]).float())
    seis_and_vm_loader = data_utils.DataLoader(seis_and_vm, batch_size=test_batch_size, shuffle=True)

    return seis_and_vm_loader, data_set, label_sets

def data_analyse(npy_src, npy_name):
    '''
    Analyze the saved evaluation metrics test results (.npy)

    :param npy_src:     File path
    :param npy_name:    File name
    :return:
    '''

    test_infostorage = read_numpy(npy_name, npy_src)
    mse_record = test_infostorage[0]
    mae_record = test_infostorage[1]
    uqi_record = test_infostorage[2]
    lpips_record = test_infostorage[3]

    print("The average of MSE: {:.6f}".format(mse_record.mean()))
    print("The average of MAE: {:.6f}".format(mae_record.mean()))
    print("The average of UQI: {:.6f}".format(uqi_record.mean()))
    print("The average of LIPIS: {:.6f}".format(lpips_record.mean()))

def batch_test(model_path, is_copy_singleshot = False):
    '''
    Batch testing for multiple seismic data

    :param model_path:              Model path
    :param is_copy_singleshot:      Whether to unify all channels of the seismic gathers into one shot
    :return:
    '''

    loader, seismic_gathers, velocity_models = load_dataset()

    print("Loading test model:{}".format(model_path))
    dd_net, device, optimizer = determine_network(model_path)

    mse_record = np.zeros((1, test_size), dtype=float)
    mae_record = np.zeros((1, test_size), dtype=float)
    uqi_record = np.zeros((1, test_size), dtype=float)
    lpips_record = np.zeros((1, test_size), dtype=float)

    counter = 0

    lpips_object = lpips.LPIPS(net='alex', version="0.1")

    for i, (seis_image, gt_vmodel, _) in enumerate(loader):

        seis_image = seis_image.view(test_batch_size, inchannels, data_dim[0], data_dim[1])
        gt_vmodel = gt_vmodel.view(test_batch_size, classes, model_dim[0], model_dim[1])

        # In the ablation experiment, there is a situation:
        # its network input channel consists of the same seismic data.
        # So during testing, we also need to simulate the same scene.
        if is_copy_singleshot:
            show_shot = inchannels // 2 + 1
            for eachBatch in range(test_batch_size):
                middle_shot = seis_image[eachBatch, show_shot, :, :]
                for eachChannel in range(inchannels):
                    seis_image[eachBatch, eachChannel, :, :] = middle_shot

        if torch.cuda.is_available():
            seis_image = seis_image.cuda(non_blocking=True)
            gt_vmodel = gt_vmodel.cuda(non_blocking=True)

        # Prediction
        dd_net.eval()
        [outputs, _] = dd_net(seis_image, model_dim)

        # Both target labels and prediction tags return to "numpy"
        outputs = outputs.view(test_batch_size, model_dim[0], model_dim[1])
        pd_vmodel = outputs.cpu().detach().numpy()
        pd_vmodel = np.where(pd_vmodel > 0.0, pd_vmodel, 0.0)   # Delete bad points
        gt_vmodel = gt_vmodel.cpu().detach().numpy()

        # Calculate MSE, MAE, UQI and LPIPS of the current batch
        for k in range(test_batch_size):

            pd_vmodel_of_k = pd_vmodel[k, :, :].reshape(model_dim[0], model_dim[1])
            gt_vmodel_of_k = gt_vmodel[k, :, :].reshape(model_dim[0], model_dim[1])

            mse_record[0, counter]   = run_mse(pd_vmodel_of_k, gt_vmodel_of_k)
            mae_record[0, counter]   = run_mae(pd_vmodel_of_k, gt_vmodel_of_k)
            uqi_record[0, counter]   = run_uqi(gt_vmodel_of_k, pd_vmodel_of_k)
            lpips_record[0, counter] = run_lpips(gt_vmodel_of_k, pd_vmodel_of_k, lpips_object)

            print('The %d testing MSE: %.4f\tMAE: %.4f\tUQI: %.4f\tLPIPS: %.4f' %
                  (counter, mse_record[0, counter], mae_record[0, counter], uqi_record[0, counter], lpips_record[0, counter]))
            counter = counter + 1

    save_file = []
    save_file.append(mse_record)
    save_file.append(mae_record)
    save_file.append(uqi_record)
    save_file.append(lpips_record)
    save_file = np.array(save_file)

    save_numpy(src_path = results_dir,
               src_name = "{}_MetricsResults.npy".format(dataset_name),
               para_data = save_file)

    data_analyse(results_dir, "{}_MetricsResults.npy".format(dataset_name))

def single_test(model_path, select_id, train_or_test = "test", is_copy_singleshot = False):
    '''
    Batch testing for single seismic data

    :param model_path:              Model path
    :param select_id:               The ID of the selected data. if it is openfwi, here is a pair,
                                    e.g. [11, 100], otherwise it is just a single number, e.g. 56.
    :param train_or_test:           Whether the data set belongs to the training set or the testing set
    :param is_copy_singleshot:      Whether to unify all channels of the seismic gathers into one shot
    :return:
    '''

    print("Loading test model:{}".format(model_path))
    dd_net, device, optimizer = determine_network(model_path)

    if dataset_name in ['SEGReal', 'SEGSimulation']:
        seismic_data, velocity_model, _ = single_read_matfile(data_dir, data_dim, model_dim, select_id, train_or_test = train_or_test)
    else:
        seismic_data, velocity_model, _, max_velocity, min_velocity = single_read_npyfile(data_dir, select_id, train_or_test = train_or_test)

    lpips_object = lpips.LPIPS(net='alex', version="0.1")

    # In the ablation experiment, there is a situation:
    # its network input channel consists of the same seismic data.
    # So during testing, we also need to simulate the same scene.
    if is_copy_singleshot:
        show_shot = inchannels // 2 + 1
        for eachBatch in range(test_batch_size):
            middle_shot = seismic_data[eachBatch, show_shot, :, :]
            for eachChannel in range(inchannels):
                seismic_data[eachBatch, eachChannel, :, :] = middle_shot

    dd_net.eval()
    # Convert numpy to tensor and load it to GPU
    seismic_data_tensor = torch.from_numpy(np.array([seismic_data])).float()
    if torch.cuda.is_available():
        seismic_data_tensor = seismic_data_tensor.cuda(non_blocking=True)

    # Prediction
    [predicted_vmod_tensor, _] = dd_net(seismic_data_tensor, model_dim)
    predicted_vmod = predicted_vmod_tensor.cpu().detach().numpy()[0][0]     # (1, 1, X, X)
    predicted_vmod = np.where(predicted_vmod > 0.0, predicted_vmod, 0.0)    # Delete bad points

    mse   = run_mse(predicted_vmod, velocity_model)
    mae   = run_mae(predicted_vmod, velocity_model)
    uqi   = run_uqi(velocity_model, predicted_vmod)
    lpi = run_lpips(velocity_model, predicted_vmod, lpips_object)

    print('MSE: %.6f\nMAE: %.6f\nUQI: %.6f\nLPIPS: %.6f' % (mse, mae, uqi, lpi))

    if dataset_name in ['SEGReal', 'SEGSimulation']:
        pain_seg_seismic_data(seismic_data[15])
        pain_seg_velocity_model(velocity_model, min_velocity=np.min(velocity_model), max_velocity=np.max(velocity_model))
        pain_seg_velocity_model(predicted_vmod, min_velocity=np.min(velocity_model), max_velocity=np.max(velocity_model))
    else:
        pain_openfwi_seismic_data(seismic_data[2])
        minV = np.min(min_velocity + velocity_model * (max_velocity - min_velocity))
        maxV = np.max(min_velocity + velocity_model * (max_velocity - min_velocity))
        pain_openfwi_velocity_model(min_velocity + velocity_model * (max_velocity - min_velocity), minV, maxV)
        pain_openfwi_velocity_model(min_velocity + predicted_vmod * (max_velocity - min_velocity), minV, maxV)


if __name__ == "__main__":
    ##############
    # Batch test #
    ##############
    # if dataset_name == 'SEGReal':
    #     batch_test(models_dir + "{}_TL.pkl".format(dataset_name))
    # else:
    #     batch_test(models_dir + "{}_CL.pkl".format(dataset_name))

    ###############
    # Single test #
    ###############
    select_id = [11, 125]
    if dataset_name == 'SEGReal':
        single_test(models_dir + "{}_TL.pkl".format(dataset_name), select_id)
    else:
        single_test(models_dir + "{}_CL.pkl".format(dataset_name), select_id)


