# -*- coding: utf-8 -*-
"""
Direct method for reading datasets

Created on Feb 2023

@author: Xing-Yi Zhang (Zhangzxy20004182@163.com)

"""

from func.net import *
from path_config import *
from param_config import *
from func.utils import extract_contours, pain_openfwi_seismic_data, pain_openfwi_velocity_model, pain_seg_seismic_data, pain_seg_velocity_model

def batch_read_matfile(dataset_dir,
                       seismic_data_size,
                       velocity_model_size,
                       start,
                       batch_length,
                       train_or_test = "train",
                       data_channels = 29):         # In this code, only SEG data is used in .mat, and they are all 29 channels
    '''
    Batch read seismic gathers and velocity models for .mat file

    :param dataset_dir:             Path to the dataset
    :param seismic_data_size:       Size of the seimic data
    :param velocity_model_size:     Size of the velocity model
    :param start:                   Start reading from the number of data
    :param batch_length:            Starting from the defined first number of data, how long to read
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :param data_channels:           The total number of channels read into the data itself
    :return:                        a quadruple: (seismic data, [velocity model, contour of velocity model])
                                    Among them, the dimensions of seismic data, velocity model and contour of velocity model are all (number of read data, channel, width x height)
    '''
    for i in range(start, start + batch_length):

        ##############################
        ##    Load Seismic Data     ##
        ##############################

        # Determine the seismic data path in the dataset
        filename_seis = dataset_dir + '{}_data/seismic/seismic{}.mat'.format(train_or_test, i)
        print("Reading: {}".format(filename_seis))

        seis_data_multi_shots = scipy.io.loadmat(filename_seis)
        seis_data_multi_shots = np.float32(seis_data_multi_shots["data"].reshape([seismic_data_size[0], seismic_data_size[1], data_channels]))

        for k in range(data_channels):
            one_shot = np.float32(seis_data_multi_shots[:, :, k])
            one_shot_downsampling = block_reduce(one_shot, block_size=(1, 1), func=decimate)
            data_dsp_dim = one_shot_downsampling.shape
            onedim_vector = one_shot_downsampling.reshape(1, data_dsp_dim[0] * data_dsp_dim[1])
            if k == 0:
                multishots_for_oneImage = onedim_vector
            else:
                multishots_for_oneImage = np.append(multishots_for_oneImage, onedim_vector, axis=0)

        ##############################
        ##    Load Velocity Model   ##
        ##############################

        # Determine the velocity model path in the dataset
        filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.mat'.format(train_or_test, i)
        print("Reading: {}".format(filename_label))

        vm_data = scipy.io.loadmat(filename_label)
        vm_data = np.float32(vm_data["data"].reshape(velocity_model_size))
        contours_vm_data = extract_contours(vm_data)                # Use Canny to extract contour features

        vm_data_downsampling = block_reduce(vm_data, block_size=(1,1), func=np.max)
        contours_vm_data_downsampling = block_reduce(contours_vm_data, block_size=(1,1), func=np.max)

        # Dimensions of data
        label_dsp_dim = vm_data_downsampling.shape

        vm_data_one_dim = vm_data_downsampling.reshape(1, label_dsp_dim[0] * label_dsp_dim[1])
        vm_data_one_dim = np.float32(vm_data_one_dim)
        contours_vm_data_one_dim = contours_vm_data_downsampling.reshape(1, label_dsp_dim[0] * label_dsp_dim[1])
        contours_vm_data_one_dim = np.float32(contours_vm_data_one_dim)

        ##############################
        ## Finite Current Loading   ##
        ##############################

        # Accumulation of seismic data and velocity models each
        if i == start:
            input_set = multishots_for_oneImage
            label_set = vm_data_one_dim
            clabel_set = contours_vm_data_one_dim
        else:
            input_set = np.append(input_set, multishots_for_oneImage, axis=0)
            label_set = np.append(label_set, vm_data_one_dim, axis=0)
            clabel_set = np.append(clabel_set, contours_vm_data_one_dim, axis=0)

    # Finally, the seismic data and velocity model are reconstructed as data with a dim: (number of read data, channel, width x height)
    data_set    = input_set.reshape((batch_length, data_channels,  data_dsp_dim[0] *  data_dsp_dim[1]))     # seismic data
    label_set   = label_set.reshape((batch_length,       1,       label_dsp_dim[0] * label_dsp_dim[1]))     # velocity model
    clabel_set  = clabel_set.reshape((batch_length,      1,       label_dsp_dim[0] * label_dsp_dim[1]))     # contour of velocity model

    return data_set, [label_set, clabel_set]

def decimate(a, axis):
    idx = np.round((np.array(a.shape)[np.array(axis).reshape(1, -1)] + 1.0) / 2.0 - 1).reshape(-1)
    downa = np.array(a)[:, :, idx[0].astype(int), idx[1].astype(int)]
    return downa

def batch_read_npyfile(dataset_dir,
                       start,
                       batch_length,
                       train_or_test = "train"):
    '''
    Batch read seismic gathers and velocity models for .npy file

    :param dataset_dir:             Path to the dataset
    :param start:                   Start reading from the number of data
    :param batch_length:            Starting from the defined first number of data, how long to read
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :return:                        a pair: (seismic data, [velocity model, contour of velocity model])
                                    Among them, the dimensions of seismic data, velocity model and contour of velocity
                                    model are all (number of read data * 500, channel, height, width)
    '''

    dataset = None
    labelset = None

    for i in range(start, start + batch_length):

        ##############################
        ##    Load Seismic Data     ##
        ##############################

        # Determine the seismic data path in the dataset
        filename_seis = dataset_dir + '{}_data/seismic/seismic{}.npy'.format(train_or_test, i)
        print("Reading: {}".format(filename_seis))

        if i == start:
            dataset = np.load(filename_seis)
        else:
            dataset = np.append(dataset, np.load(filename_seis), axis=0)

        ##############################
        ##    Load Velocity Model   ##
        ##############################

        # Determine the velocity model path in the dataset
        filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(train_or_test, i)
        print("Reading: {}".format(filename_label))

        if i == start:
            labelset = np.load(filename_label)
        else:
            labelset = np.append(labelset, np.load(filename_label), axis=0)

    # OpenFWI related networks have normalized the velocity model
    print("Velocity model normalization in progress...")
    for i in range(labelset.shape[0]):
        for j in range(labelset.shape[1]):
            temp = labelset[i, j, ...]
            labelset[i, j, ...] = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))

    print("Generating velocity model profile......")
    conlabels = np.zeros([batch_length * 500, classes, model_dim[0], model_dim[1]])
    for i in range(labelset.shape[0]):
        for j in range(labelset.shape[1]):
            conlabels[i, j, ...] = extract_contours(labelset[i, j, ...])

    return dataset, [labelset, conlabels]


def single_read_matfile(dataset_dir,
                        seismic_data_size,
                        velocity_model_size,
                        readID,
                        train_or_test = "train",
                        data_channels = 29):
    '''
    Single read seismic gathers and velocity models for .mat file

    :param dataset_dir:             Path to the dataset
    :param seismic_data_size:       Size of the seimic data
    :param velocity_model_size:     Size of the velocity model
    :param readID:                  The ID number of the selected data
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :param data_channels:           The total number of channels read into the data itself
    :return:                        a triplet: (seismic data, velocity model, contour of velocity model)
                                    Among them, the dimensions of seismic data, velocity model and contour of velocity model are
                                    (channel, width, height), (width, height) and (width, height) respectively
    '''
    filename_seis = dataset_dir + '{}_data/seismic/seismic{}.mat'.format(train_or_test, readID)
    print("Reading: {}".format(filename_seis))
    filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.mat'.format(train_or_test, readID)
    print("Reading: {}".format(filename_label))

    se_data = scipy.io.loadmat(filename_seis)
    se_data = np.float32(se_data["data"].reshape([seismic_data_size[0], seismic_data_size[1], data_channels]))
    vm_data = scipy.io.loadmat(filename_label)
    vm_data = np.float32(vm_data["data"].reshape(velocity_model_size[0], velocity_model_size[1]))

    # (400, 301, 29) -> (29, 400, 301)
    se_data = se_data.swapaxes(0, 2)
    se_data = se_data.swapaxes(1, 2)

    contours_vm_data = extract_contours(vm_data)  # Use Canny to extract contour features

    return se_data, vm_data, contours_vm_data

def single_read_npyfile(dataset_dir,
                        readIDs,
                        train_or_test = "train"):
    '''
    Single read seismic gathers and velocity models for .npy file

    :param dataset_dir:             Path to the dataset
    :param readID:                  The IDs number of the selected data
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :return:                        seismic data, velocity model, contour of velocity model, maximum velocity, minimum velocity
    '''

    # Determine the seismic data path in the dataset
    filename_seis = dataset_dir + '{}_data/seismic/seismic{}.npy'.format(train_or_test, readIDs[0])
    print("Reading: {}".format(filename_seis))
    # Determine the velocity model path in the dataset
    filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(train_or_test, readIDs[0])
    print("Reading: {}".format(filename_label))

    se_data = np.load(filename_seis)[readIDs[1]]
    vm_data = np.load(filename_label)[readIDs[1]][0]

    # OpenFWI related networks have normalized the velocity model
    print("Velocity model normalization in progress...")
    vmax, vmin = np.max(vm_data), np.min(vm_data)
    vm_data = (vm_data - np.min(vm_data)) / (np.max(vm_data) - np.min(vm_data))
    print("Generating velocity model profile......")
    conlabel = extract_contours(vm_data)

    return se_data, vm_data, conlabel, vmax, vmin




if __name__ == "__main__":
    ######################################################################################
    # Read test for SEG (You need to select the appropriate dataset name before running) #
    ######################################################################################

    # Individual read tests of the SEG dataset
    seismic_data, velocity_model, contours  = single_read_matfile(data_dir, data_dim, model_dim, 25, "train")
    print(seismic_data.shape)               # (29, 400, 301)
    print(velocity_model.shape)             # (201, 301)
    print(contours.shape)                   # (201, 301)
    pain_seg_seismic_data(seismic_data[15])
    pain_seg_velocity_model(velocity_model, np.min(velocity_model), np.max(velocity_model))
    pain_seg_velocity_model(contours, 0, 1)
    print("-----------------")

    # Batch read test of SEG dataset
    data_set, [label_set, clabel_set], _, _ = batch_read_matfile(data_dir, data_dim, model_dim, 100, 12, "train")
    print(data_set.shape)                   # (12, 29, 120400) | 120400 = 400 x 301
    print(label_set.shape)                  # (12, 1, 60501)   | 60501  = 201 x 301
    print(clabel_set.shape)                 # (12, 1, 60501)   | 60501  = 201 x 301

    ##########################################################################################
    # Read test for OpenFWI (You need to select the appropriate dataset name before running) #
    ##########################################################################################

    # # Individual read tests of the OpenFWI dataset
    # seismic_data, velocity_model, contours  = single_read_npyfile(data_dir, [5, 100], "train")
    # print(seismic_data.shape)               # (5, 1000, 70)
    # print(velocity_model.shape)             # (70, 70)
    # print(contours.shape)                   # (70, 70)
    # pain_openfwi_seismic_data(seismic_data[2])
    # pain_openfwi_velocity_model(velocity_model, np.min(velocity_model), np.max(velocity_model))
    # pain_openfwi_velocity_model(contours, 0, 1)
    #
    # # Batch read test of OpenFWI dataset
    # data_set, [label_set, clabel_set]       = batch_read_npyfile(data_dir, 5, 3, "train")
    # print(data_set.shape)                   # (1500, 5, 1000, 70)
    # print(label_set.shape)                  # (1500, 1, 70, 70)
    # print(clabel_set.shape)                 # (5000, 1, 70, 70)