# -*- coding: utf-8 -*-
"""
Direct method for reading datasets

Created on Sep 2023

@author: Xing-Yi Zhang (zxy20004182@163.com)

"""

from param_config import *
import scipy.io
import scipy
import numpy as np
from func.utils import extract_contours

def batch_read_matfile(dataset_dir,
                       start,
                       batch_length,
                       train_or_test="train",
                       data_channels=29):
    '''
    Batch read seismic gathers and velocity models for .mat file

    :param dataset_dir:             Path to the dataset
    :param start:                   Start reading from the number of data
    :param batch_length:            Starting from the defined first number of data, how long to read
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :param data_channels:           The total number of channels read into the data itself
    :return:                        a quadruple: (seismic data, [velocity model, contour of velocity model])
                                    Among them, the dimensions of seismic data, velocity model and contour of velocity model are all (number of read data, channel, width x height)
    '''

    data_set = np.zeros([batch_length, data_channels, data_dim[0], data_dim[1]])
    label_set = np.zeros([batch_length, classes, model_dim[0], model_dim[1]])
    clabel_set = np.zeros([batch_length, classes, model_dim[0], model_dim[1]])

    for indx, i in enumerate(range(start, start + batch_length)):

        # Load Seismic Data
        filename_seis = dataset_dir + '{}_data/seismic/seismic{}.mat'.format(train_or_test, i)
        print("Reading: {}".format(filename_seis))
        sei_data = scipy.io.loadmat(filename_seis)["data"]
        # (400, 301, 29) -> (29, 400, 301)
        sei_data = sei_data.swapaxes(0, 2)
        sei_data = sei_data.swapaxes(1, 2)
        for ch in range(inchannels):
            data_set[indx, ch, ...] = sei_data[ch, ...]

        # Load Velocity Model
        filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.mat'.format(train_or_test, i)
        print("Reading: {}".format(filename_label))
        vm_data = scipy.io.loadmat(filename_label)["data"]
        label_set[indx, 0, ...] = vm_data
        clabel_set[indx, 0, ...] = extract_contours(vm_data)

    return data_set, [label_set, clabel_set]

def batch_read_npyfile(dataset_dir,
                       start,
                       batch_length,
                       train_or_test="train"):
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

    dataset = []
    labelset = []

    for i in range(start, start + batch_length):

        ##############################
        ##    Load Seismic Data     ##
        ##############################

        # Determine the seismic data path in the dataset
        filename_seis = dataset_dir + '{}_data/seismic/seismic{}.npy'.format(train_or_test, i)
        print("Reading: {}".format(filename_seis))
        temp_data = np.load(filename_seis)

        dataset.append(temp_data)

        ##############################
        ##    Load Velocity Model   ##
        ##############################

        # Determine the velocity model path in the dataset
        filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(train_or_test, i)
        print("Reading: {}".format(filename_label))
        temp_data = np.load(filename_label)

        labelset.append(temp_data)

    dataset = np.vstack(dataset)
    labelset = np.vstack(labelset)

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
    :return:                        seismic data, velocity model, contour of velocity model
    '''

    # Determine the seismic data path in the dataset
    filename_seis = dataset_dir + '{}_data/seismic/seismic{}.npy'.format(train_or_test, readIDs[0])
    print("Reading: {}".format(filename_seis))
    # Determine the velocity model path in the dataset
    filename_label = dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(train_or_test, readIDs[0])
    print("Reading: {}".format(filename_label))

    se_data = np.load(filename_seis)[readIDs[1]]
    vm_data = np.load(filename_label)[readIDs[1]][0]

    print("Generating velocity model profile......")
    conlabel = extract_contours(vm_data)

    return se_data, vm_data, conlabel
