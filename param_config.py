# -*- coding: utf-8 -*-
"""
Parameters setting

Created on Sep 2023

@author: Xing-Yi Zhang (Zhangzxy20004182@163.com)

"""

####################################################
####             MAIN PARAMETERS                ####
####################################################

# Existing datasets: SEGSalt|SEGSimulation|FlatVelA|CurveFaultA|FlatFaultA|CurveVelA)
dataset_name  = 'SEGSimulation'
learning_rate = 0.001                               # Learning rate
classes = 1                                         # Number of output channels
display_step = 2                                    # Number of training sessions required to print a "loss"

####################################################
####            DATASET PARAMETERS              ####
####################################################

if dataset_name  == 'SEGSimulation':
    data_dim = [400, 301]                           # Dimension of original one-shot seismic data
    model_dim = [201, 301]                          # Dimension of one velocity model
    inchannels = 29                                 # Number of input channels
    train_size = 1600                               # Number of training sets
    test_size = 100                                 # Number of testing sets

    firststage_epochs = 40
    secondstage_epochs = 30
    thirdstage_epochs = 30
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 10                           # Number of batches fed in network in one training epoch.
    test_batch_size = 2

elif dataset_name  == 'SEGSalt':
    data_dim = [400, 301]
    model_dim = [201, 301]
    inchannels = 29
    train_size = 130
    test_size = 10

    firststage_epochs = 0
    secondstage_epochs = 0
    thirdstage_epochs = 50                          # SEGSalt for transfer learning and does not require curriculum tasks
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 10
    test_batch_size = 2

elif dataset_name == 'FlatVelA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 5000
    test_size = 1000

    firststage_epochs = 30
    secondstage_epochs = 20
    thirdstage_epochs = 50
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 50
    test_batch_size = 5

elif dataset_name == 'CurveVelA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 5000
    test_size = 1000

    firststage_epochs = 10
    secondstage_epochs = 10
    thirdstage_epochs = 80
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 50
    test_batch_size = 5

elif dataset_name == 'FlatFaultA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 5000
    test_size = 1000

    firststage_epochs = 10
    secondstage_epochs = 10
    thirdstage_epochs = 80
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 50
    test_batch_size = 5

elif dataset_name == 'CurveFaultA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 5000
    test_size = 1000

    firststage_epochs = 30
    secondstage_epochs = 20
    thirdstage_epochs = 50
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 50
    test_batch_size = 5

else:
    print('The selected dataset is invalid')
    exit(0)

