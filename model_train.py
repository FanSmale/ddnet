# -*- coding: utf-8 -*-
"""
Curriculum Learning

Created on Sep 2023

@author: Xing-Yi Zhang (zxy20004182@163.com)

"""

from func.datasets_reader import *
from func.comparison_net import InversionNet, FCNVMB

def determine_network(external_model_src = None, model_type = "DDNet"):
    '''
    Request a network object and import an external network, or create an initialized network

    :param external_model_src:  External pkl file path
    :param model_type:          The main model used, this model is differentiated based on different papers.
                                The available key model keywords are [DDNet | DDNet70 | InversionNet | FCNVMB]
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
    elif model_type == "InversionNet":
        net_model = InversionNet()
    elif model_type == "FCNVMB":
        net_model = FCNVMB(n_classes=classes,
                           in_channels=inchannels,
                           is_deconv=True,
                           is_batchnorm=True)
    else:
        print(
            'The "model_type" parameter selected in the determine_network(...)'
            ' is the undefined network model keyword! Please check!')
        exit(0)

    # Inherit the previous network structure
    if external_model_src != None:
        net_model = model_reader(net=net_model, device=device, save_src=external_model_src)

    # Allocate GPUs and set optimizers
    if torch.cuda.is_available():
        net_model = torch.nn.DataParallel(net_model, device_ids=gpus).cuda()

    if model_type != "InversionNet":
        optimizer = torch.optim.Adam(net_model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(net_model.parameters(), lr=0.0001)

    return net_model, device, optimizer

def load_dataset():
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

    # Training set
    seis_and_vm = data_utils.TensorDataset(
        torch.from_numpy(data_set[:train_size, ...]).float(),
        torch.from_numpy(label_sets[0][:train_size, ...]).float(),
        torch.from_numpy(label_sets[1][:train_size, ...]).long())
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

def preparation_first_stage_task(data_set, label_sets):
    '''
    Generate the corresponding stage1 difficulty set and dataset loader for
    all seismic gathers in advance, by this way, saving overhead

    :param data_set:    seismic gathers
    :param label_sets:  velocity models
    :return:            two datasets loader
    '''

    print('Pre-processing to obtain all data for the first stage (space-for-time).')

    dataset_forStage1 = np.zeros([train_size, inchannels, data_dim[0], data_dim[1]])

    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        middle_shot_id = 15
        first_p = 9
        second_p = 18
    else:
        middle_shot_id = 2
        first_p = 2
        second_p = 4
    for eachData in range(train_size):
        middle_shot = data_set[eachData, middle_shot_id, :, :]
        middle_shot_with_noise = add_gasuss_noise(middle_shot)
        middle_shot_magnified = magnify_amplitude_fornumpy(middle_shot.copy())
        for j in range(second_p, inchannels):
            dataset_forStage1[eachData, j, :, :] = middle_shot
        for j in range(first_p, second_p):
            dataset_forStage1[eachData, j, :, :] = middle_shot_magnified
        for j in range(0, first_p):
            dataset_forStage1[eachData, j, :, :] = middle_shot_with_noise


    # Training set
    seis_and_vm_forStage1 = data_utils.TensorDataset(torch.from_numpy(dataset_forStage1[:train_size, ...]).float(),
                                                     torch.from_numpy(label_sets[0][:train_size, ...]).float(),
                                                     torch.from_numpy(label_sets[1][:train_size, ...]).long())
    seis_and_vm_loader_forStage1 = data_utils.DataLoader(seis_and_vm_forStage1,
                                                         batch_size=train_batch_size,
                                                         pin_memory=True,
                                                         shuffle=True)

    return seis_and_vm_loader_forStage1

def custom_difficulty_measurer(seismic_gathers, stage_keyword):
    '''
    Match different difficulty states of the input seismic data

    :param seismic_gathers: Seismic gathers
    :param stage_keyword:   The selected difficulty keyword
    :return:
    '''

    # Splitting the seismic gathers
    if dataset_name in ['SEGSalt', 'SEGSimulation']:
        middle_shot_id = 15
        first_p = 9
        second_p = 18
    else:
        middle_shot_id = 2
        first_p = 2
        second_p = 4

    temp_gathers = np.zeros([train_batch_size, inchannels, data_dim[0], data_dim[1]])                                   # numpy

    if stage_keyword == "stage1":
        '''
        Take the data of the middle shot of the seismic gathers
        Then generate its noise map and amplification map
        '''
        for eachBatch in range(train_batch_size):
            middle_shot = seismic_gathers[eachBatch, middle_shot_id, :, :]
            middle_shot_with_noise = add_gasuss_noise(middle_shot)
            middle_shot_magnified = magnify_amplitude_fortensor(middle_shot.clone())
            for j in range(second_p, inchannels):
                temp_gathers[eachBatch, j, :, :] = middle_shot
            for j in range(first_p, second_p):
                temp_gathers[eachBatch, j, :, :] = middle_shot_magnified
            for j in range(0, first_p):
                temp_gathers[eachBatch, j, :, :] = middle_shot_with_noise

        temp_gathers = torch.from_numpy(temp_gathers).float()                                                           # numpy->tensor

    elif stage_keyword == "stage2":
        '''
        Take the data of the middle shot of the seismic gathers
        Then Make an equal number of n copies of this data
        '''
        for eachBatch in range(train_batch_size):
            middle_shot = seismic_gathers[eachBatch, middle_shot_id, :, :]
            for eachChannel in range(inchannels):
                temp_gathers[eachBatch, eachChannel, :, :] = middle_shot

        temp_gathers = torch.from_numpy(temp_gathers).float()                                                           # numpy->tensor

    elif stage_keyword in ["stage3", "no settings"]:
        '''
        Training using the seismic gathers
        '''
        temp_gathers = seismic_gathers                                                                                  # numpy->tensor

    else:
        print("The stage you selected does not exist.")
        exit(0)

    return temp_gathers

def train_for_one_stage(cur_epochs, model, training_loader, optimizer, key_word = "CLstage1", stage_keyword = "stage1", model_type = "DDNet"):
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
    :return:                Model save path and runtime
    '''

    loss_of_stage = 0.0
    last_model_save_path = ""
    step = int(train_size / train_batch_size)       # Total number of batches to train
    save_times = 2                                  # How many times do I need to save the intermediate results of the model
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

            # Determine how difficult a task should be by using a difficulty measurer
            if stage_keyword in ["stage3", "no settings"]:
                pass
            else:
                images = custom_difficulty_measurer(images, stage_keyword)

            # Load to GPU
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                contours_labels = contours_labels.cuda(non_blocking=True)

            # Gradient cache clearing
            optimizer.zero_grad()
            criterion = LossDDNet(weights=[1, 1e6] if model_type == "DDNet" else [1, 1], entropy_weight=[1, 1])
            outputs = model(images, model_dim)
            loss = criterion(outputs[0], outputs[1], labels, contours_labels)

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
            loss_of_stage = np.append(loss_of_stage, loss_of_epoch / i)

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

    save_results(loss=loss_of_stage, epochs=cur_epochs, save_path=results_dir,
                 xtitle='Num. of epochs', ytitle='Num. of epochs',
                 title='Training Loss {}'.format(key_word))

    return last_model_save_path, training_time

def curriculum_learning_training(model_type):
    '''
    Curriculum learning
    '''
    
    all_training_time = 0
    priori_model_src = None
    training_loader, seismic_gathers, velocity_models = load_dataset()

    ###########
    # Stage 1 #
    ###########
    if firststage_epochs != 0:
        dd_net_init, device, optimizer = determine_network(priori_model_src, model_type=model_type)
        # Because it takes time to set up the structure of the first task one by one during training, we directly
        # construct a loader that contains all the gathers. Thus realizing space for time.
        training_loader_stage1 = preparation_first_stage_task(seismic_gathers, velocity_models)  # Construct stage1 specialized loader
        priori_model_src, training_time = train_for_one_stage(firststage_epochs, dd_net_init, training_loader_stage1, optimizer,
                                               key_word = "CLStage1",
                                               stage_keyword = "no settings",
                                               model_type = model_type)
        all_training_time += training_time
        # Seismic gathers are very space-consuming, stage1's specialized
        # seismic gathers need to be cleaned up when they are used up
        del training_loader_stage1

    ###########
    # Stage 2 #
    ###########
    if secondstage_epochs != 0:
        dd_net_st1, device, optimizer = determine_network(priori_model_src, model_type=model_type)
        priori_model_src, training_time = train_for_one_stage(secondstage_epochs, dd_net_st1, training_loader, optimizer,
                                               key_word = "CLStage2",
                                               stage_keyword = "stage2",
                                               model_type = model_type)
        all_training_time += training_time

    ###########
    # Stage 3 #
    ###########
    dd_net_st2, device, optimizer = determine_network(priori_model_src, model_type=model_type)
    priori_model_src, training_time = train_for_one_stage(thirdstage_epochs, dd_net_st2, training_loader, optimizer,
                                           key_word="CLStage3",
                                           stage_keyword="stage3",
                                           model_type = model_type)
    all_training_time += training_time
    print("training runtime: {}s".format(all_training_time))

    loss_mat_dir = results_dir + "Training Loss CLStage1.mat"
    loss_stage1 = scipy.io.loadmat(loss_mat_dir)['loss'][0][0:]
    loss_mat_dir = results_dir + "Training Loss CLStage2.mat"
    loss_stage2 = scipy.io.loadmat(loss_mat_dir)['loss'][0][1:]
    loss_mat_dir = results_dir + "Training Loss CLStage3.mat"
    loss_stage3 = scipy.io.loadmat(loss_mat_dir)['loss'][0][1:]

    save_results(loss=np.hstack([loss_stage1, loss_stage2, loss_stage3]),
                epochs=epochs, save_path=results_dir, xtitle='Num. of epochs',
                ytitle='Num. of epochs', title='Loss (all stage)', is_show=True)

def transfer_learning_training(model_type):
    '''
    Transfer learning
    '''

    priori_model_src = models_dir + "CL_SEGSimulation_DDNet.pkl"
    loader, seismic_gathers, velocity_models = load_dataset()

    dd_net, device, optimizer = determine_network(priori_model_src, model_type=model_type)
    _, training_time = train_for_one_stage(thirdstage_epochs, dd_net, loader, optimizer,
                        key_word="Tranfer Learning",
                        stage_keyword="no settings",
                        model_type=model_type)
    print("training runtime: {}s".format(training_time))


if __name__ == "__main__":
    curriculum_learning_training(model_type="DDNet")
    # transfer_learning_training()
