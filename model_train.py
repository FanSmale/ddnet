# -*- coding: utf-8 -*-
"""
Curriculum Learning

Created on Feb 2023

@author: Xing-Yi Zhang (Zhangzxy20004182@163.com)

"""

from func.utils import *
from func.datasets_reader import *


def determine_network(external_model_src = None):
    '''
    Request a network object and import an external network, or create an initialized network

    :param external_model_src:  External pkl file path
    :return:                    A triplet: model object, GPU environment object and optimizer
    '''
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    gpus = [0]

    # Network initialization
    if dataset_name in ['SEGReal', 'SEGSimulation']:
        dd_net = DDNetModel(n_classes=classes,
                            in_channels=inchannels,
                            is_deconv=True,
                            is_batchnorm=True)
    else:
        dd_net = DDNet70Model(n_classes=classes,
                            in_channels=inchannels,
                            is_deconv=True,
                            is_batchnorm=True)

    # Inherit the previous network structure
    if external_model_src != None:
        dd_net = model_reader(net=dd_net, device=device, save_src=external_model_src)

    # Allocate GPUs and set optimizers
    if torch.cuda.is_available():
        dd_net = torch.nn.DataParallel(dd_net, device_ids=gpus).cuda()
    optimizer = torch.optim.Adam(dd_net.parameters(), lr=learning_rate)

    return dd_net, device, optimizer

def load_dataset():
    '''
    Load the training data according to the parameters in "param_config"

    :return:    A triplet: datasets loader, seismic gathers and velocity models
    '''

    print("---------------------------------")
    print("· Loading the datasets...")
    if dataset_name in ['SEGReal', 'SEGSimulation']:
        data_set, label_sets = batch_read_matfile(data_dir, data_dim, model_dim, 1, train_size, "train")
        data_set = data_set.reshape(train_size, inchannels, data_dim[0], data_dim[1])
        label_sets[0] = label_sets[0].reshape(train_size, classes, model_dim[0], model_dim[1])
        label_sets[1] = label_sets[1].reshape(train_size, classes, model_dim[0], model_dim[1])
    else:
        data_set, label_sets = batch_read_npyfile(data_dir, 1, train_size // 500, "train")

    print("· Number of seismic gathers included in the training set: {}.".format(train_size))
    print("· Dimensions of seismic data: ({},{},{},{}).".format(train_size, inchannels, data_dim[0], data_dim[1]))
    print("· Dimensions of velocity model: ({},{},{},{}).".format(train_size, classes, model_dim[0], model_dim[1]))
    print("---------------------------------")

    seis_and_vm = data_utils.TensorDataset(torch.from_numpy(data_set).float(),
                                           torch.from_numpy(label_sets[0]).float(),
                                           torch.from_numpy(label_sets[1]).float())
    seis_and_vm_loader = data_utils.DataLoader(seis_and_vm, batch_size=train_batch_size, shuffle=True)

    return seis_and_vm_loader, data_set, label_sets

def preparation_first_stage_task(data_set, label_sets):
    '''
    Generate the corresponding stage1 difficulty set and dataset loader for
    all seismic gathers in advance, by this way, saving overhead

    :param data_set:    seismic gathers
    :param label_sets:  velocity models
    :return:            A pair: datasets load and seismic gathers for stage1
    '''

    print('Pre-processing to obtain all data for the first stage (space-for-time).')
    dataset_forStage1 = np.zeros([train_size, inchannels, data_dim[0], data_dim[1]])

    if dataset_name in ['SEGReal', 'SEGSimulation']:
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

    seis_and_vm_forStage1 = data_utils.TensorDataset(torch.from_numpy(dataset_forStage1).float(),
                                                     torch.from_numpy(label_sets[0]).float(),
                                                     torch.from_numpy(label_sets[1]).float())
    seis_and_vm_loader_forStage1 = data_utils.DataLoader(seis_and_vm_forStage1, batch_size=train_batch_size, shuffle=True)

    return seis_and_vm_loader_forStage1, dataset_forStage1

def custom_difficulty_measurer(seismic_gathers, stage_keyword = "no settings"):
    '''
    Match different difficulty states of the input seismic data

    :param seismic_gathers: Seismic gathers
    :param stage_keyword:   The selected difficulty keyword
    :return:
    '''

    # Splitting the seismic gathers
    if dataset_name in ['SEGReal', 'SEGSimulation']:
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

    elif stage_keyword == "stage3" or "no settings":
        '''
        Training using the seismic gathers
        '''
        temp_gathers = seismic_gathers                                                                                  # numpy->tensor

    else:
        print("The stage you selected does not exist.")
        exit(0)

    return temp_gathers



def train_for_one_stage(cur_epochs, model, dataset_load, optimizer, key_word = "CLstage1", stage_keyword = "stage1"):
    '''
    Training for designated epochs

    :param cur_epochs:      Designated epochs
    :param model:           Network model objects to be used for training
    :param dataset_load:    Dataset loader to be fed into the network
    :param optimizer:       Optimizer
    :param key_word:        After the training, the keywords will be saved to the model
    :param stage_keyword:   The selected difficulty keyword (set "no settings" to ignore CL)
    :return:                Model save path
    '''

    loss_of_stage = 0.0
    last_model_save_path = ""
    step = int(train_size / train_batch_size)       # Total number of batches to train
    save_times = 1                                  # How many times do I need to save the intermediate results of the model
    save_epoch = cur_epochs // save_times

    model_save_name = "{}_{}_TrSize{}_AllEpo{}".format(dataset_name, key_word, train_size, cur_epochs)

    for epoch in range(cur_epochs):
        # Training for the current epoch
        loss_of_epoch = 0.0
        cur_node_time = time.time()
        for i, (images, labels, contours_labels) in enumerate(dataset_load):


            iteration = epoch * step + i + 1
            model.train()

            # Determining training batches
            images = images.view(train_batch_size, inchannels, data_dim[0], data_dim[1])
            labels = labels.view(train_batch_size, classes, model_dim[0], model_dim[1])
            contours_labels = contours_labels.view(train_batch_size, classes, model_dim[0], model_dim[1]).long()

            # Determine how difficult a task should be by using a difficulty measurer
            images = custom_difficulty_measurer(images, stage_keyword=stage_keyword)

            # Load to GPU
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                contours_labels = contours_labels.cuda(non_blocking=True)

            # Gradient cache clearing
            optimizer.zero_grad()

            # Forward
            outputs = model(images, model_dim)

            # Set the loss function and calculate it
            criterion = LossDDNet(weights=([1, 1e6] if dataset_name in ['SEGReal', 'SEGSimulation'] else [1, 1]), entropy_weight=[1, 1])
            loss = criterion(outputs[0], outputs[1], labels, contours_labels)
            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while training')
            loss_of_epoch += loss.item()            # Count the sum of all losses of the current epoch

            # Loss backward propagation
            loss.backward()

            # Optimize
            optimizer.step()

            if iteration % display_step == 0:
                print('[{}] Firststage_Epochs: {}/{}, Iteration: {}/{} --- Training Loss:{:.6f}'
                      .format(key_word, epoch + 1, cur_epochs, iteration, step * cur_epochs, loss.item()))

        ################################
        # The end of the current epoch #
        ################################
        if (epoch + 1) % 1 == 0:

            # Calculate the average loss of the current epoch
            print('[{}] Firststage_Epochs: {:d} finished ! Loss: {:.5f}'
                  .format(key_word, epoch + 1, loss_of_epoch / i))

            # Include the average loss in the array belonging to the current stage
            loss_of_stage = np.append(loss_of_stage, loss_of_epoch / i)

            # Statistics of the time spent in a epoch
            time_elapsed = time.time() - cur_node_time
            print('[{}] Firststage_Epochs consuming time: {:.0f}m {:.0f}s'
                  .format(key_word, time_elapsed // 60, time_elapsed % 60))

        #########################################################################
        # When it reaches the point where intermediate results can be stored... #
        #########################################################################
        if (epoch + 1) % save_epoch == 0:
            last_model_save_path = models_dir + model_save_name + '_CurEpo' + str(epoch + 1) + '.pkl'
            torch.save(model.state_dict(), last_model_save_path)
            print('[' + key_word + '] Trained model saved: %d percent completed' % int((epoch + 1) * 100 / cur_epochs))

    save_results(loss=loss_of_stage, epochs=cur_epochs, save_path=results_dir,
                 xtitle='Num. of epochs', ytitle='Num. of epochs', title='Training Loss {}'.format(key_word))

    return last_model_save_path

def curriculum_learning_training():
    '''

    Curriculum learning
    (Not applicable to SEGReal)

    '''
    if dataset_name == 'SEGReal':
        print("SEGReal is for transfer learning, not curriculum learning")
        exit(0)
    # priori_model_src = "H:\Study and programming\My Paper Code\DD-Net\models\FlatVelAModel\FlatVelA_CLStage1_TrSize1000_AllEpo3_CurEpo3.pkl"
    priori_model_src = None
    loader, seismic_gathers, velocity_models = load_dataset()

    ###########
    # Stage 1 #
    ###########
    dd_net_init, device, optimizer = determine_network(priori_model_src)                                       # Initialize model settings
    # Because it takes time to set up the structure of the first task one by one during training, we directly
    # construct a loader that contains all the gathers. Thus realizing space for time.
    stage1_specialized_loader, stage1_seismic_gathers = preparation_first_stage_task(seismic_gathers, velocity_models)  # Construct stage1 specialized loader
    priori_model_src = train_for_one_stage(firststage_epochs, dd_net_init, stage1_specialized_loader, optimizer,
                                           key_word = "CLStage1",
                                           stage_keyword = "no settings")
    # Seismic gathers are very space-consuming, stage1's specialized
    # seismic gathers need to be cleaned up when they are used up
    del stage1_specialized_loader
    del stage1_seismic_gathers


    ###########
    # Stage 2 #
    ###########
    dd_net_st1, device, optimizer = determine_network(priori_model_src)
    priori_model_src = train_for_one_stage(secondstage_epochs, dd_net_st1, loader, optimizer,
                                           key_word = "CLStage2",
                                           stage_keyword = "stage2")

    ###########
    # Stage 3 #
    ###########
    dd_net_st2, device, optimizer = determine_network(priori_model_src)
    priori_model_src = train_for_one_stage(thirdstage_epochs, dd_net_st2, loader, optimizer,
                                           key_word="CLStage3",
                                           stage_keyword="stage3")

    loss_mat_dir = results_dir + "Training Loss CLStage1.mat"
    loss_stage1 = scipy.io.loadmat(loss_mat_dir)['loss'][0][0:]
    loss_mat_dir = results_dir + "Training Loss CLStage2.mat"
    loss_stage2 = scipy.io.loadmat(loss_mat_dir)['loss'][0][1:]
    loss_mat_dir = results_dir + "Training Loss CLStage3.mat"
    loss_stage3 = scipy.io.loadmat(loss_mat_dir)['loss'][0][1:]


    save_results(loss=np.hstack([loss_stage1, loss_stage2, loss_stage3]),       # The first element of the array is 0
                epochs=epochs, save_path=results_dir, xtitle='Num. of epochs',
                ytitle='Num. of epochs', title='Training Loss (all stage)', is_show=True)

def transfer_learning_training():
    '''

    Transfer learning
    (Only applicable to SEGReal)

    '''
    if dataset_name != 'SEGReal':
        print("Only SEGReal is to be used for transfer learning, the rest of the dataset cannot be used for this")
        exit(0)

    priori_model_src = main_dir + "models/SEGSimulationModel/SEGSimulation_CL.pkl"        # The base network must be pre-trained by SEGSimulation
    loader, seismic_gathers, velocity_models = load_dataset()

    ###########
    # Stage 3 #
    ###########
    dd_net, device, optimizer = determine_network(priori_model_src)
    train_for_one_stage(thirdstage_epochs, dd_net, loader, optimizer, key_word="Tranfer Learning", stage_keyword="no settings")


if __name__ == "__main__":
    # curriculum_learning_training()
    transfer_learning_training()







