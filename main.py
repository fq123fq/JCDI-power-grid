import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

from utils.util import find_max_epoch, print_size, training_loss, calc_diffusion_hyperparams


from mNN.powergrid_hybrid import PowerGridHybrid
from mNN.resnet_1d import ResNetEncoder

#import matplotlib.pyplot as plt

from utils.util import sampling
from sklearn.metrics import mean_squared_error
from statistics import mean

import time
from ema_pytorch import EMA
import mat73


loss_all = []

RESNET_BLOCK_SPECS = [
    # curr shape: (2, 192),128
    ('stem', {
        'features' : 32, 'kernel_size' : 4, 'padding' : 0, 'stride' : 4
    }),
    # curr shape: (32, 48),32
    ('resnet', 1),
    #('stem', {
    #    'features' : 64, 'kernel_size' : 4, 'padding' : 1, 'stride' : 2
    #}),
    ## curr shape: (64, 24),16
    #('resnet', 1),
    ('stem', {
        'features' : 128, 'kernel_size' : 4, 'padding' : 1, 'stride' : 8
    }),
    # curr shape: (128, 12),8
    ('resnet', 1),
    ('stem', {
        'features' : 256, 'kernel_size' : 4, 'padding' : 1, 'stride' : 4
    }),
    # curr shape: (256, 6),4
    #('resnet', 1),
    #('stem', {
    #    'features' : 512, 'kernel_size' : 4, 'padding' : 1, 'stride' : 2
    #}),
    ## curr shape: (512, 3),2
]



def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          use_model):
          
    # generate experiment (local) path

    #output_directory_fq=output_directory
    
    local_path = "n_iters{}_T{}_beta0{}_betaT{}_data300000".format(train_config["n_iters"],
                                              diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()
#            diffusion_hyperparams[key] = diffusion_hyperparams[key]
    

    ##########################################data preparation##########################################
    import scipy.io as scio
    #data = scio.loadmat(trainset_config['train_data_path'])
    data = mat73.loadmat(trainset_config['train_data_path'])
    para_for_gene_eventmore_merge_trs_nom = data['para_for_gene_eventmore_merge_trs_nom']
    traj_p_gene_eventmore_merge_trs = data['traj_p_gene_eventmore_merge_trs']
    traj_q_gene_eventmore_merge_trs = data['traj_q_gene_eventmore_merge_trs']


    #   training_data = np.split(training_data, 42, 0)
    para_for_gene_eventmore_merge_trs_nom = torch.from_numpy(para_for_gene_eventmore_merge_trs_nom).float()
    traj_p_gene_eventmore_merge_trs = torch.from_numpy(traj_p_gene_eventmore_merge_trs).float()
    traj_q_gene_eventmore_merge_trs = torch.from_numpy(traj_q_gene_eventmore_merge_trs).float()

    num_sample_all=300000
    num_sample_train=250000
    num_sample_test=50000

    para_length_fq=30
    traj_length_fq=512


    print("traj_p_gene_eventmore_merge_trs.shape",traj_p_gene_eventmore_merge_trs.shape)
    n_events=traj_p_gene_eventmore_merge_trs.shape[1]
    print("n_events",n_events)

    #training_traj_pq_eventmore=torch.empty((num_sample_all,n_events,2,traj_length_fq))
    traj_p_mins=[]
    traj_p_maxs=[]
    traj_q_mins=[]
    traj_q_maxs=[]
    for ii in range(n_events):
        traj_p_gene_eventnow_merge_trs=traj_p_gene_eventmore_merge_trs[:,ii,:].unsqueeze(1)
        traj_q_gene_eventnow_merge_trs=traj_q_gene_eventmore_merge_trs[:,ii,:].unsqueeze(1)
        print("traj_p_gene_eventnow_merge_trs.shape",traj_p_gene_eventnow_merge_trs.shape)
        traj_p_min_min=torch.min(traj_p_gene_eventmore_merge_trs)
        traj_p_max_max=torch.max(traj_p_gene_eventmore_merge_trs)
        traj_q_min_min=torch.min(traj_q_gene_eventmore_merge_trs)
        traj_q_max_max=torch.max(traj_q_gene_eventmore_merge_trs)

        traj_p_eventnow_min=torch.min(torch.min(traj_p_gene_eventnow_merge_trs))
        traj_p_eventnow_max=torch.max(torch.max(traj_p_gene_eventnow_merge_trs))
        traj_q_eventnow_min=torch.min(torch.min(traj_q_gene_eventnow_merge_trs))
        traj_q_eventnow_max=torch.max(torch.max(traj_q_gene_eventnow_merge_trs))

        print("traj_p_eventnow_min",traj_p_eventnow_min)
        print("traj_q_eventnow_min",traj_q_eventnow_min)
        

        traj_p_mins.append(torch.min(torch.min(traj_p_gene_eventnow_merge_trs)))
        traj_p_maxs.append(torch.max(torch.max(traj_p_gene_eventnow_merge_trs)))
        traj_q_mins.append(torch.min(torch.min(traj_q_gene_eventnow_merge_trs)))
        traj_q_maxs.append(torch.max(torch.max(traj_q_gene_eventnow_merge_trs)))
        

        traj_p_gene_eventnow_merge_trs_nom=(traj_p_gene_eventnow_merge_trs - traj_p_eventnow_min) / (traj_p_eventnow_max - traj_p_eventnow_min)
        traj_q_gene_eventnow_merge_trs_nom=(traj_q_gene_eventnow_merge_trs - traj_q_eventnow_min) / (traj_q_eventnow_max - traj_q_eventnow_min)


        traj_pq_gene_eventnow_merge_trs_nom = torch.cat([traj_p_gene_eventnow_merge_trs_nom, traj_q_gene_eventnow_merge_trs_nom], dim=1)
        print("traj_pq_gene_eventnow_merge_trs_nom.shape",traj_pq_gene_eventnow_merge_trs_nom.shape)
        print("torch.unsqueeze(traj_pq_gene_eventnow_merge_trs_nom,1).shape",torch.unsqueeze(traj_pq_gene_eventnow_merge_trs_nom,1).shape)
        if ii==0:
            traj_pq_gene_eventmore_merge_trs_nom=torch.unsqueeze(traj_pq_gene_eventnow_merge_trs_nom,1)
        else:
            traj_pq_gene_eventmore_merge_trs_nom=torch.cat([traj_pq_gene_eventmore_merge_trs_nom,torch.unsqueeze(traj_pq_gene_eventnow_merge_trs_nom,1)],dim=1)
            
        #training_traj_pq_eventmore=torch.stack (training_traj_pq_eventmore,torch.unsqueeze(traj_pq_gene_eventnow_merge_trs,1),dim=1)
        print("traj_pq_gene_eventmore_merge_trs_nom.shape",traj_pq_gene_eventmore_merge_trs_nom.shape)

    traj_pq_gene_eventmore_merge_trs_nom_each=torch.unbind(traj_pq_gene_eventmore_merge_trs_nom,dim=1)
    print("len(traj_pq_gene_eventmore_merge_trs_nom_each)",len(traj_pq_gene_eventmore_merge_trs_nom_each))
    print("traj_pq_gene_eventmore_merge_trs_nom_each[0].shape",traj_pq_gene_eventmore_merge_trs_nom_each[0].shape)

    print("traj_p_min_min",traj_p_min_min)
    print("traj_p_max_max",traj_p_max_max)
    print("traj_q_min_min",traj_q_min_min)
    print("traj_q_max_max",traj_q_max_max)

    print("traj_p_mins",traj_p_mins)
    print("traj_p_mins[ii]",traj_p_mins[ii])
    print("traj_p_maxs",traj_p_maxs)
    print("traj_q_mins",traj_q_mins)
    print("traj_q_maxs",traj_q_maxs)



    para_for_gene_eventmore_merge_trs_nom = torch.reshape(para_for_gene_eventmore_merge_trs_nom, (num_sample_all, 1, para_length_fq))

    training_para = para_for_gene_eventmore_merge_trs_nom[:num_sample_train,:,:]
    testing_para = para_for_gene_eventmore_merge_trs_nom[num_sample_train:num_sample_all,:,:]

    print("training_para size:", training_para.shape)
    print("testing_para size:", testing_para.shape)

    print("training_para:", training_para)


    training_traj_pq=traj_pq_gene_eventmore_merge_trs_nom[:num_sample_train,:,:,:]

    testing_traj_pq=traj_pq_gene_eventmore_merge_trs_nom[num_sample_train:num_sample_all,:,:,:]
    #testing_traj_pq=testing_traj_pq[:num_sample_test, :,:]


    print("training_traj_pq size:", training_traj_pq.shape)
    print("testing_traj_pq size:", testing_traj_pq.shape)


    import torch.utils.data as Data
    from torch.utils.data import DataLoader
    # create dataloader
    #torch_dataset = Data.TensorDataset(training_para,training_traj_pq,training_time)
    #torch_test_dataset = Data.TensorDataset(testing_para,testing_traj_pq,testing_time)

    torch_dataset = Data.TensorDataset(training_para,training_traj_pq)
    #torch_test_dataset = Data.TensorDataset(testing_para,testing_traj_pq)
    torch_train_sp_dataset = Data.TensorDataset(training_para[::1,:,:],training_traj_pq[::1,:,:,:])
    torch_test_dataset = Data.TensorDataset(testing_para[::1,:,:],testing_traj_pq[::1,:,:,:])


    train_batch_size=128
    test_batch_size=128


    loader = DataLoader(torch_dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 0)
    train_sp_loader = DataLoader(torch_train_sp_dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 0)
    test_loader = DataLoader(torch_test_dataset, batch_size = test_batch_size, shuffle = True, pin_memory = True, num_workers = 0)


    print('Data loaded')

    ####real  data preparation####
    #import scipy.io as scio
    #data = scio.loadmat(trainset_config['real_sp_data_path'])
    data = mat73.loadmat(trainset_config['real_sp_data_path'])
    para_for_gene_eventmore_merge_trs_nom_real = data['para_real_eventmore_merge_trs_nom']
    traj_p_gene_eventmore_merge_trs_real = data['traj_p_real_eventmore_merge_trs']
    traj_q_gene_eventmore_merge_trs_real = data['traj_q_real_eventmore_merge_trs']


    para_for_gene_eventmore_merge_trs_nom_real = torch.from_numpy(para_for_gene_eventmore_merge_trs_nom_real).float()
    traj_p_gene_eventmore_merge_trs_real = torch.from_numpy(traj_p_gene_eventmore_merge_trs_real).float()
    traj_q_gene_eventmore_merge_trs_real = torch.from_numpy(traj_q_gene_eventmore_merge_trs_real).float()

    num_sample_real_all=5000

    para_length_fq=30
    traj_length_fq=512


    for ii in range(n_events):
        traj_p_gene_eventnow_merge_trs_real=traj_p_gene_eventmore_merge_trs_real[:,ii,:].unsqueeze(1)
        traj_q_gene_eventnow_merge_trs_real=traj_q_gene_eventmore_merge_trs_real[:,ii,:].unsqueeze(1)
        print("traj_p_gene_eventnow_merge_trs_real.shape",traj_p_gene_eventnow_merge_trs_real.shape)
        

        traj_p_gene_eventnow_merge_trs_nom_real=(traj_p_gene_eventnow_merge_trs_real - traj_p_mins[ii]) / (traj_p_maxs[ii] - traj_p_mins[ii])
        traj_q_gene_eventnow_merge_trs_nom_real=(traj_q_gene_eventnow_merge_trs_real - traj_q_mins[ii]) / (traj_q_maxs[ii] - traj_q_mins[ii])

        traj_pq_gene_eventnow_merge_trs_nom_real = torch.cat([traj_p_gene_eventnow_merge_trs_nom_real, traj_q_gene_eventnow_merge_trs_nom_real], dim=1)
        print("traj_pq_gene_eventnow_merge_trs_nom_real.shape",traj_pq_gene_eventnow_merge_trs_nom_real.shape)
        print("torch.unsqueeze(traj_pq_gene_eventnow_merge_trs_nom_real,1).shape",torch.unsqueeze(traj_pq_gene_eventnow_merge_trs_nom_real,1).shape)
        if ii==0:
            traj_pq_gene_eventmore_merge_trs_nom_real=torch.unsqueeze(traj_pq_gene_eventnow_merge_trs_nom_real,1)
        else:
            traj_pq_gene_eventmore_merge_trs_nom_real=torch.cat([traj_pq_gene_eventmore_merge_trs_nom_real,torch.unsqueeze(traj_pq_gene_eventnow_merge_trs_nom_real,1)],dim=1)
            
        #training_traj_pq_eventmore=torch.stack (training_traj_pq_eventmore,torch.unsqueeze(traj_pq_gene_eventnow_merge_trs,1),dim=1)
        print("traj_pq_gene_eventmore_merge_trs_nom_real.shape",traj_pq_gene_eventmore_merge_trs_nom_real.shape)

    traj_pq_gene_eventmore_merge_trs_nom_each_real=torch.unbind(traj_pq_gene_eventmore_merge_trs_nom_real,dim=1)
    print("len(traj_pq_gene_eventmore_merge_trs_nom_each_real)",len(traj_pq_gene_eventmore_merge_trs_nom_each_real))
    print("traj_pq_gene_eventmore_merge_trs_nom_each_real[0].shape",traj_pq_gene_eventmore_merge_trs_nom_each_real[0].shape)


    para_for_gene_eventmore_merge_trs_nom_real = torch.reshape(para_for_gene_eventmore_merge_trs_nom_real, (num_sample_real_all, 1, para_length_fq))

    real_para = para_for_gene_eventmore_merge_trs_nom_real[:num_sample_real_all,:,:]

    print("real_para size:", real_para.shape)

    real_traj_pq=traj_pq_gene_eventmore_merge_trs_nom_real[:num_sample_real_all,:,:,:]

    #testing_traj_pq=testing_traj_pq[:num_sample_test, :,:]

    print("real_traj_pq size:", real_traj_pq.shape)


    import torch.utils.data as Data
    from torch.utils.data import DataLoader
    # create dataloader
    #torch_dataset = Data.TensorDataset(training_para,training_traj_pq,training_time)
    #torch_test_dataset = Data.TensorDataset(testing_para,testing_traj_pq,testing_time)

    torch_real_dataset = Data.TensorDataset(real_para,real_traj_pq)

    real_batch_size=128

    real_sp_loader = DataLoader(torch_real_dataset, batch_size = real_batch_size, shuffle = True, pin_memory = True, num_workers = 0)

    # predefine model
    if use_model == 0:
    #elif use_model == 1:
        net = PowerGridHybrid(
          #traj_shape = (2, 192),
         traj_shape = (2, traj_length_fq),
         n_events=n_events,
         n_params   = para_length_fq,
         resnet_params = {
             'block_specs' : RESNET_BLOCK_SPECS,
             'activ'   : 'relu',
             'norm'    : 'instance-1d',
             'rezero'  : True,
         },
         transformer_params = {
             'features'     : 256, # 512
             'features_ffn' : 512,  #2048
             'n_heads'      : 4,  #8
             'n_layers'     : 3,  #8
             'norm_first'   : True,
             'time_embed'   : 'sin',
         },
         ).cuda()
    else:
        print('Model chosen not available.')
    print_size(net)

    ema = EMA(
    net,
    beta = 0.995,              # exponential moving average factor
    update_after_step = 100,#100,    # only after this number of .update() calls will it start updating
    update_every = 10,#10,          # how often to actually update, to save on compute (updates every 10th .update() call)
    )


    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            ema.load_state_dict(checkpoint['ema_state_dict'])

            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')


    
    loss_all = []
    test_loss_all=[]
    epoch_train_loss_all=[]
    epoch_test_loss_all=[]
    para_error_test_all=[]
    n_iters_all=[]
    n_epoches_all=[]
    
    # training
    T1 = time.time()

    n_iter = ckpt_iter + 1
    n_epoch=0
    while n_iter < n_iters + 1:
        n_epoch=n_epoch+1

        #for batch in training_data:
        net.train()
        ema.ema_model.train()
        epoch_train_loss=0

        for i, (batch_para, batch_traj) in enumerate(loader):
            batch_para=batch_para.cuda()
            batch_traj=batch_traj.cuda()


            # back-propagation
            optimizer.zero_grad()
            X = batch_para, batch_traj

            loss = training_loss(net, nn.MSELoss(), X, diffusion_hyperparams)

            loss.backward()
            optimizer.step()
            ema.update()

            epoch_train_loss=epoch_train_loss+loss.item()


            if n_iter % iters_per_logging == 0:
                loss_all.append([loss.item()])

            if n_iter % (iters_per_logging*10) == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss.item()))
                np.save(os.path.join(output_directory, 'loss_all.npy'), loss_all)

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'ema_state_dict': ema.state_dict()},
                           os.path.join(output_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

            n_iter += 1

        epoch_train_loss=epoch_train_loss/ len(loader)
        epoch_train_loss_all.append([epoch_train_loss])
        np.save(os.path.join(output_directory, 'epoch_train_loss_all.npy'), epoch_train_loss_all)

        # test
        if (n_epoch-1) % 50 == 0:
            print("calculate test loss")
            net.eval()
            ema.ema_model.eval()
            epoch_test_loss=0
            with torch.no_grad():
              for i, (batch_para, batch_traj) in enumerate(test_loader):
                  #calculate test loss
                  batch_para=batch_para.cuda()
                  batch_traj=batch_traj.cuda()
   
                  X = batch_para, batch_traj
                  #X = batch, None, mask, loss_mask
                  test_loss = training_loss(net, nn.MSELoss(), X, diffusion_hyperparams)

                  epoch_test_loss=epoch_test_loss+test_loss.item()

                  '''
                  #sample test traj and calculate parameter error
                  num_samples = batch_para.size(0)
                  sample_length = batch_para.size(2)
                  sample_channels = batch_para.size(1)
                  predict_para = sampling(ema.ema_model, (num_samples, sample_channels, sample_length),
                                                 diffusion_hyperparams,
                                                 cond=batch_traj)
                                       
                  if i == 0:
                      predict_para_test_all = predict_para
                  else:
                      predict_para_test_all = torch.cat([predict_para_test_all, predict_para], dim=0)

        

                  if i == 0:
                      original_para_test_all = batch_para
                  else:
                      original_para_test_all = torch.cat([original_para_test_all, batch_para], dim=0)
                  #np.save(os.path.join(output_directory, 'original_para_test_all_epoch{}.npy'.format(n_epoch)), original_para_test_all.detach().cpu().numpy())
                  '''

            epoch_test_loss=epoch_test_loss/ len(test_loader)
            epoch_test_loss_all.append([epoch_test_loss])
            print("iteration: {} \tepoch_train_loss: {}".format(n_iter, epoch_train_loss))
            print("iteration: {} \tepoch_test_loss: {}".format(n_iter, epoch_test_loss))
            n_iters_all.append([n_iter])
            n_epoches_all.append([n_epoch])
            np.savez(os.path.join(output_directory, 'epoch_train_test_losses_all'), epoch_train_loss_all=epoch_train_loss_all,epoch_test_loss_all=epoch_test_loss_all,n_iters_all=n_iters_all,n_epoches_all=n_epoches_all)

            #np.save(os.path.join(output_directory, 'predict_para_test_all_epoch{}.npy'.format(n_epoch)), predict_para_test_all.detach().cpu().numpy())
            #np.save(os.path.join(output_directory, 'original_para_test_all_epoch{}.npy'.format(n_epoch)), original_para_test_all.detach().cpu().numpy())

            '''
            with torch.no_grad():
                for i, (batch_para, batch_traj) in enumerate(real_sp_loader):
                    batch_para=batch_para.cuda()
                    batch_traj=batch_traj.cuda()

                    num_samples = batch_para.size(0)
                    sample_length = batch_para.size(2)
                    sample_channels = batch_para.size(1)

                    predict_para = sampling(ema.ema_model, (num_samples, sample_channels, sample_length),
                                                 diffusion_hyperparams,
                                                 cond=batch_traj)
                                       
                    if i == 0:
                        predict_para_real_sp_all = predict_para
                    else:
                        predict_para_real_sp_all = torch.cat([predict_para_real_sp_all, predict_para], dim=0)


                    if i == 0:
                        original_para_real_sp_all = batch_para
                    else:
                        original_para_real_sp_all = torch.cat([original_para_real_sp_all, batch_para], dim=0)


            np.save(os.path.join(output_directory, 'predict_para_real_sp_all_epoch{}.npy'.format(n_epoch)), predict_para_real_sp_all.detach().cpu().numpy())  
            np.save(os.path.join(output_directory, 'original_para_real_sp_all_epoch{}.npy'.format(n_epoch)), original_para_real_sp_all.detach().cpu().numpy())

            with torch.no_grad():
                for i, (batch_para, batch_traj) in enumerate(train_sp_loader):
                    batch_para=batch_para.cuda()
                    batch_traj=batch_traj.cuda()

                    num_samples = batch_para.size(0)
                    sample_length = batch_para.size(2)
                    sample_channels = batch_para.size(1)

                    predict_para = sampling(ema.ema_model, (num_samples, sample_channels, sample_length),
                                                 diffusion_hyperparams,
                                                 cond=batch_traj)
                                       
                    if i == 0:
                        predict_para_train_sp_all = predict_para
                    else:
                        predict_para_train_sp_all = torch.cat([predict_para_train_sp_all, predict_para], dim=0)


                    if i == 0:
                        original_para_train_sp_all = batch_para
                    else:
                        original_para_train_sp_all = torch.cat([original_para_train_sp_all, batch_para], dim=0)


            np.save(os.path.join(output_directory, 'predict_para_train_sp_all_epoch{}.npy'.format(n_epoch)), predict_para_train_sp_all.detach().cpu().numpy())  
            np.save(os.path.join(output_directory, 'original_para_train_sp_all_epoch{}.npy'.format(n_epoch)), original_para_train_sp_all.detach().cpu().numpy())
            '''

    print('Training finished.')

    #############################real sampling###########################################
    ema.ema_model.eval()

    print('sampling from real trajectories.')

    #for i, batch in enumerate(training_data):
    for i, (batch_para, batch_traj) in enumerate(real_sp_loader):
        batch_para=batch_para.cuda()
        batch_traj=batch_traj.cuda()

        num_samples = batch_para.size(0)
        sample_length = batch_para.size(2)
        sample_channels = batch_para.size(1)
        T3 = time.time()
        predict_para = sampling(ema.ema_model, (num_samples, sample_channels, sample_length),
                                       diffusion_hyperparams,
                                       cond=batch_traj)
                                       
        T4 = time.time()

        if i == 0:
            predict_para_real_all = predict_para
        else:
            predict_para_real_all = torch.cat([predict_para_real_all, predict_para], dim=0)
        np.save(os.path.join(output_directory, 'predict_para_real_all.npy'), predict_para_real_all.detach().cpu().numpy())
        

        if i == 0:
            original_para_real_all = batch_para
        else:
            original_para_real_all = torch.cat([original_para_real_all, batch_para], dim=0)
        np.save(os.path.join(output_directory, 'original_para_real_all.npy'), original_para_real_all.detach().cpu().numpy())

        predict_para = predict_para.detach().cpu().numpy()
        batch_para = batch_para.detach().cpu().numpy()



    #############################train sampling###########################################
    print('sampling from training trajectories.')
    for i, (batch_para, batch_traj) in enumerate(train_sp_loader):
        batch_para=batch_para.cuda()
        batch_traj=batch_traj.cuda()


        num_samples = batch_para.size(0)
        sample_length = batch_para.size(2)
        sample_channels = batch_para.size(1)
        predict_para = sampling(ema.ema_model, (num_samples, sample_channels, sample_length),
                                       diffusion_hyperparams,
                                       cond=batch_traj)
                                       

        if i == 0:
            predict_para_train_all = predict_para
        else:
            predict_para_train_all = torch.cat([predict_para_train_all, predict_para], dim=0)
        np.save(os.path.join(output_directory, 'predict_para_train_all.npy'), predict_para_train_all.detach().cpu().numpy())
        

        if i == 0:
            original_para_train_all = batch_para
        else:
            original_para_train_all = torch.cat([original_para_train_all, batch_para], dim=0)
        np.save(os.path.join(output_directory, 'original_para_train_all.npy'), original_para_train_all.detach().cpu().numpy())

        predict_para = predict_para.detach().cpu().numpy()
        batch_para = batch_para.detach().cpu().numpy()


    #############################test sampling###########################################
    print('sampling from testing trajectories.')
    for i, (batch_para, batch_traj) in enumerate(test_loader):
        batch_para=batch_para.cuda()
        batch_traj=batch_traj.cuda()

        num_samples = batch_para.size(0)
        sample_length = batch_para.size(2)
        sample_channels = batch_para.size(1)
        predict_para = sampling(ema.ema_model, (num_samples, sample_channels, sample_length),
                                       diffusion_hyperparams,
                                       cond=batch_traj)
                                       

        if i == 0:
            predict_para_test_all = predict_para
        else:
            predict_para_test_all = torch.cat([predict_para_test_all, predict_para], dim=0)
        np.save(os.path.join(output_directory, 'predict_para_test_all.npy'), predict_para_test_all.detach().cpu().numpy())
        

        if i == 0:
            original_para_test_all = batch_para
        else:
            original_para_test_all = torch.cat([original_para_test_all, batch_para], dim=0)
        np.save(os.path.join(output_directory, 'original_para_test_all.npy'), original_para_test_all.detach().cpu().numpy())

        predict_para = predict_para.detach().cpu().numpy()
        batch_para = batch_para.detach().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config_JCDI.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    train(**train_config)






