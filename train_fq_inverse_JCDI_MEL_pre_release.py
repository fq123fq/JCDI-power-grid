import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

from   torch.utils.data import TensorDataset, DataLoader, Subset

from utils.util_pre_release import find_max_epoch, print_size, training_loss, calc_diffusion_hyperparams


from powergrid_hybrid_eventmore_mNN_ind_tokenmore.powergrid_hybrid import PowerGridHybrid
from powergrid_hybrid_eventmore_mNN_ind_tokenmore.resnet_1d import ResNetEncoder

#import matplotlib.pyplot as plt

from utils.util_pre_release import sampling
from sklearn.metrics import mean_squared_error
from statistics import mean

import time
from ema_pytorch import EMA
import mat73

#run command:
#python train_fq_inverse_JCDI_MEL_pre_release.py -c config_inverse/config_inverse_JCDI_MEL_pre_release.json


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

def norm_traj(traj, norm = None):
    if norm is None:
        traj_min = torch.min(traj)
        traj_max = torch.max(traj)
        norm     = (traj_min, traj_max)
    else:
        (traj_min, traj_max) = norm

    result = (traj - traj_min) / (traj_max - traj_min)

    return (result, norm)

def norm_traj_bundle(traj_bundle, norm = None):
    # traj_bundle : (N, n_traj, L)
    n_traj = traj_bundle.shape[1]

    if norm is None:
        norm = [ None, ] * n_traj

    norm_list      = []
    norm_traj_list = []

    # norm_traj_list : List[ (N, L) ]
    for i in range(n_traj):
        traj_norm_i, norm_i = norm_traj(traj_bundle[:, i], norm[i])
        norm_list.append(norm_i)
        norm_traj_list.append(traj_norm_i)

    # result : (N, n_traj, L)
    result = torch.stack(norm_traj_list, dim = 1)

    return (result, norm_list)

def load_dataset(
    path,
    norm_p     = None,
    norm_q     = None,
    key_params = 'para_for_gene_eventmore_merge_trs_nom',
    key_traj_p = 'traj_p_gene_eventmore_merge_trs',
    key_traj_q = 'traj_q_gene_eventmore_merge_trs',
):
    # pylint: disable=too-many-arguments
    data = mat73.loadmat(path)

    # params : (N, n_params)
    params = data[key_params]

    # traj_p : (N, n_traj, L)
    # traj_q : (N, n_traj, L)
    traj_p = data[key_traj_p]
    traj_q = data[key_traj_q]

    params = torch.from_numpy(params).float()
    traj_p = torch.from_numpy(traj_p).float()
    traj_q = torch.from_numpy(traj_q).float()

    # traj_p_norm : (N, n_traj, L)
    # traj_q_norm : (N, n_traj, L)
    traj_p_norm, norm_p = norm_traj_bundle(traj_p, norm_p)
    traj_q_norm, norm_q = norm_traj_bundle(traj_q, norm_q)

    # traj_norm : List[ (N, 2, L) ]
    traj_norm = [
        torch.stack([ traj_p_norm[:, i], traj_q_norm[:, i] ], dim = 1)
            for i in range(traj_p_norm.shape[1])
    ]

    # traj_norm : List[ (N, n_traj, 2, L) ]
    traj_norm = torch.stack(traj_norm, dim = 1)

    # params : (N, 1, n_params)
    params = params.unsqueeze(1)

    return (traj_norm, params, norm_p, norm_q)

def construct_train_test_datasets(
    path,
    n_train     = 250000,
    n_test      = 50000,
    norm_p      = None,
    norm_q      = None,
    key_params  = 'para_for_gene_eventmore_merge_trs_nom',
    key_traj_p  = 'traj_p_gene_eventmore_merge_trs',
    key_traj_q  = 'traj_q_gene_eventmore_merge_trs',
):
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals

    (traj_norm_all, params_all, norm_p, norm_q) = load_dataset(
        path, norm_p, norm_q, key_params, key_traj_p, key_traj_q,
    )

    params_train = params_all[:n_train]
    params_test  = params_all[n_train:n_train+n_test]

    traj_norm_train = traj_norm_all[:n_train]
    traj_norm_test  = traj_norm_all[n_train:n_train+n_test]

    dset_train = TensorDataset(params_train, traj_norm_train)
    dset_test  = TensorDataset(params_test,  traj_norm_test)

    return (dset_train, dset_test, norm_p, norm_q)

def construct_train_test_dl(
    path,
    batch_size_train = 128,
    batch_size_test  = 128,
    n_train          = 250000,
    n_test           = 50000,
    norm_p           = None,
    norm_q           = None,
    key_params       = 'para_for_gene_eventmore_merge_trs_nom',
    key_traj_p       = 'traj_p_gene_eventmore_merge_trs',
    key_traj_q       = 'traj_q_gene_eventmore_merge_trs',
    stride_test      = 50,
    workers          = 1,
    use_strided_train_dataset = True,
):
    # pylint: disable=too-many-arguments

    (dset_train, dset_test, norm_p, norm_q) = construct_train_test_datasets(
        path, n_train, n_test, norm_p, norm_q, key_params,
        key_traj_p, key_traj_q
    )

    dl_train = DataLoader(
        dset_train,
        batch_size  = batch_size_train,
        shuffle     = True,
        pin_memory  = True,
        num_workers = workers
    )

    if stride_test != 1:
        strided_indices = range(0, len(dset_test), stride_test)
        dset_test       = Subset(dset_test, strided_indices)

    dl_test = DataLoader(
        dset_test,
        batch_size  = batch_size_test,
        shuffle     = False,
        pin_memory  = True,
        num_workers = workers
    )

    dl_train_stride = None

    if use_strided_train_dataset:
        strided_indices   = range(0, len(dset_train), stride_test)
        dset_train_stride = Subset(dset_train, strided_indices)

        dl_train_stride = DataLoader(
            dset_train_stride,
            batch_size  = batch_size_test,
            shuffle     = False,
            pin_memory  = True,
            num_workers = workers
        )

    return (dl_train, dl_test, dl_train_stride, norm_p, norm_q)

def construct_dataset(
    path,
    n_samples   = 50000,
    norm_p      = None,
    norm_q      = None,
    key_params  = 'para_for_gene_eventmore_merge_trs_nom',
    key_traj_p  = 'traj_p_gene_eventmore_merge_trs',
    key_traj_q  = 'traj_q_gene_eventmore_merge_trs',
):
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    (traj_norm, params, norm_p, norm_q) = load_dataset(
        path, norm_p, norm_q, key_params, key_traj_p, key_traj_q,
    )

    if n_samples is not None:
        traj_norm = traj_norm[:n_samples]
        params    = params[:n_samples]

    dataset = TensorDataset(params, traj_norm)

    return (dataset, norm_p, norm_q)

def construct_dl(
    path,
    batch_size = 128,
    n_samples  = 5000,
    norm_p     = None,
    norm_q     = None,
    key_params = 'para_for_gene_eventmore_merge_trs_nom',
    key_traj_p = 'traj_p_gene_eventmore_merge_trs',
    key_traj_q = 'traj_q_gene_eventmore_merge_trs',
    shuffle    = False,
    workers    = 1,
):
    # pylint: disable=too-many-arguments
    (dataset, norm_p, norm_q) = construct_dataset(
        path, n_samples, norm_p, norm_q, key_params, key_traj_p, key_traj_q
    )

    result = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        pin_memory  = True,
        num_workers = workers
    )

    return (result, norm_p, norm_q)

def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          use_model):
          
    
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint, 
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate

    use_model (int):                0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    only_generate_missing (int):    0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    masking(str):                   'mnr': missing not at random, 'bm': blackout missing, 'rm': random missing
    missing_k (int):                k missing time steps for each feature across the sample length.
    """

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
    para_length_fq = 30
    traj_length_fq = 512
    n_events       = 3

    print('Loading Train/Test dataset...')
    (dl_train, dl_test, dl_train_stride, norm_p, norm_q) = \
        construct_train_test_dl(
            path             = trainset_config['train_data_path'],
            batch_size_train = 128,
            batch_size_test  = 128,
            n_train          = 250000,
            n_test           = 50000,
            norm_p           = None,
            norm_q           = None,
            key_params       = 'para_for_gene_eventmore_merge_trs_nom',
            key_traj_p       = 'traj_p_gene_eventmore_merge_trs',
            key_traj_q       = 'traj_q_gene_eventmore_merge_trs',
            stride_test      = 50,
            workers          = 1,
            use_strided_train_dataset = True,
        )

    print('Loading Real data dataset...')

    (dl_real, _norm_p, _norm_q) = construct_dl(
        path       = trainset_config['real_sp_data_path'],
        batch_size = 128,
        n_samples  = 5000,
        norm_p     = norm_p,
        norm_q     = norm_q,
        key_params = 'para_real_eventmore_merge_trs_nom',
        key_traj_p = 'traj_p_real_eventmore_merge_trs',
        key_traj_q = 'traj_q_real_eventmore_merge_trs',
        shuffle    = False,
        workers    = 1,
    )

    # predefine model
    if use_model == 0:
    #elif use_model == 1:
        net = PowerGridHybrid(
          #traj_shape = (2, 192),
         traj_shape = (2, traj_length_fq),
         n_events   = n_events,
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

        for i, (batch_para, batch_traj) in enumerate(dl_train):
            batch_para=batch_para.cuda()
            batch_traj=batch_traj.cuda()
            #print("batch_para size:", batch_para.shape)
            #print("batch_traj size:", batch_traj.shape)

            #batch_para = batch_para.permute(0, 2, 1)

            #print("mask.shape",mask.shape)
            #print("mask[0]", mask[0])
            #aa=mask[0][0]
            #bb = mask[0][1]
            #cc = mask[0][2]
            #aa=aa.permute(1, 0)
            #import matplotlib.pyplot as plt
            #plt.figure()
            ##plt.plot(mask[0].permute(1, 0))
            #plt.plot(aa)
            #plt.plot(bb)
            #plt.plot(cc)
            #plt.show()


            # back-propagation
            optimizer.zero_grad()
            #            print("batch size:", batch.shape)
            #           print("mask size:", mask.shape)
            #           print("loss_mask size:", loss_mask.shape)
            X = batch_para, batch_traj

            #pred_fq=net(batch_para, batch_traj, torch.ones(32,1))
        
            #X = batch, None, mask, loss_mask
            loss = training_loss(net, nn.MSELoss(), X, diffusion_hyperparams)
            #loss = training_loss(net, nn.MSELoss(), batch_para, batch_traj, diffusion_hyperparams)

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

        epoch_train_loss=epoch_train_loss/ len(dl_train)
        epoch_train_loss_all.append([epoch_train_loss])
        #print("iteration: {} \tepoch_train_loss: {}".format(n_iter, epoch_train_loss))
        np.save(os.path.join(output_directory, 'epoch_train_loss_all.npy'), epoch_train_loss_all)

        # test
        if (n_epoch-1) % 50 == 0:
            print("calculate test loss")
            net.eval()
            ema.ema_model.eval()
            epoch_test_loss=0
            with torch.no_grad():
              for i, (batch_para, batch_traj) in enumerate(dl_test):
                  #calculate test loss
                  batch_para=batch_para.cuda()
                  batch_traj=batch_traj.cuda()
   
                  X = batch_para, batch_traj
                  #X = batch, None, mask, loss_mask
                  test_loss = training_loss(net, nn.MSELoss(), X, diffusion_hyperparams)

                  epoch_test_loss=epoch_test_loss+test_loss.item()

                  #sample test traj and calculate parameter error
                  num_samples = batch_para.size(0)
                  sample_length = batch_para.size(2)
                  sample_channels = batch_para.size(1)
                  #print("num_samples:", num_samples)
                  #print("sample_length:", sample_length)
                  #print("sample_channels:", sample_channels)
                  predict_para = sampling(ema.ema_model, (num_samples, sample_channels, sample_length),
                                                 diffusion_hyperparams,
                                                 cond=batch_traj)
                                       
                  if i == 0:
                      predict_para_test_all = predict_para
                  else:
                      predict_para_test_all = torch.cat([predict_para_test_all, predict_para], dim=0)

                  #np.save(os.path.join(output_directory, 'predict_para_test_all_epoch{}.npy'.format(n_epoch)), predict_para_test_all.detach().cpu().numpy())
        

                  if i == 0:
                      original_para_test_all = batch_para
                  else:
                      original_para_test_all = torch.cat([original_para_test_all, batch_para], dim=0)
                  #np.save(os.path.join(output_directory, 'original_para_test_all_epoch{}.npy'.format(n_epoch)), original_para_test_all.detach().cpu().numpy())

            epoch_test_loss=epoch_test_loss/ len(dl_test)
            epoch_test_loss_all.append([epoch_test_loss])
            print("iteration: {} \tepoch_train_loss: {}".format(n_iter, epoch_train_loss))
            print("iteration: {} \tepoch_test_loss: {}".format(n_iter, epoch_test_loss))
            n_iters_all.append([n_iter])
            n_epoches_all.append([n_epoch])
            np.savez(os.path.join(output_directory, 'epoch_train_test_losses_all'), epoch_train_loss_all=epoch_train_loss_all,epoch_test_loss_all=epoch_test_loss_all,n_iters_all=n_iters_all,n_epoches_all=n_epoches_all)

            np.save(os.path.join(output_directory, 'predict_para_test_all_epoch{}.npy'.format(n_epoch)), predict_para_test_all.detach().cpu().numpy())
            np.save(os.path.join(output_directory, 'original_para_test_all_epoch{}.npy'.format(n_epoch)), original_para_test_all.detach().cpu().numpy())


            with torch.no_grad():
                for i, (batch_para, batch_traj) in enumerate(dl_real):
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

                    #np.save(os.path.join(output_directory, 'predict_para_real_sp_all_epoch{}.npy'.format(n_epoch)), predict_para_real_sp_all.detach().cpu().numpy())   

                    if i == 0:
                        original_para_real_sp_all = batch_para
                    else:
                        original_para_real_sp_all = torch.cat([original_para_real_sp_all, batch_para], dim=0)

                    #np.save(os.path.join(output_directory, 'original_para_real_sp_all_epoch{}.npy'.format(n_epoch)), original_para_real_sp_all.detach().cpu().numpy())

            np.save(os.path.join(output_directory, 'predict_para_real_sp_all_epoch{}.npy'.format(n_epoch)), predict_para_real_sp_all.detach().cpu().numpy())  
            np.save(os.path.join(output_directory, 'original_para_real_sp_all_epoch{}.npy'.format(n_epoch)), original_para_real_sp_all.detach().cpu().numpy())

            with torch.no_grad():
                for i, (batch_para, batch_traj) in enumerate(dl_train_stride):
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

                    #np.save(os.path.join(output_directory, 'predict_para_real_sp_all_epoch{}.npy'.format(n_epoch)), predict_para_real_sp_all.detach().cpu().numpy())   

                    if i == 0:
                        original_para_train_sp_all = batch_para
                    else:
                        original_para_train_sp_all = torch.cat([original_para_train_sp_all, batch_para], dim=0)

                    #np.save(os.path.join(output_directory, 'original_para_real_sp_all_epoch{}.npy'.format(n_epoch)), original_para_real_sp_all.detach().cpu().numpy())

            np.save(os.path.join(output_directory, 'predict_para_train_sp_all_epoch{}.npy'.format(n_epoch)), predict_para_train_sp_all.detach().cpu().numpy())  
            np.save(os.path.join(output_directory, 'original_para_train_sp_all_epoch{}.npy'.format(n_epoch)), original_para_train_sp_all.detach().cpu().numpy())




    #############################real sampling###########################################
    ema.ema_model.eval()

    #for i, batch in enumerate(training_data):
    for i, (batch_para, batch_traj) in enumerate(dl_real):
        batch_para=batch_para.cuda()
        batch_traj=batch_traj.cuda()

        #batch_para = batch_para.permute(0, 2, 1)

        #print("mask.shape",mask.shape)
        #print("mask[0]", mask[0])
        #plt.figure()
        #plt.plot(mask[0].permute(1, 0))
        #plt.show()

        #       start = torch.Event(enable_timing=True)
        #       end = torch.Event(enable_timing=True)
        #       start.record()
        num_samples = batch_para.size(0)
        sample_length = batch_para.size(2)
        sample_channels = batch_para.size(1)
        print("num_samples:", num_samples)
        print("sample_length:", sample_length)
        print("sample_channels:", sample_channels)
        T3 = time.time()
        predict_para = sampling(ema.ema_model, (num_samples, sample_channels, sample_length),
                                       diffusion_hyperparams,
                                       cond=batch_traj)
                                       
        T4 = time.time()
        print('程序运行时间:%s毫秒' % ((T4 - T3) * 1000))
        print("okok1")

        if i == 0:
            predict_para_real_all = predict_para
        else:
            predict_para_real_all = torch.cat([predict_para_real_all, predict_para], dim=0)
        #np.save(
        #    "results/inverse_nomask/event1_12para_s_uniform/predict_para_train_all.npy", predict_para_train_all)
        np.save(os.path.join(output_directory, 'predict_para_real_all.npy'), predict_para_real_all.detach().cpu().numpy())
        

        if i == 0:
            original_para_real_all = batch_para
        else:
            original_para_real_all = torch.cat([original_para_real_all, batch_para], dim=0)
        #np.save(
        #    "results/inverse_nomask/event1_12para_s_uniform/original_para_train_all.npy", original_para_train_all)
        np.save(os.path.join(output_directory, 'original_para_real_all.npy'), original_para_real_all.detach().cpu().numpy())

        predict_para = predict_para.detach().cpu().numpy()
        batch_para = batch_para.detach().cpu().numpy()

        print("predict_para:",predict_para.shape)
        print("predict_para_real_all:", predict_para_real_all.shape)
        print("original_para_real_all:", original_para_real_all.shape)
        print("batch_para:", batch_para.shape)


    #############################train sampling###########################################
    #for i, batch in enumerate(training_data):
    for i, (batch_para, batch_traj) in enumerate(dl_train_stride):
        batch_para=batch_para.cuda()
        batch_traj=batch_traj.cuda()

        print("batch_para.size:", batch_para.size)

        #batch_para = batch_para.permute(0, 2, 1)

        #print("mask.shape",mask.shape)
        #print("mask[0]", mask[0])
        #plt.figure()
        #plt.plot(mask[0].permute(1, 0))
        #plt.show()

        #       start = torch.Event(enable_timing=True)
        #       end = torch.Event(enable_timing=True)
        #       start.record()
        num_samples = batch_para.size(0)
        sample_length = batch_para.size(2)
        sample_channels = batch_para.size(1)
        print("num_samples:", num_samples)
        print("sample_length:", sample_length)
        print("sample_channels:", sample_channels)
        T3 = time.time()
        predict_para = sampling(ema.ema_model, (num_samples, sample_channels, sample_length),
                                       diffusion_hyperparams,
                                       cond=batch_traj)
                                       
        T4 = time.time()
        print('程序运行时间:%s毫秒' % ((T4 - T3) * 1000))
        print("okok1")

        if i == 0:
            predict_para_train_all = predict_para
        else:
            predict_para_train_all = torch.cat([predict_para_train_all, predict_para], dim=0)
        #np.save(
        #    "results/inverse_nomask/event1_12para_s_uniform/predict_para_train_all.npy", predict_para_train_all)
        np.save(os.path.join(output_directory, 'predict_para_train_all.npy'), predict_para_train_all.detach().cpu().numpy())
        

        if i == 0:
            original_para_train_all = batch_para
        else:
            original_para_train_all = torch.cat([original_para_train_all, batch_para], dim=0)
        #np.save(
        #    "results/inverse_nomask/event1_12para_s_uniform/original_para_train_all.npy", original_para_train_all)
        np.save(os.path.join(output_directory, 'original_para_train_all.npy'), original_para_train_all.detach().cpu().numpy())

        predict_para = predict_para.detach().cpu().numpy()
        batch_para = batch_para.detach().cpu().numpy()

        print("predict_para:",predict_para.shape)
        print("predict_para_train_all:", predict_para_train_all.shape)
        print("original_para_train_all:", original_para_train_all.shape)
        print("batch_para:", batch_para.shape)


    #############################test sampling###########################################

    #for i, batch in enumerate(training_data):
    for i, (batch_para, batch_traj) in enumerate(dl_test):
        batch_para=batch_para.cuda()
        batch_traj=batch_traj.cuda()

        #batch_para = batch_para.permute(0, 2, 1)

        #print("mask.shape",mask.shape)
        #print("mask[0]", mask[0])
        #plt.figure()
        #plt.plot(mask[0].permute(1, 0))
        #plt.show()

        #       start = torch.Event(enable_timing=True)
        #       end = torch.Event(enable_timing=True)
        #       start.record()
        num_samples = batch_para.size(0)
        sample_length = batch_para.size(2)
        sample_channels = batch_para.size(1)
        print("num_samples:", num_samples)
        print("sample_length:", sample_length)
        print("sample_channels:", sample_channels)
        T3 = time.time()
        predict_para = sampling(ema.ema_model, (num_samples, sample_channels, sample_length),
                                       diffusion_hyperparams,
                                       cond=batch_traj)
                                       
        T4 = time.time()
        print('程序运行时间:%s毫秒' % ((T4 - T3) * 1000))
        print("okok1")

        if i == 0:
            predict_para_test_all = predict_para
        else:
            predict_para_test_all = torch.cat([predict_para_test_all, predict_para], dim=0)
        #np.save(
        #    "results/inverse_nomask/event1_12para_s_uniform/predict_para_train_all.npy", predict_para_train_all)
        np.save(os.path.join(output_directory, 'predict_para_test_all.npy'), predict_para_test_all.detach().cpu().numpy())
        

        if i == 0:
            original_para_test_all = batch_para
        else:
            original_para_test_all = torch.cat([original_para_test_all, batch_para], dim=0)
        #np.save(
        #    "results/inverse_nomask/event1_12para_s_uniform/original_para_train_all.npy", original_para_train_all)
        np.save(os.path.join(output_directory, 'original_para_test_all.npy'), original_para_test_all.detach().cpu().numpy())

        predict_para = predict_para.detach().cpu().numpy()
        batch_para = batch_para.detach().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/SSSDS4_fq.json',
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






