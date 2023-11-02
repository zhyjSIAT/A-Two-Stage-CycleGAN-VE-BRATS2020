"""Training and evaluation for score-based generative models. """

from dataclasses import dataclass
import gc
import io
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
from models import ncsnpp, ddpm,attention_block
import losses
import sampling
from datetime import datetime
import time
from models import model_utils as mutils
from models.ema import ExponentialMovingAverage
# import evaluation
import sde_lib
from absl import flags
import torch
from torch import nn
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils.utils import *
import utils.datasets as datasets
from metric import *

from skimage.filters import threshold_otsu
from visdom import Visdom
viz = Visdom(port=2012)

import seaborn as sns
import matplotlib.pyplot as plt
import csv
import warnings
from tqdm import tqdm

FLAGS = flags.FLAGS

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def create_mask(difftot):
    diff = np.array(difftot.cpu()) # dtype('float32')
    thresh = threshold_otsu(diff) # 
    mask = torch.where(torch.tensor(diff) > thresh, 1, 0) # torch.Size([1, 1, 256, 256])
    viz.image(visualize(mask), opts=dict(caption='mask')) 
    return mask


def Img1(src):
        src = (src * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(src, connectivity=8, ltype=None)
        img = np.zeros((src.shape[0], src.shape[1]), np.uint8)    
        for i in range(1, num_labels):
            mask = labels == i            
            if stats[i][4] > 400:         
                img[mask] = 255
                img[mask] = 255
                img[mask] = 255           
            else:
                img[mask] = 0
                img[mask] = 0
                img[mask] = 0
        return img
    
def save_heatmap(difftot,folder,filename):
    fig = plt.figure()
    diff_img = visualize(difftot).cpu()
    heatdiff = np.array(diff_img)
    print('heatdiff',heatdiff.shape) 
    
    ax  = fig.add_subplot(1,1,1)
    p1 = sns.heatmap(heatdiff,annot=False,ax=ax,square=True, cmap = 'viridis', xticklabels=False, yticklabels=False, cbar=False) # GnBu
    s1 = p1.get_figure()
    path = folder + '/' + str(filename) +'_HeatMap.tiff'
    print("save_path", path)
    s1.savefig(path, dpi=600, bbox_inches='tight',format='tiff')
    
    plt.close('all')


def add_weight(weight_list, tumor, recon):
    multidiff = 0
    for i in range(len(weight_list)):
        multidiff = multidiff + weight_list[i] * (abs(tumor[i]-recon[i]))
    viz.image(visualize(multidiff), opts=dict(caption='multidiff1')) 
    return multidiff

def cal_all(img,label):
    '''
    img,label: torch cpu
    shape:(256,256)
    '''
    DSC=dice_score(img, label)
    auprc = auc_roc(img, label)
    iou_score = iou(img, label)
    recall_score = recall(img, label)
    pre_score = precision(img, label)
    f11 = f1_score(img, label)
    hd = hausdorff_95(img, label)
    if DSC>1 or auprc>1 or iou_score>1 or recall_score>1 or pre_score>1 or f11>1:
        raise NotImplementedError(f"metric value should be between 0 and 1.")
    print('--metric--',DSC,auprc, iou_score, recall_score, pre_score, f11, hd)
    return DSC, auprc, iou_score, recall_score, pre_score, f11, hd

def multi_eval(weight_list, tumor, recon, label, filename):
    
    multidiff = add_weight(weight_list, tumor, recon) 
    save_heatmap(torch.squeeze(multidiff).cpu(), FLAGS.config.sampling.folder, filename[0])
    
    mask = torch.squeeze(create_mask(multidiff)) # otsu threshold method
    res_mask = Img1(mask.cpu().numpy().astype(np.float64)) # Remove interference from nearby pixels
    
    res_mask = res_mask.astype(np.int64) / 255
    viz.image(visualize(res_mask), opts=dict(caption='res_img2')) 

    DSC, auprc, iou_score, recall_score, pre_score, f11, hd = cal_all(torch.from_numpy(res_mask),label.cpu())
    mask_name = filename[0] + '_mask_'+ str(format(DSC.item(),'.4f'))
    save_heatmap(torch.from_numpy(res_mask), FLAGS.config.sampling.folder, mask_name)
    return DSC, auprc, iou_score, recall_score, pre_score, f11, hd  

 
def health_eval(weight_list, tumor, recon, label, filename):
    multidiff = add_weight(weight_list, tumor, recon) 
    save_heatmap(torch.squeeze(multidiff).cpu(), FLAGS.config.sampling.folder, filename[0])
    mask = torch.squeeze(create_mask(multidiff)) 
    res_mask = Img1(mask.cpu().numpy().astype(np.float64)) 
    res_mask = res_mask.astype(np.int64) / 255
    viz.image(visualize(res_mask), opts=dict(caption='res_img2')) 
    mask_name = filename[0] + '_mask_'
    save_heatmap(torch.from_numpy(res_mask), FLAGS.config.sampling.folder, mask_name)
  
    
def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    # The directory for saving test results during training
    sample_dir = os.path.join(workdir, "samples_in_train")
    tf.io.gfile.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)

    if config.training.continue_train:
        score_model = mutils.create_model(config)
        optimizer = losses.get_optimizer(config, score_model.parameters())
        ema = ExponentialMovingAverage(
                score_model.parameters(), decay=config.model.ema_rate)
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
        checkpoint_dir = os.path.join("results", FLAGS.config.training.continue_folder, "checkpoints")
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{config.training.continue_ckpt}.pth')
        state = restore_checkpoint(ckpt_path, state, device=config.device)
        print("load continue_train weights:", ckpt_path)
    # Initialize model.
    else:
        score_model = mutils.create_model(config) #NCSNpp
        ema = ExponentialMovingAverage(
            score_model.parameters(), decay=config.model.ema_rate)
        optimizer = losses.get_optimizer(config, score_model.parameters())
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
        config.training.continue_ckpt = 0
        # Create checkpoints directory
        checkpoint_dir = os.path.join(workdir, "checkpoints")
        tf.io.gfile.makedirs(checkpoint_dir)
    # Resume training when intermediate checkpoints are detected
    # state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Build pytorch dataloader for training
    train_dl = datasets.get_dataset(config, 'training')
    num_data = len(train_dl.dataset)

    # Create data scaler and its inverse
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(config)
        sampling_eps = 1e-5
    elif config.training.sde.lower() == 'bratsde':
        sde = sde_lib.BRATSDE(config)
        sampling_eps = 1e-5  
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    
    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(config, sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)
    # eval_step_fn = losses.get_step_fn(config, sde, train=False, optimize_fn=optimize_fn,
    #                                   reduce_mean=reduce_mean, continuous=continuous,
    #                                   likelihood_weighting=likelihood_weighting)

    # Building sampling functions
    if config.training.snapshot_sampling:
        # sampling_shape = (config.training.batch_size, config.data.num_channels,
        #                   config.data.image_size, config.data.image_size)
        # sampling_fn = sampling.get_sampling_fn(
        #     config, sde, sampling_shape, inverse_scaler, sampling_eps)
        pass

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))

    for epoch in range(config.training.continue_ckpt,config.training.epochs):
        loss_sum = 0
        for step, batch in enumerate(train_dl):
            t0 = time.time()
            ###########################################
            health,tumor = batch
            
            health = scaler(health).to(config.device)
            tumor = scaler(tumor).to(config.device)
        
            loss = train_step_fn(state, health, tumor)
            loss_sum += loss

            param_num = sum(param.numel()
                            for param in state["model"].parameters())
            if step % 10 == 0:
                print('Epoch', epoch + 1, '/', config.training.epochs, 'Step', step,
                        'loss = ', loss.cpu().data.numpy(),
                        'loss mean =', loss_sum.cpu().data.numpy() / (step + 1),
                        'time', time.time() - t0, 'param_num', param_num)

            if step % config.training.log_freq == 0:
                logging.info("step: %d, training_loss: %.5e" %
                             (step, loss.item()))
                global_step = num_data * epoch + step
                # writer.add_scalar(
                #     "training_loss", scalar_value=loss, global_step=global_step)
                pass

            # Report the loss on an evaluation dataset periodically
            if step % config.training.eval_freq == 0:
                pass
            

        # Save a checkpoint for every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch + 1 == 1:
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{epoch + 1}.pth'), state)

        # Generate and save samples for every epoch
        if config.training.snapshot_sampling and (epoch + 1) % config.training.snapshot_freq == 0:
            # config.sampling.ckpt = epoch + 1
            # sample_dir = ""
            pass


def sample(config, workdir):
    """Generate samples.

    Args:
      config: Configuration to use.
      workdir: Working directory.
    """
   
    health_score_model = mutils.create_model(config)
    health_optimizer = losses.get_optimizer(config, health_score_model.parameters())
    health_ema = ExponentialMovingAverage(health_score_model.parameters(), decay=config.model.ema_rate)
    health_state = dict(optimizer=health_optimizer, model=health_score_model, ema=health_ema, step=0)
    
    
    health_checkpoint_dir = os.path.join(workdir, FLAGS.config.sampling.folder, "checkpoints")
    health_ckpt_path = os.path.join(health_checkpoint_dir, f'checkpoint_{config.sampling.ckpt}.pth')
    health_state = restore_checkpoint(health_ckpt_path, health_state, device=config.device)
    print("load brat_health weights:", health_ckpt_path)
    
    if not config.training.joint:
        
        tumor_score_model = mutils.create_model(config)
        tumor_optimizer = losses.get_optimizer(config, health_score_model.parameters())
        tumor_ema = ExponentialMovingAverage(tumor_score_model.parameters(), decay=config.model.ema_rate)
        tumor_state = dict(optimizer=tumor_optimizer, model=tumor_score_model, ema=tumor_ema, step=0)
        tumor_checkpoint_dir = os.path.join(workdir, FLAGS.config.sampling.mri_folder, "checkpoints")
        tumor_ckpt_path = os.path.join(tumor_checkpoint_dir, f'checkpoint_{config.sampling.ckpt}.pth')
        tumor_state = restore_checkpoint(tumor_ckpt_path, tumor_state, device=config.device)
      
        
    else:
        tumor_score_model = None
        

    SAMPLING_FOLDER_ID = '_'.join(['ckpt', str(config.sampling.ckpt),
                        FLAGS.config.sampling.predictor, 
                        FLAGS.config.sampling.corrector,
                        str(config.sampling.snr),
                        'predictor_mse', str(FLAGS.config.sampling.mse),
                        'corrector_mse', str(FLAGS.config.sampling.corrector_mse),
                        str(FLAGS.config.model.beta_max)])
    # Build data pipeline
    test_dl = datasets.get_dataset(config, 'test') 

    FLAGS.config.sampling.folder = os.path.join(workdir, FLAGS.config.sampling.folder, SAMPLING_FOLDER_ID)
    tf.io.gfile.makedirs(FLAGS.config.sampling.folder)   

    # Create data scaler and its inverse
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(config)
        sampling_eps = 1e-5
    elif config.training.sde.lower() == 'bratsde':
        sde = sde_lib.BRATSDE(config)
        sampling_eps = 1e-5 
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    self_attention_block = attention_block.SelfAttentionBlock(in_channels=1, out_channels=64)
    # Build the sampling function when sampling is enabled
    sampling_shape = (config.sampling.batch_size, 1,
                                config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, 
                                inverse_scaler, sampling_eps)

    dice = []
    AUPRC_list = []
    IOU_list = []
    Recall_list = []
    Pre_list = []
    f1_list = []
    hd_list = []
    time_list = []
    filename_list = []
    best_weight_list = []
    for index, point in enumerate(test_dl):
        if index >= config.sampling.num_test:  # only apply our model to opt.num_test images.
            break
       
        tumor,label,filename = point
       
        ###########################################
        label = scaler(label).to(config.device)
        label = torch.where(label > 0, 1, 0)
        label = torch.squeeze(label) 
        
        tumor_list = []
        recon_list = []
        
        seqtypes = ['t1', 't1ce', 't2', 'flair']
        for i in range(4):
           
            tumor_input = torch.unsqueeze(tumor[:,i,...],0)
            tumor_input = scaler(tumor_input).to(config.device)
            start = time.time()
            recon = sampling_fn(health_score_model, tumor_score_model, tumor_input) 
            recon = torch.unsqueeze(recon[0][0,...],0)
            recon = visualize(recon)
            end = time.time()
            print('time consume',end-start, 'min', (end-start)/60)
            con_t = (end-start)
            time_list.append(con_t)
            cap_name = filename[0] + seqtypes[i]
            viz.images(visualize(recon), opts=dict(caption=cap_name))
            save_png(FLAGS.config.sampling.folder, recon, cap_name, index, normalize=False)
            tumor_list.append(tumor_input.detach().cpu()) 
            recon_list.append(recon.detach().cpu()) # [4,1,1,256,256]
            
        
        # best_weight = search_best_weight(tumor_list, recon_list, label,config.device)
        best_weight = [0.07,0.19,0.3,0.44]
        DSC, auprc, iou_score, recall_score, pre_score, f11, hd = multi_eval(best_weight, tumor_list, recon_list, label, filename)
       
        # health_eval(weight_list, tumor_list, recon_list, label, filename) # test healthy
        filename_list.append(filename[0])
        dice.append(DSC)
        AUPRC_list.append(auprc)
        IOU_list.append(iou_score)
        Recall_list.append(recall_score)
        Pre_list.append(pre_score)
        f1_list.append(f11)
        hd_list.append(hd)
        best_weight_list.append(best_weight)
    
       
    data_len = len(dice)
    assert len(dice)==len(AUPRC_list)==len(IOU_list)==len(Recall_list)==len(Pre_list)==len(f1_list)==len(hd_list)
     # Calculate average indicator
    avg_dice = sum(dice)/data_len
    avg_AUPRC = sum(AUPRC_list)/data_len
    avg_IOU = sum(IOU_list)/ data_len
    avg_recall = sum(Recall_list)/ data_len
    avg_pre = sum(Pre_list)/ data_len
    avg_f1 = sum(f1_list)/ data_len
    avg_hd = sum(hd_list)/ data_len
    
    print("total dice", avg_dice,'len',len(dice), '±',np.std(dice,ddof=1))
    print('AUPRC averange', avg_AUPRC,'',np.std(AUPRC_list,ddof=1))
    print('IOU averange',avg_IOU,'',np.std(IOU_list,ddof=1))
    print('Recall averange',  avg_recall,'',np.std(Recall_list,ddof=1))
    print('Pre averange',  avg_pre,'',np.std(Pre_list,ddof=1))
    print('f1 averange', avg_f1,'',np.std(f1_list,ddof=1))
    print('hd averange', avg_hd,'',np.std(hd_list,ddof=1))
    print('time len',len(time_list))
    print('time cost averange',sum(time_list)/ len(time_list))
    
    
    filename_list.append('average')
    dice.append(avg_dice)
    AUPRC_list.append(avg_AUPRC)
    IOU_list.append(avg_IOU)
    Recall_list.append(avg_recall)
    Pre_list.append(avg_pre)
    f1_list.append(avg_f1)
    hd_list.append(avg_hd)
    
    
    filename_list.append('std')
    dice.append(np.std(dice,ddof=1))
    AUPRC_list.append(np.std(AUPRC_list,ddof=1))
    IOU_list.append(np.std(IOU_list,ddof=1))
    Recall_list.append(np.std(Recall_list,ddof=1))
    Pre_list.append(np.std(Pre_list,ddof=1))
    f1_list.append(np.std(f1_list,ddof=1))
    hd_list.append(np.std(hd_list,ddof=1))
    
    
    csv_name = FLAGS.config.sampling.folder + 'sampling.csv'
    csvfile = open(csv_name,'wt',encoding="UTF8")
    writer = csv.writer(csvfile,delimiter=",")
    header = ['label','dice','auprc', 'iou_score', 'recall_score', 'pre_score', 'f11', 'hd','weight']
    writer.writerow(header)
    
    writer.writerows(zip(filename_list, dice, AUPRC_list, IOU_list, Recall_list, Pre_list, f1_list, hd_list,best_weight_list))   
    csvfile.close()    

def eval_all(weight_list, tumor, recon, label,device):
    
    multidiff = add_weight(weight_list, tumor, recon) # multidiff torch.Size([1, 1, 256, 256]) 
    
    mask = torch.squeeze(create_mask(multidiff)) 
    res_mask = Img1(mask.cpu().numpy().astype(np.float64)) 
    res_mask = res_mask.astype(np.int64) / 255
   
    DSC=dice_score(torch.from_numpy(res_mask),label.cpu())
    
    return DSC

def search_best_weight(tumor_list, recon_list, label,device):
    best_DSC = 0
    best_weight = []
    b_1 = 0
    b_2 = 0
    b_3 = 0
    b_4 = 0
    for w1 in np.arange(0, 1, 0.1):
        for w2 in np.arange(0, 1-w1, 0.1):
            for w3 in np.arange(0, 1-w1-w2, 0.1):
                    weight_list = []
                    w1_1 = round(w1, 2)
                    w2_2 = round(w2, 2)
                    w3_3 = round(w3, 2)
                    w4 = 1 - w1 - w2 - w3
                    w4_4 = round(w4, 2)
                    weight_list.append(w1_1)
                    weight_list.append(w2_2)
                    weight_list.append(w3_3)
                    weight_list.append(w4_4)
                    
                    DSC = eval_all(weight_list, tumor_list, recon_list, label,device)
                    print("w",w1_1,w2_2,w3_3,w4_4,DSC)
                    # print("dice",DSC)
                    
                    if DSC>best_DSC:
                        best_DSC = DSC
                        b_1 = w1_1
                        b_2 = w2_2
                        b_3 = w3_3
                        b_4 = w4_4
    best_weight.append(b_1) 
    best_weight.append(b_2) 
    best_weight.append(b_3) 
    best_weight.append(b_4)
    print("best_weight",best_weight[0],best_weight[1],best_weight[2],best_weight[3],best_DSC)
    return best_weight
    
