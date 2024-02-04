"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images,save_multimages
from util import html
from metric import *
import csv
import torch
import numpy as np
import cv2
from skimage.filters import threshold_otsu
import SimpleITK as sitk
import seaborn as sns
import matplotlib.pyplot as plt
import math
import time
from visdom import Visdom
viz = Visdom(port=8911)
# dice_window = viz.line( Y=torch.zeros((1)).cpu(), X=torch.zeros((1)).cpu(), opts=dict(xlabel='size', ylabel='dice', title='dice score'))

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img
 
def cal_vote(flair_seg, t1_seg, t1ce_seg, t2_seg):
    weights = np.array([0.8,0.05,0.05,0.2])
    stacked = np.stack((flair_seg, t1_seg, t1ce_seg, t2_seg), axis=0)
    weighted_sum = np.sum(stacked * weights[:, None, None, None], axis=0)
    # voted_seg = np.argmax(weighted_sum, axis=0)
    return (weighted_sum).astype(np.uint8)

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

def add_weight(weight_list, real, fake):
    multidiff = 0
    for i in range(4):
        multidiff = multidiff + weight_list[i] * (abs(real[i]-fake[i]))
    # viz.image(visualize(multidiff), opts=dict(caption='multidiff1')) 
    return multidiff

def create_mask(difftot):
    diff = np.array(difftot.cpu()) # dtype('float32')
    thresh = threshold_otsu(diff) # 
    mask = torch.where(torch.tensor(diff) > thresh, 1, 0) # torch.Size([1, 1, 256, 256])
    viz.image(visualize(mask), opts=dict(caption='mask')) 
    return mask

def Img1(src):
    """
    connected domain detection
    src: a binary segmentation mask
    """
    src = (src * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(src, connectivity=8, ltype=None)
    img = np.zeros((src.shape[0], src.shape[1]), np.uint8)    #创建个全0的黑背景
    for i in range(1, num_labels):
        mask = labels == i             #这一步是通过labels确定区域位置，让labels信息赋给mask数组，再用mask数组做img数组的索引
        if stats[i][4] > 300:         #300是面积 可以随便调
            img[mask] = 255
            img[mask] = 255
            img[mask] = 255           #面积大于300的区域涂白留下，小于300的涂0抹去
        else:
            img[mask] = 0
            img[mask] = 0
            img[mask] = 0
    return img

def cal_all(img,label):
    '''
    img,label: torch cpu
    shape:(256,256)
    '''
    DSC=dice_score(img, label)
    print("Dice Score3 res ", DSC)
    auprc = auc_roc(img, label)
    iou_score = iou(img, label)
    recall_score = recall(img, label)
    pre_score = precision(img, label)
    f11 = f1_score(img, label)
    hd = hausdorff_95(img, label)
    if DSC>1 or auprc>1 or iou_score>1 or recall_score>1 or pre_score>1 or f11>1:
        raise NotImplementedError(f"metric value should be between 0 and 1.")
    print('--metric--',auprc, iou_score, recall_score, pre_score, f11, hd)
    return DSC, auprc, iou_score, recall_score, pre_score, f11, hd

def eval_all(weight_list, real, fake, label):
    # 给予权重 调用 add_weight
    multidiff = add_weight(weight_list, real, fake) # multidiff torch.Size([1, 1, 256, 256]) 
    # 根据otsu阈值法 创建mask
    image_dir = webpage.get_image_dir()
    label_name = os.path.basename(label_path[0]).split('.')[0]
    save_heatmap(torch.squeeze(multidiff), image_dir, label_name)
    mask = torch.squeeze(create_mask(multidiff)) # mask ([1, 1, 256, 256]) -> squeeze torch.Size([256, 256])
    # 去除旁边像素的干扰
    res_mask = Img1(mask.cpu().numpy().astype(np.float64)) # res_mask (256,256) np.ndarray 值的范围 (0,256)
    # 将res_mask 范围转换为(0,1) dtype转换为 np.int64
    res_mask = res_mask.astype(np.int64) / 255
    viz.image(visualize(res_mask), opts=dict(caption='res_img2')) 
    # 存储mask图
    
    
    # 计算指标
    DSC, auprc, iou_score, recall_score, pre_score, f11, hd = cal_all(torch.from_numpy(res_mask),label.cpu())
    mask_name = label_name + '_mask_'+ str(format(DSC.item(),'.4f'))
    save_heatmap(torch.from_numpy(res_mask).to(device), image_dir, mask_name)
    return DSC, auprc, iou_score, recall_score, pre_score, f11, hd

def dice_best(weight_list, real, fake, label):
    # 给予权重 调用 add_weight
    multidiff = add_weight(weight_list, real, fake) # multidiff torch.Size([1, 1, 256, 256]) 
    # 根据otsu阈值法 创建mask
    image_dir = webpage.get_image_dir()
    label_name = os.path.basename(label_path[0]).split('.')[0]
    # save_heatmap(torch.squeeze(multidiff), image_dir, label_name)
    mask = torch.squeeze(create_mask(multidiff)) # mask ([1, 1, 256, 256]) -> squeeze torch.Size([256, 256])
    # 去除旁边像素的干扰
    res_mask = Img1(mask.cpu().numpy().astype(np.float64)) # res_mask (256,256) np.ndarray 值的范围 (0,256)
    # 将res_mask 范围转换为(0,1) dtype转换为 np.int64
    res_mask = res_mask.astype(np.int64) / 255
   
    # 计算指标
    # DSC, auprc, iou_score, recall_score, pre_score, f11, hd = cal_all(torch.from_numpy(res_mask),label.cpu())
    DSC = dice_score(torch.from_numpy(res_mask),label.cpu())
   
    return DSC

def search_best_weight(real, fake, label):
    best_DSC = 0
    best_weight = []
    ci = 0
    X = []
    X_labels = []
    Y1 = []
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
                    # print("w",w1_1,w2_2,w3_3,w4_4)
                    DSC = dice_best(weight_list, real, fake, label)
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
    # print("best_weight_len",len(best_weight))
    print("best_weight",best_weight[0],best_weight[1],best_weight[2],best_weight[3],best_DSC)
    # print("best_record",b_1,b_2,b_3,b_4)
    return best_weight    

def multi_eval(real,fake,label):
       
    # 给予权重 flair t2 提供主要信息 t1 t1ce补充
    
    # weight_list = [0.05, 0.05, 0.1, 0.8]
    best_weight = search_best_weight(real, fake, label)
    
    DSC, auprc, iou_score, recall_score, pre_score, f11, hd = eval_all(best_weight, real, fake, label)
    
    return DSC, auprc, iou_score, recall_score, pre_score, f11, hd
 
    # ensemble
    
    
    
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
        
    # calculate the metrics
    filename_list = []
    dice = []
    AUPRC_list = []
    IOU_list = []
    Recall_list = []
    Pre_list = []
    f1_list = []
    hd_list = []
    time_list = []
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        
        device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        label = data['label'].to(device)
        label = torch.where(label > 0, 1, 0) # torch.Size([1, 1, 256, 256])
        label = torch.squeeze(label) # torch.Size([256, 256]) cuda
        seqtypes = ['t1', 't1ce', 't2', 'flair']
        # seqtypes = ['T2_Flair','T1_Flair', 'T2','T1_Flair_CE']
        real_list = []
        fake_list = []
        
        # combine the four model results
        start = time.time()
        for j in range(4):
            model.set_singleinput(data,j,seqtypes[j])  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results real:torch.Size([1, 1, 256, 256]) fake:torch.Size([1, 1, 256, 256])
            img_path,label_path = model.get_image_paths()     # get image paths
            print('processing (%04d)-th image... %s' % (i, img_path))
            
            # Calculate difference image
            fake = visuals['fake'] # torch.Size([1, 1, 256, 256])
            real = visuals['real'] # torch.Size([1, 1, 256, 256])
            # viz.image(visualize(fake), opts=dict(caption=img_path))
            capname = img_path + ['real']
            # viz.image(visualize(real), opts=dict(caption=capname))
            real_list.append(real)
            fake_list.append(fake)
            # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
        
        end = time.time()
        
        time_list.append(end-start)
    #     DSC, auprc, iou_score, recall_score, pre_score, f11, hd = multi_eval(real_list,fake_list,label)
        
    #     filename_list.append(os.path.basename(label_path[0]).split('.')[0])
    #     dice.append(DSC)
    #     AUPRC_list.append(auprc)
    #     IOU_list.append(iou_score)
    #     Recall_list.append(recall_score)
    #     Pre_list.append(pre_score)
    #     f1_list.append(f11)
    #     if math.isnan(hd) :
    #         print('hd is nan')
    #         hd_list.append(0) 
    #     else:    
    #         hd_list.append(hd)
        
    
    # webpage.save()  # save the HTML
    # data_len = len(dice)
    # assert len(time_list)==len(dice)==len(AUPRC_list)==len(IOU_list)==len(Recall_list)==len(Pre_list)==len(f1_list)==len(hd_list)

    # data_len = len(dice)
    # assert len(dice)==len(AUPRC_list)==len(IOU_list)==len(Recall_list)==len(Pre_list)==len(f1_list)==len(hd_list)
    
    # # 计算平均指标
    # avg_dice = sum(dice)/data_len
    # avg_AUPRC = sum(AUPRC_list)/data_len
    # avg_IOU = sum(IOU_list)/ data_len
    # avg_recall = sum(Recall_list)/ data_len
    # avg_pre = sum(Pre_list)/ data_len
    # avg_f1 = sum(f1_list)/ data_len
    # avg_hd = sum(hd_list)/ data_len
    
    # print("total dice", avg_dice,'len',len(dice), '±',np.std(dice,ddof=1))
    # print('AUPRC averange', avg_AUPRC,'',np.std(AUPRC_list,ddof=1))
    # print('IOU averange',avg_IOU,'',np.std(IOU_list,ddof=1))
    # print('Recall averange',  avg_recall,'',np.std(Recall_list,ddof=1))
    # print('Pre averange',  avg_pre,'',np.std(Pre_list,ddof=1))
    # print('f1 averange', avg_f1,'',np.std(f1_list,ddof=1))
    # print('hd averange', avg_hd,'',np.std(hd_list,ddof=1))
    print('time len',len(time_list))
    print('time cost averange',sum(time_list)/ len(time_list))
    
    # 写入csv
    filename_list.append('average')
    dice.append(avg_dice)
    AUPRC_list.append(avg_AUPRC)
    IOU_list.append(avg_IOU)
    Recall_list.append(avg_recall)
    Pre_list.append(avg_pre)
    f1_list.append(avg_f1)
    hd_list.append(avg_hd)
    
    # 增加csv
    dir = os.path.join(opt.results_dir, opt.name)
    print('result_dir',dir) 
    csv_name = dir + 'inhouse.csv'
    csvfile = open(csv_name,'wt',encoding="UTF8")
    writer = csv.writer(csvfile,delimiter=",")
    header = ['label','dice','auprc', 'iou_score', 'recall_score', 'pre_score', 'f11', 'hd']
    writer.writerow(header)
    
    writer.writerows(zip(filename_list, dice, AUPRC_list, IOU_list, Recall_list, Pre_list, f1_list, hd_list))   
    csvfile.close()    
    
