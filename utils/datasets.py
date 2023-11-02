import os
import sys
import torch
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.utils import *
from torchvision import transforms as T
from functools import partial
from pathlib import Path
from torch import nn, einsum
from PIL import Image
import nibabel as nib
from visdom import Visdom
viz = Visdom(port=2012)

def exists(x):
    return x is not None

class Tumor_Health_Dataset(Dataset):
    def __init__(
        self,
        config,
        folder,
        image_size,
        test_flag=False,
        exts=['jpg', 'jpeg', 'png', 'tiff'],
        convert_image_to = None
    ):
        super().__init__()
        self.config = config
        self.folder_tumor = os.path.join(folder, 'Tumor_png')
        self.folder_health = os.path.join(folder, 'Health_png')
        self.image_size = image_size
        self.test_flag = test_flag
        self.paths_tumor = sorted([p for ext in exts for p in Path(f'{self.folder_tumor}').glob(f'**/*.{ext}')])
        self.paths_health = sorted([q for ext in exts for q in Path(f'{self.folder_health}').glob(f'**/*.{ext}')])

        maybe_convert_fn = partial(convert_image_to, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.Grayscale(1),
            
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths_health)

    def __getitem__(self, index):
        path_health = self.paths_health[index]
        file_name_health = os.path.basename(path_health)
        img_health = Image.open(path_health) 
        img_health = self.transform(img_health) 

        path_tumor = self.paths_tumor[index]
        file_name_tumor = os.path.basename(path_tumor)
        
        img_tumor = Image.open(path_tumor) # the corresponding tumor
        img_tumor = self.transform(img_tumor)
        
        if self.config.data.normalize_type == 'minmax':
            img_health = normalize(img_health)
            img_tumor = normalize(img_tumor)
            # Visualization
            viz.images(img_health, opts=dict(caption=file_name_health))
            viz.images(img_tumor, opts=dict(caption=file_name_tumor))
            
        elif self.config.data.normalize_type == 'zscore':
            
            # elif self.config.data.normalize_type == 'std':
            #     minv = np.std(ksp_idx)
            #     ksp_idx = ksp_idx / (self.config.data.normalize_coeff * minv)
            img_health = z_Score_Normalization(img_health)
            
            img_tumor = z_Score_Normalization(img_tumor)
          
            T.Normalize(mean = [0.485,0.456,0,406],
                        std = [0.229,0.224,0.225])
            
        if self.test_flag:
            path2 = './dataset/Brats/test/label_png'
            file_corr_name = file_name_health.split('_tumor')[0]  
            first_name = file_name_health[0:21] 
            number = file_corr_name.split('_')[-1] 
            file_name_label = first_name + 'seg' +'_'+ number + '_label.png'
            path_label = os.path.join(path2,file_name_label)
            
            img_label = Image.open(path_label)
            img_label = self.transform(img_label)
            return img_health, img_tumor, img_label, file_corr_name
        
        return img_health, img_tumor


class Multi_Tumor_Health_Dataset(Dataset):
    def __init__(
        self,
        config,
        folder,
        image_size,
        test_flag=False,
        convert_image_to = None
    ):
        super().__init__()
        self.config = config
        self.folder_tumor = os.path.join(folder, 'Tumor_png')
        
        self.image_size = image_size
        self.test_flag = test_flag
        
        self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        self.seqtypes_set = set(self.seqtypes)
        self.paths_tumor = (make_multidataset(self.folder_tumor, self.seqtypes_set))   # load images from '/path/to/data/trainA'
       
        maybe_convert_fn = partial(convert_image_to, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.Grayscale(1),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths_tumor)

    def __getitem__(self, index):
        path_tumor = self.paths_tumor[index]
       
        B_out = []
        file_list = []
        for seqtype in self.seqtypes:
            file_name_tumor = os.path.basename(path_tumor[seqtype])
              
            img_tumor = Image.open(path_tumor[seqtype])
            img_tumor = self.transform(img_tumor)
            img_tumor = torch.squeeze(img_tumor)
            
            if self.config.data.normalize_type == 'minmax':
                img_tumor = normalize(img_tumor)
                viz.images(img_tumor, opts=dict(caption=file_name_tumor))
            elif self.config.data.normalize_type == 'zscore':
                img_tumor = z_Score_Normalization(img_tumor)
            
            B_out.append(img_tumor)
        B_out = torch.stack(B_out)
        print('B_out',B_out.shape)
        
        if self.test_flag:
            path2 = './dataset/Brats/test/label_png'
            file_corr_name = file_name_tumor.split('_tumor')[0]  
            first_name = file_name_tumor[0:21]
            number = file_corr_name.split('_')[-1]
            file_name_label = first_name + 'seg' +'_'+ number + '_label.png'
            path_label = os.path.join(path2,file_name_label)
            
            img_label = Image.open(path_label)
            img_label = self.transform(img_label)
            # print('img_label',img_label.shape)
            viz.images(img_label, opts=dict(caption=file_name_label))
            return  B_out, img_label, file_corr_name
        
        return  B_out
    

def get_dataset(config, mode):
    print("Dataset name:", config.data.dataset_name)
    if config.data.dataset_name == 'brats':
        if mode == 'training':
            dataset = Tumor_Health_Dataset(config, 
                        './dataset/Brats/', image_size=256, test_flag=False,
                        )
        else: 
            dataset = Multi_Tumor_Health_Dataset(config, 
                        './dataset/Brats/test/', image_size=256, test_flag=True,
                        )
   
    
    print('dataset:', config.data.dataset_name)

    if mode == 'training':
        data = DataLoader(
            dataset, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
    
    else:
        data = DataLoader(
            dataset, batch_size=config.sampling.batch_size, shuffle=False, pin_memory=True)

    print(mode, "data loaded")

    return data
