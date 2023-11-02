import os
import torch
import numpy as np
import argparse
import torch.fft as FFT
import glob
import scipy.io as scio
import tensorflow as tf
import logging
from PIL import Image
import cv2
def init_seeds(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed(seed)
    # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

def make_multidataset(dir,seqtypes_set):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dirs, fnames in sorted(os.walk(dir)):
        if not dirs:
            fnames.sort()
            datapoint = dict()
            for fname in fnames:
                seqtype = fname.split('_')[3]
                datapoint[seqtype] = os.path.join(root,fname)
                # images.append(path)
            assert set(datapoint.keys()) == seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
            images.append(datapoint)
    return images



  
def save_mat(save_dict, variable, file_name, index=0, normalize=True):
    # variable = variable.cpu().detach().numpy()
    if normalize:
        # variable_abs = np.absolute(variable)
        # max = np.max(variable_abs)
        # variable = variable / max
        variable = normalize_complex(variable)
    variable = variable.cpu().detach().numpy()
    file = os.path.join(save_dict, str(file_name) +
                        '_' + str(index + 1) + '.mat')
    datadict = {str(file_name): np.squeeze(variable)}
    scio.savemat(file, datadict)

def save_png(save_dict, variable, file_name, index=0, normalize=True):
    if normalize:
        variable = normalize(variable)
    variable = variable.cpu().detach().numpy()
    file = os.path.join(save_dict, str(file_name) +
                        '_' + str(index + 1) + '.tiff')
    datadict = {str(file_name): np.squeeze(variable)}
   
    img = (np.squeeze(variable)*255).astype(np.uint8)
    im = Image.fromarray(img)
    im.save(file)
    data1 = (np.squeeze(variable)*255).astype(np.uint8)
   


def get_all_files(folder, pattern='*'):
    files = [x for x in glob.iglob(os.path.join(folder, pattern))]
    return sorted(files)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def to_tensor(x):
    re = np.real(x)
    im = np.imag(x)
    x = np.concatenate([re, im], 1)
    del re, im
    return torch.from_numpy(x)


def crop(img, cropx, cropy):
    nb, c, y, x = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[:, :, starty:starty + cropy, startx: startx + cropx]


def normalize(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= torch.min(img)
    img /= (torch.max(img)-torch.min(img))
    return img

def z_Score_Normalization(data):
    data = data.reshape(-1)
    print('data',data.shape)
    data_z_np = (data - torch.mean(data, axis=0)) / torch.std(data, axis=0)
    return data_z_np

def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img




def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x



def r2c(x):
    re, im = torch.chunk(x, 2, 1)
    x = torch.complex(re, im)
    return x


def c2r(x):
    x = torch.cat([torch.real(x), torch.imag(x)], 1)
    return x


def sos(x):
    xr, xi = torch.chunk(x, 2, 1)
    x = torch.pow(torch.abs(xr), 2)+torch.pow(torch.abs(xi), 2)
    x = torch.sum(x, dim=1)
    x = torch.pow(x, 0.5)
    x = torch.unsqueeze(x, 1)
    return x


def Abs(x):
    x = r2c(x)
    return torch.abs(x)


def l2mean(x):
    result = torch.mean(torch.pow(torch.abs(x), 2))

    return result


def TV(x, norm='L1'):
    nb, nc, nx, ny = x.size()
    Dx = torch.cat([x[:, :, 1:nx, :], x[:, :, 0:1, :]], 2)
    Dy = torch.cat([x[:, :, :, 1:ny], x[:, :, :, 0:1]], 3)
    Dx = Dx - x
    Dy = Dy - x
    tv = 0
    if norm == 'L1':
        tv = torch.mean(torch.abs(Dx)) + torch.mean(torch.abs(Dy))
    elif norm == 'L2':
        Dx = Dx * Dx
        Dy = Dy * Dy
        tv = torch.mean(Dx) + torch.mean(Dy)
    return tv


def restore_checkpoint(ckpt_dir, state, device):
    # if not tf.io.gfile.exists(ckpt_dir):
    #     tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    #     logging.warning(f"No checkpoint found at {ckpt_dir}. "
    #                     f"Returned the same state as input")
    #     return state
    # else:

    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    
    return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)
