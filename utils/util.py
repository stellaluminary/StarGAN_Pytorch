import os
import math
from datetime import datetime
import numpy as np
import cv2
from torchvision.utils import make_grid
import random
import torch
import logging
import time

logger = logging.getLogger('base')

####################
# make dir & print loss
####################

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)

def print_current_losses(save_txt_path, epochs, iters, total_iters, losses, add_val):
    """
    Parameters:
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs,
                                get from the model.get_current_losses()
    """
    message = get_timestamp()+'(epoch: %d/%d, iters: %d/%d, total iters: %d) ' \
              % (epochs[0], epochs[1], iters[0], iters[1], total_iters)
    txt_log_file = get_timestamp()+', %d, %d, %d' % (epochs[0], epochs[1], total_iters)
    for k, v in losses.items():
        message += 'loss %s: %.3f ' % (k, v)
        txt_log_file += ', %.3f' % (v)

    for k, v in add_val.items():
        if k == 'lr':
            message += '%s: %.8f ' % (k, v)
            txt_log_file += ', %.8f' % (v)
        else:
            message += '%s: %.3f ' % (k, v)
            txt_log_file += ', %.3f' % (v)

    print(message)  # print the message

    with open(save_txt_path, "a") as log_file:
        log_file.write('%s\n' % txt_log_file)  # save the message

def init_log_file(path, loss_name, add_val_name):
    if os.path.exists(path):
        new_name = path.split('.')[0] + '_' + get_timestamp() + '.txt'
        print('Txt File already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)

    log_column = 'epoch, epoch_iter, total_iters'
    for i in loss_name:
        log_column += ', %s' % i

    for i in add_val_name:
        log_column += ', %s' % i

    with open(path, "w") as log_file:
        log_file.write('%s\n' % log_column)  # save the message

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))

    os.makedirs(root, exist_ok=True)

    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)



####################
# image convert and save img
####################

def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)

def save_current_imgs(images, save_dirs, phase, id, epoch=0, min_max=(-1, 1)):
    """
    Parameters:
        images (OrderedDict) -- training images stored in the format of (name, 3D tensor) pairs,
                                get from the model.get_current_visuals()
    """
    for i,(name, img_tensor) in enumerate(images.items()):
        img = tensor2img(img_tensor, min_max=min_max)
        if phase == 'train':
            save_img_path = os.path.join(save_dirs[i], f'{name}_epoch_{epoch}_iter_{id}.png')
        else:
            save_img_path = os.path.join(save_dirs[i], f'{name}_{id}.png')
        save_img(img, save_img_path)

####################
# time
####################

def print_time(epoch, start_epoch_time):
    t = time.time() - start_epoch_time
    hour = t // 3600
    min = t % 3600 // 60
    sec = t % 60
    logger.info('--------%d Epoch takes %d hours %d minutes %.3f seconds--------' % (epoch, hour, min, sec))

####################
# time
####################

def is_pth_file(filename):
    return any(filename.endswith(extension) for extension in ['pth','pt'])

def extract_epoch(dir):
    epochs = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_pth_file(fname):
                epochs.append(fname.split('_')[0])
    return sorted(epochs)[-1]

