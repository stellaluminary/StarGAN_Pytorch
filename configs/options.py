import os
import os.path as osp
import logging
import yaml
import torch

def parser(parse_args):

    with open(parse_args.filename, 'r') as f:
        try:
            opt = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    if opt['Setting']['phase'] == 'train':
        opt['is_train'] = True
    else:
        opt['is_train'] = False

    # choose the device in torch version
    opt['device'] = torch.device('cuda' if opt['Setting']['gpu_ids'] else 'cpu')

    # show the gpu_list based on configuration
    gpu_list = ','.join(str(x) for x in opt['Setting']['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('Export CUDA_VISIBLE_DEVICES = ' + gpu_list)
    print('Running Device :', opt['device'])

    return opt