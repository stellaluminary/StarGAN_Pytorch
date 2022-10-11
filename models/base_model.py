import os
import torch
from collections import OrderedDict
from . import networks
import torch.nn as nn
from utils import util
import logging

logger = logging.getLogger('base')

class BaseModel():

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt['Setting']['gpu_ids']
        self.is_train = opt['Setting']['phase']
        self.device = torch.device('cuda' if self.gpu_ids else 'cpu')
        self.save_dir = opt['Path']['checkpoint_dir']
        self.resume_dir = opt['Path']['resume_model_dir']
        self.pretrain_dir = opt['Path']['pretrain_model_dir']

        self.loss_names = []
        self.model_names = []
        self.visual_names = []

        self.add_value_names=[]

        self.image_paths = []

        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def forward(self):
        pass

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        pass

    def load_pretrained_nets(self):
        epoch = util.extract_epoch(self.pretrain_dir)

        for name in self.model_names:
            if isinstance(name, str):

                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.pretrain_dir, load_filename)

                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path)
                net.load_state_dict(state_dict)

    def get_image_paths(self):
        return self.image_paths

    def update_learning_rate(self):
        for e, scheduler in enumerate(self.schedulers):
            old_lr = self.optimizers[e].param_groups[0]['lr']
            scheduler.step()
            lr = self.optimizers[e].param_groups[0]['lr']
            logger.info('[%d] learning rate %.7f -> %.7f' % (e, old_lr, lr))

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name).detach()[0].float().cpu()
        return visual_ret

    def make_visual_dir(self, path):
        vis_path = []
        for name in self.visual_names:
            new_path = os.path.join(path, name)
            util.mkdir(new_path)
            vis_path.append(new_path)
        return vis_path

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def get_current_add_values(self):
        ordic = OrderedDict()
        for name in self.add_value_names:
            if isinstance(name, str):
                ordic[name] = float(getattr(self, name))
        return ordic

    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if isinstance(net, nn.DataParallel):
                    net = net.module
                state_dict = net.state_dict()
                for key, param in state_dict.items():
                    state_dict[key] = param.cpu()
                torch.save(state_dict, save_path)

    def save_training_state(self, epoch):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}_epoch.state'.format(epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(state, save_path)

    def resume_networks(self, epoch):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                if self.is_train == 'train':
                    load_path = os.path.join(self.resume_dir, load_filename)
                else:
                    load_path = os.path.join(self.pretrain_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path)
                net.load_state_dict(state_dict)

    def resume_others(self, resume_state):
        """
        Resume the optimizers and schedulers for training
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def print_networks(self):
        """
        Print the total number of parameters in the network
        """
        logger.info('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                logger.info('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        logger.info('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
