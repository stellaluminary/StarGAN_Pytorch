from .base_model import BaseModel
import models.networks as networks
import logging
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

logger = logging.getLogger('base')

class StarGANModel(BaseModel):
    def __init__(self, opt):
        super(StarGANModel, self).__init__(opt)

        #self.fixed_noise = torch.randn(1, self.opt['Model_Param']['nz']).to(self.device)
        self.batch_size = self.opt['Data_Param']['batch_size']
        self.c_dim = self.opt['Model_Param']['c_dim']

        # define self.model_names for saving pth file
        if self.is_train == 'train':
            self.model_names = ['G', 'D']
            self.visual_names = ['imgs', 'fake', 'recov_fake']
        else:
            self.model_names = ['G']
            self.visual_names = ['imgs', 'fake']

        # define self.loss_names for saving loss log file
        self.loss_names = ['G', 'D']
        self.criterion_cycle = torch.nn.L1Loss()

        # define generator and discriminator
        self.netG = networks.define_G(opt, 'G').to(self.device)

        if self.is_train == 'train':
            self.netD = networks.define_D(opt, 'D').to(self.device)
            self.netG.train()
            self.netD.train()

            # define optimizers : D & G
            self.optim_G = optim.Adam(params=self.netG.parameters(), lr=opt['Train']['lr'],
                                      betas=(opt['Train']['beta1'], opt['Train']['beta2']))
            self.optim_D = optim.Adam(params=self.netD.parameters(), lr=opt['Train']['lr'],
                                      betas=(opt['Train']['beta1'], opt['Train']['beta2']))

            self.optimizers.append(self.optim_G)
            self.optimizers.append(self.optim_D)

            self.scheduler_G = optim.lr_scheduler.LambdaLR(self.optim_G, lr_lambda=self.learning_rate_decay_func)
            self.scheduler_D = optim.lr_scheduler.LambdaLR(self.optim_D, lr_lambda=self.learning_rate_decay_func)

            self.schedulers.append(self.scheduler_G)
            self.schedulers.append(self.scheduler_D)

    def feed_data(self, data):
        self.imgs = data[0].to(self.device)
        self.labels = data[1].view(data[1].size(0), self.c_dim).to(self.device)
        self.sampled_c = torch.FloatTensor(np.random.randint(0, 2, (self.imgs.size(0), self.c_dim))).to(self.device)
        #print(self.imgs.size(), self.labels.size(), self.sampled_c.size())

    def forward(self):

        self.fake = self.netG(self.imgs, self.sampled_c)  # x: img -> G(x): netG(x) = fake img
        self.recov_fake = self.netG(self.fake, self.labels)

    def optimize_parameters(self, idx):

        # ------ define fake data ------

        self.forward()

        # -------------------------- train generator G --------------------------

        if idx % self.opt['Train']['n_critic'] == 0:
            self.set_requires_grad([self.netD], requires_grad=False)

            disc_fake, pred_cls = self.netD(self.fake)
            # Adversarial loss
            loss_g_adv = -torch.mean(disc_fake)
            # Classification loss
            loss_g_cls = self.classification_loss(pred_cls, self.sampled_c)
            # Reconstruction loss
            loss_g_rec = self.criterion_cycle(self.imgs, self.recov_fake)
            self.loss_G = loss_g_adv + loss_g_cls * self.opt['Train']['lambda_cls'] + loss_g_rec * self.opt['Train']['lambda_rec']

            self.optim_G.zero_grad()
            self.loss_G.backward()
            self.optim_G.step()

        # -------------------------- train discriminator D --------------------------

        self.set_requires_grad([self.netD], requires_grad=True)

        disc_real, pred_cls = self.netD(self.imgs)
        disc_fake, _ = self.netD(self.fake.detach())

        gp = self.gradient_penalty(self.imgs, self.fake.detach())
        # Adversarial loss
        loss_d_adv = -(torch.mean(disc_real) - torch.mean(disc_fake)) + self.opt['Train']['lambda_gp'] * gp
        # Classification loss
        loss_d_cls = self.classification_loss(pred_cls, self.labels.float())

        self.loss_D = loss_d_adv + loss_d_cls * self.opt['Train']['lambda_cls']

        self.optim_D.zero_grad()
        self.loss_D.backward()
        self.optim_D.step()

    def test(self):
        self.modi_labels = torch.zeros_like(self.labels)
        for e, val in enumerate(self.labels):
            if self.labels[e][0] == 1:
                self.modi_labels[e][0] = 0
            else:
                self.modi_labels[e][0] = 1

        print(self.labels)
        print(self.modi_labels)

        with torch.no_grad():
            self.fake = self.netG(self.imgs, self.modi_labels)

    def gradient_penalty(self, real_samples, fake_samples):
        BATCH_SIZE, C, H, W = real_samples.shape
        # uniform distribution alpha
        alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(self.device)
        interpolated_images = real_samples * alpha + fake_samples * (1 - alpha)

        d_interpolate, _ = self.netD(interpolated_images.requires_grad_(True))

        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=d_interpolate,
            grad_outputs=torch.ones_like(d_interpolate),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty

    def classification_loss(self, logit, target):
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

    def learning_rate_decay_func(self, iterations):
        if iterations <= 150000:
            return self.opt['Train']['lr']
        else:
            self.opt['Train']['lr'] -= self.opt['Train']['lr'] / float(150000)
            return self.opt['Train']['lr']