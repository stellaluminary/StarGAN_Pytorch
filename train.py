import argparse
import torch
import os
from configs import options
from data_loader import create_dataset
from models import create_model
from utils import util
import math
import time
import logging
import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', '-o',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/stargan.yaml')
    #parser.add_argument('--train_img_dir', type=str, default='data/celeba_hq/train')
    #parser.add_argument('--val_img_dir', type=str, default='data/celeba_hq/test')

    args = parser.parse_args()
    opt = options.parser(args)

    util.setup_logger(None, opt['Path']['log_file_path'], 'train', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info('Train Model Name : ' + opt['Model_Param']['model_name'])
    logger.info(opt)

    # train from scratch OR resume training
    if opt['Path']['resume_state']:  # resuming training
        resume_state = torch.load(opt['Path']['resume_state'])
        logger.info('Resume training')
    else:  # training from scratch
        logger.info('Training from scratch')
        resume_state = None
        util.mkdir(opt['Path']['log_file_path'])
        util.mkdir(opt['Path']['resume_model_dir'])
        util.mkdir(opt['Path']['pretrain_model_dir'])
        util.mkdir_and_rename(opt['Path']['save_img'])
        util.mkdir_and_rename(opt['Path']['checkpoint_dir'])

    dataset, data_loader = create_dataset(opt, image_dir=opt['Path']['Data_train'])
    opt['dataset_size'] = len(data_loader)
    #dataset_iter_per_epoch = math.ceil(opt['dataset_size'] / opt['Data_Param']['batch_size'])

    logger.info('The number of training iterations = %d' % opt['dataset_size'])
    #logger.info('Training iterations per epoch = %d' % dataset_iter_per_epoch)
    logger.info('Class label ' + ' '.join(dataset.classes))

    model = create_model(opt)
    model.print_networks()

    if resume_state:
        init_epoch = resume_state['epoch'] + 1
        total_iters = opt['dataset_size'] * resume_state['epoch']
        model.resume_networks(resume_state['epoch'])
        model.resume_others(resume_state)
    else:
        init_epoch = 0
        total_iters = 0
        util.init_log_file(os.path.join(opt['Path']['log_file_path'], 'loss_file.txt'),
                           model.loss_names, model.add_value_names)

    for epoch in range(init_epoch, opt['Train']['n_epochs']):
        start_time = time.time()
        for idx, data in enumerate(data_loader):
            total_iters += 1
            # feed the dict data : data & path
            model.feed_data(data)
            model.optimize_parameters(total_iters)

            # print & log the losses and values
            if (idx+1) % opt['Save']['print_iter'] == 0:
                logger.info('(epoch: %d/%d, iters: %d/%d, total iters: %d) '
                            % (epoch, opt['Train']['n_epochs'], idx+1, opt['dataset_size'], total_iters))

                f_txt = ''
                losses = model.get_current_losses()
                add_val = model.get_current_add_values()

                for k, v in losses.items():
                    f_txt += 'loss %s: %.3f ' % (k, v)

                for k, v in add_val.items():
                    if k == 'lr':
                        f_txt += '%s: %.8f ' % (k, v)
                    else:
                        f_txt += '%s: %.3f ' % (k, v)
                logger.info(f_txt)

            # get and save the result images
            if (idx+1) % opt['Save']['save_img_iter'] == 0:
                images = model.get_current_visuals()
                vis_path = model.make_visual_dir(opt['Path']['save_img'])
                util.save_current_imgs(images=images, save_dirs=vis_path, phase=opt['Setting']['phase'], 
                                        id=idx+1, epoch=epoch, min_max=(-1, 1))

        # update the learning rate based on schedulers
        model.update_learning_rate()

        # save networks and optimizers, schedulers
        model.save_networks(epoch)
        model.save_training_state(epoch=epoch)

        # print the time consume in each epoch
        util.print_time(epoch, start_time)

if __name__ == '__main__':
    main()