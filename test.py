import argparse
from configs import options
from data_loader import create_dataset
from models import create_model
from utils import util
import logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', '-o',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/stargan.yaml')
    args = parser.parse_args()
    opt = options.parser(args)

    opt['Setting']['phase'] = 'test'

    util.setup_logger(None, opt['Path']['log_file_path'], 'test', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info('Test Model Name : ' + opt['Model_Param']['model_name'])

    dataset, data_loader = create_dataset(opt, image_dir=opt['Path']['Data_val'])
    opt['dataset_size'] = len(data_loader)
    logger.info('The number of testing images = %d' % opt['dataset_size'])

    model = create_model(opt)
    model.print_networks()
    model.load_pretrained_nets()
    model.eval()

    for idx, data in enumerate(data_loader):

        model.feed_data(data)
        model.forward()

        images = model.get_current_visuals()
        vis_path = model.make_visual_dir(opt['Path']['pretrain_res'])
        util.save_current_imgs(images=images, save_dirs=vis_path, phase=opt['Setting']['phase'],
                                id=idx, min_max=(-1, 1))

if __name__ == '__main__':
    main()