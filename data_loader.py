import os
import torch
from PIL import Image
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import logging

def tensor_transforms(opt):
    tf_list = []
    if opt['Setting']['phase'] == 'train':
        tf_list.append(transforms.RandomHorizontalFlip())
    tf_list.append(transforms.Resize(opt['Model_Param']['img_size'], Image.LANCZOS))
    tf_list.append(transforms.ToTensor())
    tf_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(tf_list)

def create_dataset(opt, image_dir):
    dataset = ImageFolder(image_dir, tensor_transforms(opt))
    if opt['Setting']['phase'] == 'train':
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=opt['Data_Param']['batch_size'],
                                      shuffle=True,
                                      num_workers=opt['Data_Param']['num_threads'])
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=1)
    return dataset, data_loader


# if __name__ == '__main__':
#
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
#     # log 출력
#     stream_handler = logging.StreamHandler()
#     stream_handler.setFormatter(formatter)
#     logger.addHandler(stream_handler)
#
#     # log를 파일에 출력
#     file_handler = logging.FileHandler('my.txt')
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)
#
#     transform = []
#     transform.append(transforms.RandomHorizontalFlip())
#     transform.append(transforms.Resize(256))
#     transform.append(transforms.ToTensor())
#     transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
#     transform = transforms.Compose(transform)
#
#     dataset = ImageFolder('./data/celeba_hq_256x256/train', transform)
#     data_loader = data.DataLoader(dataset=dataset,
#                                       batch_size=1,
#                                       shuffle=True,
#                                       num_workers=4)
#
#     for e, i in enumerate(data_loader):
#         logger.info(f'{e}번째 batch 실행')
#         if e == 10:
#             break
#
#     print(dataset.classes)

