import numpy as np
import os
import sys
import ntpath
import time
import torch

def save_checkpoint(model, optimizer, filename='my_check_point.pth.tar'):
    print('Save Checkpoint')
    checkpoint= {
        'state_dict' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

