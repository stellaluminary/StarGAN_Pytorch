import os
import argparse
from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(source, dest, img_size):
    assert os.path.isdir(source), '%s is not a valid directory' % source

    os.makedirs(dest, exist_ok=True)
    f = open(dest + '/gender_img.txt', 'w')

    for root, _, fnames in sorted(os.walk(source)):
        new_path = root.replace(source, dest)
        if not os.path.isdir(new_path):
            os.makedirs(new_path, exist_ok=True)

        gender = 'male' if root.find('male') else 'female'

        for fname in tqdm(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                new_img_path = os.path.join(new_path, fname)

                img = Image.open(path).convert('RGB')
                img = img.resize((img_size, img_size), Image.LANCZOS)
                img.save(new_img_path)

                f.write(gender + ' ' + fname + '\n')
    f.close()

def label_only(source, dest):
    
    os.makedirs(dest, exist_ok=True)

    txt_fname = 'gender_img.txt'
    txt_path = os.path.join(dest, txt_fname)
    f = open(txt_path, 'w')

    for root, _, fnames in sorted(os.walk(source)):
        new_path = root.replace(source, dest)
        if not os.path.isdir(new_path):
            os.makedirs(new_path, exist_ok=True)

        gender_ = root.split('\\')[-1]
        gender = 'female' if gender_ == 'female' else 'male'

        for fname in tqdm(fnames):
            if is_image_file(fname):
                f.write(gender + ' ' + fname + '\n')
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default='./data/celeba_hq')
    parser.add_argument('--dest', type=str, default='./data/celeba_hq_256x256')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--label_only', type=bool, default=False)

    opt = parser.parse_args()
    if opt.dest is None:
        opt.dest = opt.source + '_' + str(opt.img_size) + 'x' + str(opt.img_size)

    if opt.label_only:
        label_only(opt.source, opt.dest)
    else:
        make_dataset(opt.source, opt.dest, opt.img_size)
