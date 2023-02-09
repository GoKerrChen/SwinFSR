from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from pathlib import Path
from PIL import Image
import os
import torchvision.transforms.functional as TF
from itertools import permutations
import numpy as np
import random
from torch.utils.data.dataset import Dataset

class TrainSetLoader(Dataset):
    def __init__(self, cfg):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = cfg.trainset_dir + '/HR/'
        self.file_list = sorted(os.listdir(self.dataset_dir))
        self.transform = transforms.Compose([transforms.ToTensor()])
    def __getitem__(self, index):
          lr_path = str(Path(self.dataset_dir).resolve().parent) + '/LR_x4/'
          hr_folder_img_left = Image.open(self.dataset_dir + self.file_list[index].split('_')[0] + '_' + 'L.png')
          hr_folder_img_right = Image.open(self.dataset_dir + self.file_list[index].split('_')[0] + '_' + 'R.png')
          
          lr_folder_img_left = Image.open(lr_path + self.file_list[index].split('_')[0] + '_' + 'L.png')
          lr_folder_img_right = Image.open(lr_path + self.file_list[index].split('_')[0] + '_' + 'R.png')
          #crop a patch with scale factor = 4
          i,j,h,w = transforms.RandomCrop.get_params(lr_folder_img_left, output_size = (30,90))

          left_lr = TF.crop(lr_folder_img_left, i, j, h, w)
          right_lr = TF.crop(lr_folder_img_right, i, j, h, w)
          left_hr = TF.crop(hr_folder_img_left, i*4, j*4, 4*h, 4*w)
          right_hr = TF.crop(hr_folder_img_right, i*4, j*4, 4*h, 4*w)

          img_hr_left, img_hr_right, img_lr_left, img_lr_right = augmentation(left_hr, right_hr, left_lr, right_lr)
          # img_hr_left, img_lr_left = augmentation(left_hr, left_lr)
          img_hr_left = self.transform(img_hr_left)
          img_hr_right = self.transform(img_hr_right)
          img_lr_left = self.transform(img_lr_left)
          img_lr_right = self.transform(img_lr_right)
          return img_hr_left, img_hr_right, img_lr_left, img_lr_right

    def __len__(self):
        return len(self.file_list)


def rgb2bgr(hr_image_left, hr_image_right, lr_image_left, lr_image_right):
    indices = list(permutations(range(3), 3))
    indices_used = list(indices[np.random.randint(0, len(indices) - 1)])
    # PIL IMAGE to ndarray
    hr_image_left_array = np.array(hr_image_left)
    hr_image_right_array = np.array(hr_image_right)
    lr_image_left_array = np.array(lr_image_left)
    lr_image_right_array = np.array(lr_image_right)
    # RGB shuffle
    hr_image_left_array = hr_image_left_array[..., indices_used]
    hr_image_right_array = hr_image_right_array[..., indices_used]
    lr_image_left_array = lr_image_left_array[..., indices_used]
    lr_image_right_array = lr_image_right_array[..., indices_used]
    # ndarry to PIL IMAGE
    hr_image_left = Image.fromarray(hr_image_left_array, 'RGB')
    hr_image_right = Image.fromarray(hr_image_right_array, 'RGB')
    lr_image_left = Image.fromarray(lr_image_left_array, 'RGB')
    lr_image_right = Image.fromarray(lr_image_right_array, 'RGB')
    return hr_image_left, hr_image_right, lr_image_left, lr_image_right

def augmentation(hr_image_left, hr_image_right, lr_image_left, lr_image_right):
        # print('before_aug',type(hr_image_left))
        augmentation_method = random.choice([0, 1, 2, 3])     
        '''Vertical'''
        if augmentation_method == 0:
            vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)
            hr_image_left = vertical_flip(hr_image_left)
            hr_image_right = vertical_flip(hr_image_right)
            lr_image_left = vertical_flip(lr_image_left)
            lr_image_right = vertical_flip(lr_image_right)
            # print('vf',type(hr_image_left))
            return hr_image_left, hr_image_right, lr_image_left, lr_image_right
        '''Horizontal'''
        if augmentation_method == 1:
            horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
            hr_image_right = horizontal_flip(hr_image_right)
            hr_image_left = horizontal_flip(hr_image_left)
            lr_image_right = horizontal_flip(lr_image_right)
            lr_image_left = horizontal_flip(lr_image_left)
            # print('hf',type(hr_image_left))
            return hr_image_left, hr_image_right, lr_image_left, lr_image_right
        '''no change'''
        if augmentation_method == 2:
            return hr_image_left, hr_image_right, lr_image_left, lr_image_right
        '''RGB shuffle'''
        if augmentation_method == 3:
            hr_image_left, hr_image_right, lr_image_left, lr_image_right = rgb2bgr(hr_image_left, hr_image_right, lr_image_left, lr_image_right)
            return hr_image_left, hr_image_right, lr_image_left, lr_image_right

class TestSetLoader(Dataset):
    def __init__(self, cfg_test):
        self.test_dir = cfg_test.testset_dir + '/HR/'
        self.file_list = sorted(os.listdir(self.test_dir))
        self.transform = transforms.Compose([transforms.ToTensor()])
    def __getitem__(self, index, is_train=False):
        lr_path = str(Path(self.test_dir).resolve().parent) + '/LR_x4/'
        hr_folder_img_left = Image.open(self.test_dir + self.file_list[index].split('_')[0] + '_' + 'L.png')
        hr_folder_img_right = Image.open(self.test_dir + self.file_list[index].split('_')[0] + '_' + 'R.png')  
        lr_folder_img_left = Image.open(lr_path + self.file_list[index].split('_')[0] + '_' + 'L.png')
        lr_folder_img_right = Image.open(lr_path + self.file_list[index].split('_')[0] + '_' + 'R.png')
        # hr_folder_img_left = Image.open(self.test_dir + self.file_list[index] + '/hr0.png')
        # hr_folder_img_right = Image.open(self.test_dir + self.file_list[index] + '/hr1.png')  
        # lr_folder_img_left = Image.open(lr_path + self.file_list[index] + '/lr0.png')
        # lr_folder_img_right = Image.open(lr_path + self.file_list[index] + '/lr1.png')
        img_hr_left = self.transform(hr_folder_img_left)
        img_hr_right = self.transform(hr_folder_img_right)
        img_lr_left = self.transform(lr_folder_img_left)
        img_lr_right = self.transform(lr_folder_img_right)
        return img_hr_left, img_hr_right, img_lr_left, img_lr_right


    def __len__(self):
        return len(self.file_list)