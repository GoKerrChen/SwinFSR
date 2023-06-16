from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import argparse
from swin_cam_ffc import SwinIR
import os
import torch
from torchvision import transforms

def test_sec_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='./dataset/NTIRE2023Test')
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--TTA', type=bool, default=False)
    parser.add_argument('--save_path', type=str, default='./results/ntire2023test')
    return parser.parse_args(args=[])

def test_sec(cfg_test_sec):
    net = SwinIR(upscale=4, img_size=(30, 90),
                     window_size_h=6, window_size_w=15, img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                     embed_dim=180, drop_path_rate=0.2, num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle').to(cfg_test_sec.device)

    model = torch.load(cfg_test_sec.model_path)
    net.load_state_dict(model['state_dict'])
    # sort all the files in the validation set in order to make it LR paire
    file_list = sorted(os.listdir(cfg_test_sec.testset_dir + '/LR_x4_2/'))
    net.eval()

    for idx in range(0, len(file_list), 2):
      LR_left = Image.open(cfg_test_sec.testset_dir + '/LR_x4_2/' + file_list[idx].split('_')[0] + '_' + 'L.png')
      LR_right = Image.open(cfg_test_sec.testset_dir + '/LR_x4_2/' + file_list[idx].split('_')[0] + '_' + 'R.png')

      LR_left, LR_right = ToTensor()(LR_left), ToTensor()(LR_right)
      LR_left, LR_right = LR_left.unsqueeze(0), LR_right.unsqueeze(0)
      LR_left, LR_right = Variable(LR_left).to(cfg_test_sec.device), Variable(LR_right).to(cfg_test_sec.device)
      
      scene_name_left = file_list[idx]
      print(
          'Running Scene ' + scene_name_left + ' of Flickr1024 Dataset......')
      with torch.no_grad():
        if cfg_test_sec.TTA:
            SR_left1, SR_right1 = net(LR_left, LR_right)
            SR_left2, SR_right2 = net(LR_left.flip(dims=(1,)), LR_right.flip(dims=(1,)))
            SR_left3, SR_right3 = net(LR_left.flip(dims=(2,)), LR_right.flip(dims=(2,)))
            SR_left4, SR_right4 = net(LR_left.flip(dims=(3,)), LR_right.flip(dims=(3,)))
            SR_left5, SR_right5 = net(LR_left.flip(dims=(1, 2)), LR_right.flip(dims=(1, 2)))
            SR_left6, SR_right6 = net(LR_left.flip(dims=(1, 3)), LR_right.flip(dims=(1, 3)))
            SR_left7, SR_right7 = net(LR_left.flip(dims=(2, 3)), LR_right.flip(dims=(2, 3)))
            SR_left8, SR_right8 = net(LR_left.flip(dims=(1, 2, 3)), LR_right.flip(dims=(1, 2, 3)))
            SR_left = (SR_left1 + SR_left2.flip(dims=(1,)) + SR_left3.flip(dims=(2,)) + SR_left4.flip(
                dims=(3,)) + SR_left5.flip(dims=(1, 2)) + SR_left6.flip(dims=(1, 3)) + SR_left7.flip(
                dims=(2, 3)) + SR_left8.flip(dims=(1, 2, 3))) / 8.
            SR_right = (SR_right1 + SR_right2.flip(dims=(1,)) + SR_right3.flip(dims=(2,)) + SR_right4.flip(
                dims=(3,)) + SR_right5.flip(dims=(1, 2)) + SR_right6.flip(dims=(1, 3)) + SR_right7.flip(
                dims=(2, 3)) + SR_right8.flip(dims=(1, 2, 3))) / 8.
        else:
            SR_left, SR_right = net(LR_left, LR_right)
        SR_left, SR_right = torch.clamp(SR_left, 0, 1), torch.clamp(SR_right, 0, 1)

      if not os.path.exists(cfg_test_sec.save_path):
          os.makedirs(cfg_test_sec.save_path)

      SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
      SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0))

      SR_left_img.save(cfg_test_sec.save_path + '/' + file_list[idx].split('_')[0] + '_' + 'L.png')
      SR_right_img.save(cfg_test_sec.save_path + '/' + file_list[idx].split('_')[0] + '_' + 'R.png')

def main(cfg_test_sec):
    test_sec(cfg_test_sec)
    print('Finished!')

if __name__ == '__main__':
    cfg_test_sec= test_sec_parse_args()
    main(cfg_test_sec)    

