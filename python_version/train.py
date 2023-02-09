from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import argparse
from PIL import Image
import os
from torch.utils.data.dataset import Dataset
import random
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
import torchvision
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
# from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import time
from matplotlib import cm
import copy
# import cv2
# from utils import *
from collections import OrderedDict
from torch.nn.parallel import DataParallel, DistributedDataParallel
from psnr import to_psnr
from model2_3090 import SwinIR
from dataloader import TrainSetLoader, TestSetLoader
import os
torch.backends.cudnn.enabled = True

def model_to_device(net, device):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """

        net = net.to(device)
        # if self.opt['dist']:
        # find_unused_parameters = self.opt.get('find_unused_parameters',
        #                                         False)
        # net = DistributedDataParallel(
        #     net,
        #     device_ids=[torch.cuda.current_device()],
        #     find_unused_parameters=find_unused_parameters)
        # # elif self.opt['num_gpu'] > 1:
        net = DataParallel(net)
        return net
def test_second_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=1)
    # parser.add_argument('--testset_dir', type=str, default='/home/jc/Documents/Ke/NTIRE_2022/SwinIPASSR/SwinIPASSR_files/Test/testset/')
    parser.add_argument('--testset_dir', type=str, default='/home/jc/Documents/Ke/NTIRE_2022/No_War/data/Validation/')
    parser.add_argument('--model_name', type=str, default='No_War_4xSR_epoch')
    parser.add_argument('--TTA', type=bool, default=False)
    return parser.parse_args(args=[])
def train_second_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--start_epoch', type=int, default=1, help='start epoch')
    parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=30, help='number of epochs to update learning rate')
    parser.add_argument('--trainset_dir', type=str, default='/home/jc/Documents/Ke/NTIRE_2022/No_War/data/Train')
    parser.add_argument('--model_name', type=str, default='No_War_4xSR_epoch')
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='/home/jc/Documents/Ke/NTIRE_2022/No_War/log/600_30decay/No_War600_30_stage_2/No_War_4xSR_epoch571.pth.tar')
    parser.add_argument('--training_stage', type=int, default=2)
    
    return parser.parse_args(args=[])
def train_second(train_loader, cfg, test_loader, cfg_test):
  net = SwinIR(upscale=4, img_size=(30, 90),
                     window_size_h=3, window_size_w=9, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                     embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle')
  cudnn.benchmark = True
  scale = cfg.scale_factor
  iteration = 0
#   target_decay = cfg.n_steps


  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  model = torch.load('/home/jc/Documents/Ke/NTIRE_2022/No_War/log/600_30decay/No_War600_30_stage_2/No_War_4xSR_epoch571.pth.tar')
 
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net = model_to_device(net, device)
  state_dict = model['state_dict']
  new_state_dict = OrderedDict()
  for k, v in state_dict.items():
      # name = k[7:] # remove 'module.' of dataparallel
      name = 'module.' + k # remove 'module.' of dataparallel
      # new_state_dict[name]=v

  net.load_state_dict(new_state_dict)

  # Set up tensorboard
  if cfg.training_stage == 1:
      tensorboard_dir = cfg.model_name + str(cfg.n_epochs) + '_' + str(cfg.n_steps) + 'decay/No_War' + str(cfg.n_epochs) + '_' + str(cfg.n_steps) + '_stage_1'
      if not os.path.exists(tensorboard_dir):
              os.makedirs(tensorboard_dir)
  elif cfg.training_stage == 2:
      tensorboard_dir = cfg.model_name + str(cfg.n_epochs) + '_' + str(cfg.n_steps) + 'decay/No_War' + str(cfg.n_epochs) + '_' + str(cfg.n_steps) + '_stage_2'
      if not os.path.exists(tensorboard_dir):
              os.makedirs(tensorboard_dir)
  else:
      tensorboard_dir = cfg.model_name + str(cfg.n_epochs) + '_' + str(cfg.n_steps) + 'decay/No_War' + str(cfg.n_epochs) + '_' + str(cfg.n_steps) + '_stage_3'
      if not os.path.exists(tensorboard_dir):
              os.makedirs(tensorboard_dir)

  # print(tensorboard_dir)
  writer = SummaryWriter(os.path.join('/home/jc/Documents/Ke/NTIRE_2022/No_War/python_version/Tensorboard/'+ tensorboard_dir))
  optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)

  # t = cfg.n_steps  # warmup
  # T = cfg.n_epochs
  # n_t = 0.5
  # lambda1 = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (
  #           1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 else n_t * (
  #           1 + math.cos(math.pi * (epoch - t) / (T - t)))
  # print(lambda1)
  # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
  # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[800, 1500, 2500], gamma=0.5)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
  # if cfg.load_pretrain:
  #     if os.path.isfile(cfg.model_path):
  #         model = torch.load(cfg.model_path, map_location={'cuda:0': device})
  #         net.load_state_dict(model['state_dict'])
          
  #         loss = model['loss']
          
  #         # optimizer.state_dict()['param_groups'][0]['lr'] = 5e-5
  #         model['optimizer_state_dict']['param_groups'][0]['lr'] = 1e-4
  #         model['scheduler_state_dict']['_last_lr'] = 1e-4
  #         # model['scheduler_state_dict']['last_epoch'] = 1490
  #         optimizer.load_state_dict(model['optimizer_state_dict'])
  #         scheduler.load_state_dict(model['scheduler_state_dict'])
  #         cfg.start_epoch = 701
  #         print(model['scheduler_state_dict'])
  #         print(model['optimizer_state_dict']['param_groups'][0]['lr'])
        
  #     else:
  #         print("=> no model found at '{}'".format(cfg.load_model))
  if cfg.load_pretrain:
      if os.path.isfile(cfg.model_path):
          device_id = torch.cuda.current_device()
          model = torch.load(cfg.model_path, map_location=lambda storage, loc: storage.cuda(device_id))
          net.load_state_dict(model['state_dict'])
          optimizer.load_state_dict(model['optimizer_state_dict'])
          loss = model['loss']
          # scheduler.load_state_dict(model['scheduler_state_dict'])
          cfg.start_epoch = model['epoch'] + 1
          print(model['scheduler_state_dict'])
        
      else:
          print("=> no model found at '{}'".format(cfg.load_model))
  
  # cnt_t = 0    
  # for name, param in net.named_parameters():
  #   if param.requires_grad == True:
  #       cnt_t += 1
  #       print(f'name: {name}, {param.requires_grad}')
  # print('grad num', cnt_t)
  criterion_L1 = torch.nn.L1Loss().to(device)

  loss_epoch = []
  loss_list = []
  best_psnr_list = []
  net.train()
  best_model_wts = copy.deepcopy(net.state_dict())
  best_acc = 0.0

  for idx_epoch in range(cfg.start_epoch, cfg.n_epochs + 1):
      start_epoch = time.time()
      current_lr = scheduler.get_last_lr()
      print('current_lr', current_lr)
      # print('op',optimizer.state_dict()['param_groups'][0]['lr'])

      for idx_iter, (HR_left, HR_right, LR_left, LR_right) in enumerate(train_loader):
          iteration += 1
          b, c, h, w = LR_left.shape
          # print('LR_left',LR_left.shape)
          # print('LR_right',LR_right.shape)
          # print('HR_right',HR_right.shape)
          # print('HR_left',HR_left.shape)
          HR_left, HR_right, LR_left, LR_right  = Variable(HR_left).to(device), Variable(HR_right).to(device),\
                                                  Variable(LR_left).to(device), Variable(LR_right).to(device)
          # print('train', LR_left.device)
          # print('train', LR_right.device)
          # print('train', HR_right.device)
          # print('train', HR_left.device)
          
          SR_left, SR_right, (M_right_to_left, M_left_to_right), (V_left, V_right)\
              = net(LR_left, LR_right, is_training=1)
          # print('SR_left',SR_left.shape)
          ''' SR Loss '''
          loss_SR = criterion_L1(SR_left, HR_left) + criterion_L1(SR_right, HR_right)

          ''' Photometric Loss '''
          Res_left = torch.abs(HR_left - F.interpolate(LR_left, scale_factor=scale, mode='bicubic', align_corners=False))
          Res_left = F.interpolate(Res_left, scale_factor=1 / scale, mode='bicubic', align_corners=False)
          Res_right = torch.abs(HR_right - F.interpolate(LR_right, scale_factor=scale, mode='bicubic', align_corners=False))
          Res_right = F.interpolate(Res_right, scale_factor=1 / scale, mode='bicubic', align_corners=False)
          Res_leftT = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w), Res_right.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
          Res_rightT = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), Res_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                  ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
          loss_photo = criterion_L1(Res_left * V_left.repeat(1, 3, 1, 1), Res_leftT * V_left.repeat(1, 3, 1, 1)) + \
                        criterion_L1(Res_right * V_right.repeat(1, 3, 1, 1), Res_rightT * V_right.repeat(1, 3, 1, 1))

          ''' Smoothness Loss '''
          loss_h = criterion_L1(M_right_to_left[:, :-1, :, :], M_right_to_left[:, 1:, :, :]) + \
                    criterion_L1(M_left_to_right[:, :-1, :, :], M_left_to_right[:, 1:, :, :])
          loss_w = criterion_L1(M_right_to_left[:, :, :-1, :-1], M_right_to_left[:, :, 1:, 1:]) + \
                    criterion_L1(M_left_to_right[:, :, :-1, :-1], M_left_to_right[:, :, 1:, 1:])
          loss_smooth = loss_w + loss_h

          ''' Cycle Loss '''
          Res_left_cycle = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w), Res_rightT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                      ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
          Res_right_cycle = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), Res_leftT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                      ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
          loss_cycle = criterion_L1(Res_left * V_left.repeat(1, 3, 1, 1), Res_left_cycle * V_left.repeat(1, 3, 1, 1)) + \
                        criterion_L1(Res_right * V_right.repeat(1, 3, 1, 1), Res_right_cycle * V_right.repeat(1, 3, 1, 1))

          ''' Consistency Loss '''
          SR_left_res = F.interpolate(torch.abs(HR_left - SR_left), scale_factor=1 / scale, mode='bicubic', align_corners=False)
          SR_right_res = F.interpolate(torch.abs(HR_right - SR_right), scale_factor=1 / scale, mode='bicubic', align_corners=False)
          SR_left_resT = torch.bmm(M_right_to_left.detach().contiguous().view(b * h, w, w), SR_right_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                    ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
          SR_right_resT = torch.bmm(M_left_to_right.detach().contiguous().view(b * h, w, w), SR_left_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                    ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
          loss_cons = criterion_L1(SR_left_res * V_left.repeat(1, 3, 1, 1), SR_left_resT * V_left.repeat(1, 3, 1, 1)) + \
                      criterion_L1(SR_right_res * V_right.repeat(1, 3, 1, 1), SR_right_resT * V_right.repeat(1, 3, 1, 1))

          ''' Total Loss '''
          loss = loss_SR + 0.1 * loss_cons + 0.1 * (loss_photo + loss_smooth + loss_cycle)
          

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          # Load data to the tensorboard
          writer.add_scalars('training', {'training total loss': loss.item()
                                      }, iteration)

          loss_epoch.append(loss.data.cpu())
      scheduler.step()
      loss_list.append(float(np.array(loss_epoch).mean()))
      print('current lr', optimizer.state_dict()['param_groups'][0]['lr'])
      print('Epoch--%4d, loss--%f' %
            (idx_epoch, float(np.array(loss_epoch).mean())))
      end_epoch = time.time()
      print('one epoch time', end_epoch-start_epoch) 
      loss_epoch = []
      if idx_epoch % 10 == 0:
          # print(scheduler.state_dict())
          torch.save({'epoch': idx_epoch, 'state_dict': net.state_dict(),'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'loss': loss},
            '/home/jc/Documents/Ke/NTIRE_2022/No_War/python_version/log/' + str(cfg.n_epochs) + '_' + str(cfg.n_steps) + 'decay/No_War' + str(cfg.n_epochs) + '_' + str(cfg.n_steps) + '_stage_' + str(cfg.training_stage) + '/' + cfg.model_name + str(idx_epoch) + '.pth.tar')

      # start testing every 10 epochs    
      if idx_epoch % 10 == 0: 
          start_test = time.time()         
          with torch.no_grad():
              left_psnr_list = []
              right_psnr_list = []
              left_ssim_list = []
              right_ssim_list = []
              avr_psnr_list = []
              avr_ssim_list = []
              
              net.eval()

              for idx_iter_test, (HR_left, HR_right, LR_left, LR_right) in enumerate(test_loader):
                  HR_left, HR_right, LR_left, LR_right  = Variable(HR_left).to(device), Variable(HR_right).to(device),\
                                                  Variable(LR_left).to(device), Variable(LR_right).to(device) 
                  # print('test', LR_left.device)   
                  # print('test', LR_right.device) 
                  # print('test', HR_left.device) 
                  # print('test', HR_right.device)      
                  if cfg_test.TTA:
                    SR_left1, SR_right1 = net(LR_left, LR_right, is_training=0)
                    SR_left2, SR_right2 = net(LR_left.flip(dims=(1,)), LR_right.flip(dims=(1,)), is_training=0)
                    SR_left3, SR_right3 = net(LR_left.flip(dims=(2,)), LR_right.flip(dims=(2,)), is_training=0)
                    SR_left4, SR_right4 = net(LR_left.flip(dims=(3,)), LR_right.flip(dims=(3,)), is_training=0)
                    SR_left5, SR_right5 = net(LR_left.flip(dims=(1, 2)), LR_right.flip(dims=(1, 2)), is_training=0)
                    SR_left6, SR_right6 = net(LR_left.flip(dims=(1, 3)), LR_right.flip(dims=(1, 3)), is_training=0)
                    SR_left7, SR_right7 = net(LR_left.flip(dims=(2, 3)), LR_right.flip(dims=(2, 3)), is_training=0)
                    SR_left8, SR_right8 = net(LR_left.flip(dims=(1, 2, 3)), LR_right.flip(dims=(1, 2, 3)),is_training=0)
                    SR_left = (SR_left1 + SR_left2.flip(dims=(1,)) + SR_left3.flip(dims=(2,)) + SR_left4.flip(dims=(3,)) \
                                + SR_left5.flip(dims=(1, 2)) + SR_left6.flip(dims=(1, 3)) + SR_left7.flip(
                                dims=(2, 3)) + SR_left8.flip(dims=(1, 2, 3))) / 8.
                    SR_right = (SR_right1 + SR_right2.flip(dims=(1,)) + SR_right3.flip(dims=(2,)) + SR_right4.flip(dims=(3,)) \
                                + SR_right5.flip(dims=(1, 2)) + SR_right6.flip(dims=(1, 3)) + SR_right7.flip(
                                dims=(2, 3)) + SR_right8.flip(dims=(1, 2, 3))) / 8.
                  else:
                    SR_left, SR_right = net(LR_left, LR_right, is_training=0)          
                  torch.cuda.empty_cache()
                  SR_left, SR_right = torch.clamp(SR_left, 0, 1), torch.clamp(SR_right, 0, 1)

                  # evaluation the PSNR and SSIM value of the model
                  left_psnr_list.extend(to_psnr(SR_left, HR_left))
                  right_psnr_list.extend(to_psnr(SR_right, HR_right))

                  one_psnr = np.array(left_psnr_list) + np.array(right_psnr_list)
                  # one_ssim = np.array(left_ssim_list) + np.array(right_ssim_list)

                  avr_psnr_list.extend(one_psnr/2)
                  # avr_ssim_list.extend(one_ssim/2)
                  
              avr_psnr = sum(avr_psnr_list)/len(avr_psnr_list)
              # avr_ssim = sum(avr_ssim_list)/len(avr_ssim_list)
              if avr_psnr > best_acc:
                best_acc = avr_psnr
                best_model_wts = copy.deepcopy(net.state_dict())
              best_psnr_list.append(avr_psnr)
              
              print('reconstructed_avg_psnr: ', avr_psnr)
              end_test = time.time()
              print('one test time', end_test-start_test)
              writer.add_scalars('testing', {'testing_psnr':avr_psnr}, idx_epoch)
              # print('reconstructed_avg_ssim: ', avr_ssim)

              # writer.add_scalars('testing', {'testing_ssim': avr_ssim}, idx_epoch)
              
  best_psnr = max(best_psnr_list)
#   best_net = net.load_state_dict(best_model_wts)
  # index = 0 means epoch 10, index = 1 means epoch 
  if cfg.load_pretrain:
    best_epoch =(best_psnr_list.index(best_psnr)+1)*10 + cfg.start_epoch
  else:
    best_epoch = (best_psnr_list.index(best_psnr)+1)*10
  print('best_performance_epoch: ', best_epoch, ', avg_psnr: ', best_psnr, ', best_psnr: ', best_acc)
  torch.save({'epoch': best_epoch, 'state_dict': best_model_wts, 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'loss': loss},
            '/home/jc/Documents/Ke/NTIRE_2022/No_War/python_version/log/' + str(cfg.n_epochs) + '_' + str(cfg.n_steps) + 'decay/No_War' + str(cfg.n_epochs) + '_' + str(cfg.n_steps) + '_stage_' + str(cfg.training_stage) + '/' + cfg.model_name + str(best_epoch) + '.pth.tar')
  writer.close()

def main(cfg_sec, cfg_test_sec):
    train_set = TrainSetLoader(cfg_sec)
    train_sec_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=cfg_sec.batch_size, shuffle=True, pin_memory = True, drop_last= True)
    test_set = TestSetLoader(cfg_test_sec)
    test_sec_loader = DataLoader(dataset=test_set)
    log_base_dir = '/home/jc/Documents/Ke/NTIRE_2022/No_War/python_version/log/'
    if cfg_sec.training_stage == 1:
            log_dir = log_base_dir + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + 'decay/No_War' + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + '_stage_1'
            if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
    elif cfg_sec.training_stage == 2:
            log_dir = log_base_dir + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + 'decay/No_War' + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + '_stage_2'
            if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
    else:
            log_dir = log_base_dir + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + 'decay/No_War' + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + '_stage_3'
            if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
    print(log_dir)
    Tensorboard_base_dir = '/home/jc/Documents/Ke/NTIRE_2022/No_War/python_version/Tensorboard/'
    if cfg_sec.training_stage == 1:
            tensorboard_dir = Tensorboard_base_dir + cfg_sec.model_name + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + 'decay/No_War' + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + '_stage_1'
            if not os.path.exists(tensorboard_dir):
                    os.makedirs(tensorboard_dir)
    elif cfg_sec.training_stage == 2:
            tensorboard_dir = Tensorboard_base_dir + cfg_sec.model_name + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + 'decay/No_War' + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + '_stage_2'
            if not os.path.exists(tensorboard_dir):
                    os.makedirs(tensorboard_dir)
    else:
            tensorboard_dir = Tensorboard_base_dir + cfg_sec.model_name + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + 'decay/No_War' + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + '_stage_3'
            if not os.path.exists(tensorboard_dir):
                    os.makedirs(tensorboard_dir)

    train_second(train_sec_loader, cfg_sec, test_sec_loader, cfg_test_sec)


if __name__ == '__main__':
    cfg_sec = train_second_parse_args()
    cfg_test_sec = test_second_parse_args()
    main(cfg_sec, cfg_test_sec)

