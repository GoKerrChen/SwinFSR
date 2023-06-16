from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import argparse
from PIL import Image
import os
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import math
import time
import copy
from swin_cam_ffc import SwinIR
from psnr import to_psnr
from dataloader import TrainSetLoader, TestSetLoader

torch.backends.cudnn.enabled = True

def test_second_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--testset_dir', type=str, default='./dataset/Validation/')
    return parser.parse_args(args=[])
def train_second_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--start_epoch', type=int, default=1, help='start epoch')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
    parser.add_argument('--trainset_dir', type=str, default='./dataset/Train')
    parser.add_argument('--model_name', type=str, default='4xSR_epoch')
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--training_stage', type=int, default=2)
    parser.add_argument('--Tensorboard_path', type=str, default='./Tensorboard/')
    parser.add_argument('--log_path', type=str, default='./log/')
    return parser.parse_args(args=[])

def train_second(train_loader, cfg, test_loader, cfg_test):
    net = SwinIR(upscale=4, img_size=(30, 90),
                     window_size_h=6, window_size_w=15, img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                     embed_dim=180, drop_path_rate=0.2, num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle')

    cudnn.benchmark = True
    scale = cfg.scale_factor
    iteration = 0
    target_decay = cfg.n_steps

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)


    # Set up tensorboard
    Tensorboard_base_dir = './Tensorboard/'
    if cfg_sec.training_stage == 1:
            tensorboard_dir = Tensorboard_base_dir + cfg_sec.model_name + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + 'decay/' + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + '_stage_1'
            if not os.path.exists(tensorboard_dir):
                    os.makedirs(tensorboard_dir)
    elif cfg_sec.training_stage == 2:
            tensorboard_dir = Tensorboard_base_dir + cfg_sec.model_name + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + 'decay/' + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + '_stage_2'
            if not os.path.exists(tensorboard_dir):
                    os.makedirs(tensorboard_dir)
    else:
            tensorboard_dir = Tensorboard_base_dir + cfg_sec.model_name + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + 'decay/' + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + '_stage_3'
            if not os.path.exists(tensorboard_dir):
                    os.makedirs(tensorboard_dir)

    # print(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)

    t = cfg.n_steps  # warmup
    T = cfg.n_epochs
    n_t = 0.5
    lambda1 = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (
              1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 else n_t * (
              1 + math.cos(math.pi * (epoch - t) / (T - t)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location=device)
            net.load_state_dict(model['state_dict'])
            optimizer.load_state_dict(model['optimizer_state_dict'])
            loss = model['loss']
            scheduler.load_state_dict(model['scheduler_state_dict'])
            cfg.start_epoch = model['epoch'] + 1
            print(model['scheduler_state_dict'])
          
        else:
            print("=> no model found at '{}'".format(cfg.load_model))
    
    criterion_L1 = torch.nn.L1Loss().to(device)

    loss_epoch = []
    loss_list = []
    best_psnr_list = []
    net.train()
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0

    for idx_epoch in range(cfg.start_epoch, cfg.n_epochs + 1):
        start_epoch = time.time()
        print('current_lr',optimizer.state_dict()['param_groups'][0]['lr'])

        for idx_iter, (HR_left, HR_right, LR_left, LR_right) in enumerate(train_loader):
            iteration += 1
            b, c, h, w = LR_left.shape

            HR_left, HR_right, LR_left, LR_right  = Variable(HR_left).to(device), Variable(HR_right).to(device),\
                                                    Variable(LR_left).to(device), Variable(LR_right).to(device)
            SR_left, SR_right = net(LR_left, LR_right)
            ''' SR Loss '''
            loss_SR = criterion_L1(SR_left, HR_left) + criterion_L1(SR_right, HR_right)


            # ''' Total Loss '''
            loss = loss_SR

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
        if idx_epoch % 5 == 0:
            # print(scheduler.state_dict())
            torch.save({'epoch': idx_epoch, 'state_dict': net.state_dict(),'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'loss': loss},
              cfg.log_path + str(cfg.n_epochs) + '_' + str(cfg.n_steps) + 'decay/' + str(cfg.n_epochs) + '_' + str(cfg.n_steps) + '_stage_' + str(cfg.training_stage) + '/' + cfg.model_name + str(idx_epoch) + '.pth.tar')

        # start testing every 10 epochs    
        if idx_epoch % 5 == 0: 

          start_test = time.time()         
          with torch.no_grad():
              left_psnr_list = []
              right_psnr_list = []
              avr_psnr_list = []

              net.eval()

              for idx_iter_test, (HR_left, HR_right, LR_left, LR_right) in enumerate(test_loader):
                  HR_left, HR_right, LR_left, LR_right  = Variable(HR_left).to(device), Variable(HR_right).to(device),\
                                                  Variable(LR_left).to(device), Variable(LR_right).to(device)      
                 
                  SR_left, SR_right = net(LR_left, LR_right)          
                  torch.cuda.empty_cache()
                  SR_left, SR_right = torch.clamp(SR_left, 0, 1), torch.clamp(SR_right, 0, 1)

                  # evaluation the PSNR and SSIM value of the model
                  left_psnr_list.extend(to_psnr(SR_left, HR_left))
                  right_psnr_list.extend(to_psnr(SR_right, HR_right))

                  one_psnr = np.array(left_psnr_list) + np.array(right_psnr_list)

                  avr_psnr_list.extend(one_psnr/2)

                  
              avr_psnr = sum(avr_psnr_list)/len(avr_psnr_list)

              if avr_psnr > best_acc:
                best_acc = avr_psnr
                best_model_wts = copy.deepcopy(net.state_dict())
              best_psnr_list.append(avr_psnr)
              
              print('reconstructed_avg_psnr: ', avr_psnr)
              end_test = time.time()
              print('one test time', end_test-start_test)
              writer.add_scalars('testing', {'testing_psnr':avr_psnr}, idx_epoch)
                
    best_psnr = max(best_psnr_list)
    if cfg.load_pretrain:
      best_epoch =(best_psnr_list.index(best_psnr)+1)*10 + cfg.start_epoch
    else:
      best_epoch = (best_psnr_list.index(best_psnr)+1)*10
    print('best_performance_epoch: ', best_epoch, ', avg_psnr: ', best_psnr, ', best_psnr: ', best_acc)
    torch.save({'epoch': best_epoch, 'state_dict': best_model_wts, 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'loss': loss},
              cfg.log_path + str(cfg.n_epochs) + '_' + str(cfg.n_steps) + 'decay/' + str(cfg.n_epochs) + '_' + str(cfg.n_steps) + '_stage_' + str(cfg.training_stage) + '/' + cfg.model_name + str(best_epoch) + '.pth.tar')
    writer.close()

def main(cfg_sec, cfg_test_sec):
    train_set = TrainSetLoader(cfg_sec)
    train_sec_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=cfg_sec.batch_size, shuffle=True, pin_memory = True, drop_last= True)
    test_set = TestSetLoader(cfg_test_sec)
    test_sec_loader = DataLoader(dataset=test_set)
    log_base_dir = './log/'
    if cfg_sec.training_stage == 1:
            log_dir = log_base_dir + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + 'decay/' + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + '_stage_1'
            if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
    elif cfg_sec.training_stage == 2:
            log_dir = log_base_dir + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + 'decay/' + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + '_stage_2'
            if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
    else:
            log_dir = log_base_dir + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + 'decay/' + str(cfg_sec.n_epochs) + '_' + str(cfg_sec.n_steps) + '_stage_3'
            if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
    print('checkpoint save to:', log_dir)
    

    train_second(train_sec_loader, cfg_sec, test_sec_loader, cfg_test_sec)


if __name__ == '__main__':
    cfg_sec = train_second_parse_args()
    cfg_test_sec = test_second_parse_args()
    main(cfg_sec, cfg_test_sec)

