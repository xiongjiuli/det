import os 
import argparse
import torchio as tio
# import albumentations as A
from torchvision import transforms as T
# from dataset.data import Dataset
from model.resnet import CenterNet
# from tensorboardX import SummaryWriter
import torch.utils.data as data
import torch 
from utils import focal_loss, reg_l1_loss
from tqdm import tqdm 
import numpy as np
import cv2
from IPython import embed
from data.data_v1 import luna16Dataset
import logging
# from utils import npy2nii

# 创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 创建一个handler，用于将日志输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# 创建一个handler，用于将日志写入到文件中
fh = logging.FileHandler('training.log')
fh.setLevel(logging.INFO)

# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# 将handler添加到logger中
logger.addHandler(ch)
logger.addHandler(fh)



def cycle(dl):
    while True:
        for data in dl:
            yield data

def save(step, opt, model, out_path, name='model'):

        data = {
            'step': step,
            'model': model.state_dict(),
            'opt': opt.state_dict()
        }

        torch.save(data, os.path.join(out_path, f'{name}-{step}.pt'))  

def npy2nii(image_npy, suffix=''):
    affine = np.array([[0.7, 0, 0, 0], [0, 0.7, 0, 0], [0, 0, 1.2, 0], [0, 0, 0, 1]])
    if isinstance(image_npy, np.ndarray):
        image_npy = torch.from_numpy(image_npy)
    if len(image_npy.shape) == 3:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy.unsqueeze(0), affine=affine)
        image_nii.save('./save/{}.nii.gz'.format(suffix))
    elif len(image_npy.shape) == 4:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy, affine=affine)
        image_nii.save('./save/{}.nii.gz'.format(suffix))
    elif len(image_npy.shape) == 5:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy[0, :, :, :,:], affine=affine)
        image_nii.save('./save/{}.nii.gz'.format(suffix))
    else: 
        print('DIM ERROR : npy.dim != 3 or 4 or 5')


def train(config):
    
    # augmentation if needed
    
    train_dataset = luna16Dataset('train')
    valid_dataset = luna16Dataset('valid')
    
    train_loader = cycle(data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4, drop_last=True))
    valid_loader = cycle(data.DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=False, num_workers=4))
    
    
    model = CenterNet(config.backbone_name, config.num_classes)
    # model = CenterNet(config.backbone_name, 3)
    # model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), config.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.scheduler_steps, gamma=config.gamma)
    
    # if config.pretrained_model:
    #     model.load_state_dict(torch.load(config.pretrained_model)['model'], strict=False)
    
    best_loss = 1e4

    with tqdm(total=config.total_steps) as pbar:
        for step in range(config.total_steps):

            if (step) % config.save_freq == 0:
            # if (step) % 100 == 0:
                model.eval()
                os.makedirs(config.save_dir, exist_ok=True)
                save(step, optimizer, model, config.save_dir)

                # multi-scale test could do to improve performance
                valid_loss = []
                with torch.no_grad():
                    # for val_step in range(config.valid_steps):
                    for val_step in range(100):
                        valid_batch = next(valid_loader)
                        # image = valid_batch['input'].cuda().unsqueeze(0)
                        # heatmap = valid_batch['hmap'].cuda().unsqueeze(0)
                        # wh_size = valid_batch['whd'].cuda()
                        # regression = valid_batch['offset'].cuda()
                        # masks = valid_batch['mask'].cuda()
                        image = valid_batch['input'].unsqueeze(0)
                        heatmap = valid_batch['hmap'].unsqueeze(0)
                        wh_size = valid_batch['whd']
                        regression = valid_batch['offset']
                        masks = valid_batch['mask']
                        hmap, whd, offset = model(image)
                        # embed()
                        hmap_loss = focal_loss(hmap, heatmap, config.point_weight)
                        
                        r_loss =  reg_l1_loss(offset, regression, masks)
                        whd_loss = reg_l1_loss(whd, wh_size, masks)
                        loss = hmap_loss + r_loss + whd_loss * 0.1
                        valid_loss.append(loss.item())
                        if val_step == 1: 
                            npy2nii(image, 'image')
                            npy2nii(hmap, 'pred_hmap')
                            npy2nii(whd, 'pred_whd')
                            npy2nii(offset, 'pred_offset')

                    logger.info('Epoch: %d, valid_Loss: %.4f', step, np.mean(valid_loss))

                if best_loss > np.mean(valid_loss):
                    best_loss = np.mean(valid_loss)
                    os.makedirs(config.save_dir, exist_ok=True)
                    save(step, optimizer, model, config.save_dir, name='best_model')
                    npy2nii(image, 'image_best')
                    npy2nii(hmap, 'pred_hmap_best')
                    npy2nii(whd, 'pred_whd_best')
                    npy2nii(offset, 'pred_offset_best')
                
                
                pbar.set_description(f'valid: {np.mean(valid_loss):.4f}')
                # writer.add_scalar('valid/loss', np.mean(valid_loss), step)
                            
                                
            model.train()
            
            train_batch = next(train_loader)
            # embed()
            # image = train_batch['input'].unsqueeze(1).cuda()
            # heatmap = train_batch['hmap'].unsqueeze(0).cuda()
            # wh_size = train_batch['whd'].cuda()
            # regression = train_batch['offset'].cuda()
            # masks = train_batch['mask'].cuda()
            image = train_batch['input'].unsqueeze(1)
            heatmap = train_batch['hmap'].unsqueeze(0)
            wh_size = train_batch['whd']
            regression = train_batch['offset']
            masks = train_batch['mask']
            # embed()
            hmap, whd, offset = model(image)

            hmap_loss = focal_loss(hmap, heatmap, config.point_weight)
            # embed()
            r_loss =  reg_l1_loss(offset, regression, masks)
            whd_loss = reg_l1_loss(whd, wh_size, masks)
            loss = hmap_loss + r_loss + whd_loss * 0.1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())
            logger.info('Epoch: %d, Loss: %.4f', step, loss.item())
            
            pbar.set_description(f'train: {loss.item():.4f}')
            # writer.add_scalar('train/loss', loss.item(), step)
            pbar.update(1)
            
            lr_scheduler.step()            

    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Bbox Detection")
    parser.add_argument('--log_dir', default='./log')
    parser.add_argument('--seed', default=1)
    parser.add_argument('--train_batch_size', default=1)
    parser.add_argument('--valid_batch_size', default=1)
    parser.add_argument('--model_type', default=50)
    parser.add_argument('--backbone_name', default='resnet50')
    parser.add_argument('--num_classes', default=1)
    parser.add_argument('--pretrained_model', default='/public_bme/data/meilzj/Tooth/output/panoramic_xray/detects/model-52500.pt')
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--scheduler_steps', default=10000)
    parser.add_argument('--gamma', default=0.1)
    parser.add_argument('--total_steps', default=1000)
    parser.add_argument('--valid_steps', default=1000)
    parser.add_argument('--save_freq', default=20)
    parser.add_argument('--save_dir', default='D:\Work_file\det\save')
    parser.add_argument('--point_weight', default=4)
    args = parser.parse_args()
    
    train(args)
                
        
    