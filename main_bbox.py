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
# from utils import focal_loss, reg_l1_loss
from tqdm import tqdm 
import numpy as np
import torch.nn.functional as F 
from IPython import embed
from data.data_v1 import luna16Dataset
import logging
from time import time
# from model.swin_unet_v1 import SwinTransformerSys3D
from model.swinunet3d_v1 import swinUnet_p_3D
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
# from utils import npy2nii

def focal_loss_v0(preds, targets, weight=0):#！！！！！！
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        preds (B x c x h x w x d)
        gt_regr (B x c x h x w x d)
    '''
    preds = preds.permute(0, 2, 3, 4, 1)
    targets = targets.permute(0, 2, 3, 4, 1)

    pos_inds = targets.ge(1).float()
    neg_inds = targets.lt(1).float()
    # bg_sum = targets.eq(0).sum()
    # fg_sum = targets.lt(1).sum() - bg_sum

    neg_weights = torch.pow(1 - targets, weight)
    # pos_weights = torch.pow(targets * 10., 3) # *这个后面的参数也是一个超参数
    # neg_weights = torch.where(targets == 0, torch.tensor((fg_sum * 3) / (bg_sum + fg_sum), device=preds.device), torch.tensor(bg_sum / (bg_sum + fg_sum), device=preds.device))
    # neg_weights = torch.where(targets == 0, torch.ones_like(targets)*(fg_sum * 3) / (bg_sum + fg_sum), torch.ones_like(targets)*(bg_sum / (bg_sum + fg_sum)))

    #! /10的和不除以10的
    loss = 0
    # for pred in preds:
        # pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    preds = torch.clamp(preds, min=1e-4, max=1 - 1e-4)
    # print(f'in the loss shape is {preds.shape}')
    # print(f'the target shape is {pos_inds.shape}')
    pos_loss = torch.log(preds) * torch.pow(targets - preds, 2) * pos_inds  #!原来第二个参数是2，现在改成了1，并且后面还乘了100.
    neg_loss = torch.log(1 - preds) * torch.pow(preds, 2) * neg_weights * neg_inds

    # obj_inds = targets.eq(1).float()
    # num_pos = obj_inds.float().sum()
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss / len(preds)

def focal_loss(preds, targets, weight=0):#！！！！！！
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        preds (B x c x h x w x d)
        gt_regr (B x c x h x w x d)
    '''
    preds = preds.permute(0, 2, 3, 4, 1)
    targets = targets.permute(0, 2, 3, 4, 1)

    pos_inds = targets.ge(0.5).float()
    neg_inds = targets.lt(0.5).float()
    # bg_sum = targets.eq(0).sum()
    # fg_sum = targets.lt(1).sum() - bg_sum

    neg_weights = torch.pow(1 - targets, weight)
    pos_weights = torch.pow(targets * 10., 3) # *这个后面的参数也是一个超参数
    # neg_weights = torch.where(targets == 0, torch.tensor((fg_sum * 3) / (bg_sum + fg_sum), device=preds.device), torch.tensor(bg_sum / (bg_sum + fg_sum), device=preds.device))
    # neg_weights = torch.where(targets == 0, torch.ones_like(targets)*(fg_sum * 3) / (bg_sum + fg_sum), torch.ones_like(targets)*(bg_sum / (bg_sum + fg_sum)))

    #! /10的和不除以10的
    loss = 0
    # for pred in preds:
        # pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    preds = torch.clamp(preds, min=1e-4, max=1 - 1e-4)
    # print(f'in the loss shape is {preds.shape}')
    # print(f'the target shape is {pos_inds.shape}')
    pos_loss = torch.log(preds) * torch.pow(targets - preds, 2) * pos_inds * pos_weights #!原来第二个参数是2，现在改成了1，并且后面还乘了100.
    neg_loss = torch.log(1 - preds) * torch.pow(preds, 2) * neg_weights * neg_inds

    obj_inds = targets.eq(1).float()
    num_pos = obj_inds.float().sum()
    # num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss / len(preds)


def reg_l1_loss(pred, target, mask):
    #--------------------------------#
    #   计算l1_loss
    #--------------------------------#
    pred = pred.permute(0,2,3,4,1)
    target = target.permute(0,2,3,4,1)
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 1, 3)
    
    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


def creat_logging(log_name):
# 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建一个handler，用于将日志输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 创建一个handler，用于将日志写入到文件中
    fh = logging.FileHandler(log_name)
    fh.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # 将handler添加到logger中
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


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
    image_npy = image_npy.cpu()
    affine = np.array([[0.7, 0, 0, 0], [0, 0.7, 0, 0], [0, 0, 1.2, 0], [0, 0, 0, 1]])
    if isinstance(image_npy, np.ndarray):
        image_npy = torch.from_numpy(image_npy)
    if len(image_npy.shape) == 3:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy.unsqueeze(0), affine=affine)
        image_nii.save('./nii_temp/{}.nii'.format(suffix))
    elif len(image_npy.shape) == 4:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy, affine=affine)
        image_nii.save('./nii_temp/{}.nii'.format(suffix))
    elif len(image_npy.shape) == 5:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy[0, :, :, :,:], affine=affine)
        image_nii.save('./nii_temp/{}.nii'.format(suffix))
    else: 
        print('DIM ERROR : npy.dim != 3 or 4 or 5')


def resize_and_normalize(tensor, new_size):

    # tensor = torch.tensor(input_array)
    
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif len(tensor.shape) == 4:
        tensor = tensor.unsqueeze(0)
    else:
        print('the numpy dim != 3 or 4')
    
    resized_tensor = F.interpolate(tensor, size=new_size, mode='trilinear', align_corners=True)
    resized_tensor = resized_tensor
    # normalized_tensor = (resized_tensor - resized_tensor.min()) / (resized_tensor.max() - resized_tensor.min() + 1e-8)
    return resized_tensor


def train(config):
    
    # augmentation if needed
    train_dataset = luna16Dataset(mode='train', data_process='crop', crop_size=config.crop_size)
    valid_dataset = luna16Dataset(mode='valid', data_process='crop', crop_size=config.crop_size)
    
    # train_loader = cycle(data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=4, drop_last=True))
    # valid_loader = cycle(data.DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=False, num_workers=4))
    train_loader = data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=2, drop_last=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=False, num_workers=2)
    
    logger = creat_logging(config.log_name)
    model = CenterNet(config.backbone_name, config.num_classes)
    # model = SwinTransformerSys3D(num_classes=64)
    # model = Hourglass()
    # model = get_hourglass['large_hourglass']

    # #* the swinunet3d config
    # x = torch.randn((1, 1, 160, 160, 160))
    # window_size = [i // 32 for i in x.shape[2:]]
    # model = swinUnet_p_3D(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24),
    #                 window_size=window_size, in_channel=1, num_classes=64
    #                 )
    
    time_cuda = time()
    model.cuda()
    print(time() - time_cuda)
    optimizer = torch.optim.Adam(model.parameters(), config.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.scheduler_steps, gamma=config.gamma)
    start_epoch = 0

    if config.pretrained_model: 
        model_path = config.pretrained_model
        model.load_state_dict(torch.load(model_path)['model'])
        optimizer.load_state_dict(torch.load(model_path)['opt'])  # 加载优化器参数
        start_epoch = torch.load(model_path)['step']  # 设置开始的epoch

    
    best_loss = 1e4

    with tqdm(total=(config.total_steps - start_epoch)) as pbar:
        for step in range(start_epoch + 1 , config.total_steps):

            if (step) % 10 == 0:
            # if (step) % 100 == 0:
                model.eval()
                os.makedirs(config.save_dir, exist_ok=True)
                save(step, optimizer, model, config.save_dir, name=config.save_model_name)

                # multi-scale test could do to improve performance
                valid_loss = []
                valid_hmap_loss = 0
                valid_r_loss = 0
                valid_whd_loss = 0
                with torch.no_grad():
                    # for val_step in range(config.valid_steps):
                    # for val_step in range(100):
                    val_step = 1
                    for valid_batch in valid_loader:
                        # valid_batch = next(valid_loader)
                        image = valid_batch['input'].cuda().unsqueeze(1)
                        heatmap = valid_batch['hmap'].cuda().unsqueeze(1)
                        wh_size = valid_batch['whd'].cuda()
                        regression = valid_batch['offset'].cuda()
                        masks = valid_batch['mask'].cuda()
                        hmap, whd, offset = model(image)
                        # print('the valid shape')
                        # print(f'the train the shape of the image is {image.shape}')
                        # print(f'the train the shape of the heatmap is {heatmap.shape}')
                        # print(f'the train the shape of the wh_size is {wh_size.shape}')
                        # print(f'the train the shape of the regression is {regression.shape}')
                        # print(f'the train the shape of the masks is {masks.shape}')
                        hmap_loss = focal_loss_v0(hmap, heatmap, config.point_weight)
                        
                        r_loss =  reg_l1_loss(offset, regression, masks)
                        whd_loss = reg_l1_loss(whd, wh_size, masks)
                        loss = hmap_loss + r_loss + whd_loss * 0.1
                        valid_loss.append(loss.item())
                        valid_hmap_loss += hmap_loss
                        valid_r_loss += r_loss
                        valid_whd_loss += whd_loss
                        if val_step == 1: 
                            image = resize_and_normalize(image, new_size=config.crop_size)
                            npy2nii(image, f'image-{config.save_model_name}')
                            npy2nii(hmap, f'pred_hmap-{config.save_model_name}')
                            npy2nii(whd, f'pred_whd-{config.save_model_name}')
                            npy2nii(offset, f'pred_offset-{config.save_model_name}')
                        val_step += 1

                    logger.info('Epoch: %d, valid_Loss: %.4f', step, np.mean(valid_loss))
                    logger.info('Epoch: %d, valid_hmap_Loss: %.4f', step, valid_hmap_loss)
                    logger.info('Epoch: %d, valid_r_Loss: %.4f', step, valid_r_loss)
                    logger.info('Epoch: %d, valid_whd_Loss: %.4f', step, valid_whd_loss)

                if best_loss > np.mean(valid_loss):
                    best_loss = np.mean(valid_loss)
                    os.makedirs(config.save_dir, exist_ok=True)
                    save(step, optimizer, model, config.save_dir, name=config.best_model_name)
                    image_save = resize_and_normalize(image, new_size=config.crop_size)
                    npy2nii(image_save, f'image_{config.best_model_name}')
                    npy2nii(hmap, f'pred_hmap_{config.best_model_name}')
                    npy2nii(whd, f'pred_whd_{config.best_model_name}')
                    npy2nii(offset, f'pred_offset_{config.best_model_name}')
                
                
                pbar.set_description(f'valid: {np.mean(valid_loss):.4f}')
                # writer.add_scalar('valid/loss', np.mean(valid_loss), step)
                            
                                
            model.train()
            batch_step = 0
            train_loss = []
            train_loss_save = []
            hmap_Loss = 0
            r_Loss = 0
            whd_Loss = 0
            # train_batch = next(train_loader)
            for train_batch in tqdm(train_loader):

                # embed()
                image = train_batch['input'].cuda().unsqueeze(1)
                heatmap = train_batch['hmap'].cuda().unsqueeze(1)
                wh_size = train_batch['whd'].cuda()
                regression = train_batch['offset'].cuda()
                masks = train_batch['mask'].cuda()
                name = train_batch['name']
                # embed()
                # image_savetrain = resize_and_normalize(image, new_size=(256, 256, 256))
                # npy2nii(image_savetrain, f'input_image_forsee_{name[-6:]}')
                # npy2nii(heatmap, f'input_hmap_forseee_{name[-6:]}')
                # npy2nii(wh_size, f'input_whd_forseee_{name[-6:]}')
                # npy2nii(regression, f'input_offset_forseee_{name[-6:]}')
                # npy2nii(masks, f'input_mask_forseee_{name[-6:]}')
                # # embed()
                # print('the train shape')
                # print(f'the train the shape of the image is {image.shape}')
                # print(f'the train the shape of the heatmap is {heatmap.shape}')
                # print(f'the train the shape of the wh_size is {wh_size.shape}')
                # print(f'the train the shape of the regression is {regression.shape}')
                # print(f'the train the shape of the masks is {masks.shape}')

                hmap, whd, offset = model(image)
                # print('after the model============================')
                # print(f'the pred hmap shape is {hmap.shape}')
                # print(f'the whd shape is {whd.shape}')
                # print(f'the offset shape is {offset.shape}')
                hmap_loss = focal_loss_v0(hmap, heatmap, config.point_weight)
                
                r_loss =  reg_l1_loss(offset, regression, masks)
                whd_loss = reg_l1_loss(whd, wh_size, masks)
                # print(f'the hmap loss is {hmap_loss}')
                # print(f'the r_loss is {r_loss}')
                # print(f'the whd_loss loss is {whd_loss}')
                loss = 10. * hmap_loss + r_loss + whd_loss * 0.1
                loss_save = hmap_loss + r_loss + whd_loss * 0.1
                train_loss_save.append(loss_save.item())
                train_loss.append(loss.item())
                hmap_Loss += hmap_loss
                r_Loss += r_loss
                whd_Loss += whd_loss                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_step += 1
            # print(loss.item())


            
            lr_scheduler.step()
            loss /= len(train_loader)
            logger.info('Epoch: %d, train_Loss: %.4f', step, np.mean(train_loss_save))
            logger.info('Epoch: %d, hmap_Loss: %.4f', step, hmap_loss)
            logger.info('Epoch: %d, r_Loss: %.4f', step, r_loss)
            logger.info('Epoch: %d, whd_Loss: %.4f', step, whd_loss)
            # logger.info('average_train_Loss: %.4f', np.mean(train_loss))    
            pbar.set_description(f'train: {loss.item():.4f}')
            # writer.add_scalar('train/loss', loss.item(), step)
            pbar.update(1)
            
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Bbox Detection")
    parser.add_argument('--log_dir', default='./log')
    parser.add_argument('--seed', default=1)
    parser.add_argument('--train_batch_size', default=1)
    parser.add_argument('--valid_batch_size', default=1)
    parser.add_argument('--backbone_name', default='resnet101')
    parser.add_argument('--num_classes', default=1)
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--scheduler_steps', default=10000)
    parser.add_argument('--gamma', default=0.1)
    parser.add_argument('--total_steps', default=1000)
    parser.add_argument('--valid_steps', default=1000)
    parser.add_argument('--save_freq', default=20)
    parser.add_argument('--save_dir', default='/public_bme/data/xiongjl/det/save')
    parser.add_argument('--point_weight', default=1)
    parser.add_argument('--pretrained_model', default="/public_bme/data/xiongjl/det/save/0829_v4_res101_crop256_10-1-01_hmapv6-20.pt")
    parser.add_argument('--crop_size', default=(256, 256, 256))
    # parser.add_argument('--crop_size', default=(160, 160, 160))
    parser.add_argument('--log_name', default='./log/training_0829_v4_res101_crop256_10-1-01_hmapv6.log')
    parser.add_argument('--best_model_name', default='0829_v4_res101_crop256_10-1-01_hmapv6_best')
    parser.add_argument('--save_model_name', default='0829_v4_res101_crop256_10-1-01_hmapv6')
    args = parser.parse_args()
    
    train(args)

        
    