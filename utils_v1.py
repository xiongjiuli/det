import os
from IPython import embed
import torchio as tio
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import csv
# from preprocess import get_mhd_files
import pyvista as pv
import torch
import random
import torch.nn.functional as F 
import torch.nn as nn
from time import time

def npy2nii(name, image_npy, root_dir='D:\Work_file\det_LUNA16_data', suffix='', resample=None, affine=''):
    # csv_dir = 'D:\\Work_file\\det_LUNA16_data\\annotations_pathcoord.csv'
    csv_dir = os.path.join(root_dir, 'annotations_pathcoord_noras.csv')
    df = pd.read_csv(csv_dir)
    df = df[df['seriesuid'] == name]
    # embed()
    mhd_path = str(df[['path']].values[0])[2:-2]
    image = tio.ScalarImage(mhd_path)
    if resample != None:
        if affine == '':
            print("affine isn't be given")
    else:
        affine = image.affine
    # embed()
    if isinstance(image_npy, np.ndarray):
        image_npy = torch.from_numpy(image_npy)
    if len(image_npy.shape) == 3:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy.unsqueeze(0), affine=affine)
        image_nii.save('D:\Work_file\det\\nii_data_resample_seg_crop\{}_{}.nii.gz'.format(name, suffix))
    elif len(image_npy.shape) == 4:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy, affine=affine)
        image_nii.save('D:\Work_file\det\\nii_data_resample_seg_crop\{}_{}.nii.gz'.format(name, suffix))
    else: 
        print('DIM ERROR : npy.dim != 3 or 4')

def read_names_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        names = [row[0] for row in reader]
    return list(set(names))


def name2path(name, root_dir='D:\Work_file\det_LUNA16_data'):
    # * the resampled image 'D:\\Work_file\\det_LUNA16_data\\annotations_pathcoord.csv'
    csv_dir = os.path.join(root_dir, 'annotations_pathcoord_noras.csv')
    df = pd.read_csv(csv_dir)
    df = df[df['seriesuid'] == name]
    mhd_path = str(df[['path']].values[0])[2:-2]
    return mhd_path


def name2coord(mhd_name, root_dir='D:\Work_file\det_LUNA16_data'):
    # * 输入name，输出这个name所对应着的gt坐标信息
    result = []
    # csv_file_dir = 'D:\\Work_file\\det_LUNA16_data\\annotations_pathcoord.csv'
    csv_file_dir = os.path.join(root_dir, 'annotations_pathcoord_noras.csv')
    with open(csv_file_dir, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # embed()
            if row[0] == mhd_name:
                x = float(row[2])
                y = float(row[3])
                z = float(row[4])
                radius = float(row[5])
                result.append((x, y, z, radius))
    return result


def resample_image_coord(name, new_spacing, forsee=None, root=''):

    path = name2path(name)
    coords = name2coord(name)[0]
    coord = coords[0:3]
    # embed()
    w = coords[3] 
    h = coords[3] 
    d = coords[3] 
    image = tio.ScalarImage(path)
    resample = tio.Resample(new_spacing)
    resampled_image = resample(image)
    new_coord = np.array(image.spacing) * np.array(coord) / np.array(new_spacing)
    new_whd = np.array(image.spacing) * np.array((w,h,d)) / np.array(new_spacing)
    # embed()



    # if forsee == True:
    #     if root == '':
    #         print('ROOT ERROE: root not be given')
    #     save_dir = os.path.join(root, 'nii_temp', '{}_resampled.nii.gz'.format(name))
    #     resampled_image.save(save_dir)
    return resampled_image, new_coord, new_whd, resampled_image.affine


import SimpleITK as sitk
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label, regionprops
from skimage.filters import roberts, sobel
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
 
def get_segmented_lungs(im, process_img, plot=False):
    # plt.imshow(im, cmap=plt.cm.bone)
    # plt.show()
    # plt.imshow(process_img, cmap=plt.cm.bone)
    # plt.show()
    
    '''
 	该功能从给定的2D切片分割肺部。
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: 转换成二进制图像。
    '''
    binary = im < -600
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: 移除连接到图像边框的斑点。
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: 给图片贴上标签。
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4:保持标签上有两个最大的区域。
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: 使用半径为2的圆盘进行侵蚀操作。这个手术是分离附在血管上的肺结节。
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: 使用半径为10的圆盘进行闭合操作。这个手术是为了让结节附着在肺壁上。
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: 填充肺部二元面罩内的小孔。
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: 在输入图像上叠加二值遮罩。
    '''
    get_high_vals = binary == 0
    # img = window_image(im, -600, 1100)
    # embed()
    process_img[get_high_vals] = 0
    # embed()
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(process_img, cmap=plt.cm.bone)
    plt.show()
    # embed()
    return process_img

def window_image(image, window_center, window_width):
    image = image.astype(float)
    min_value = window_center - (window_width / 2)
    max_value = window_center + (window_width / 2)
    image_x = np.clip(image, min_value, max_value)
    # embed()
    image_x = (image_x - min_value) / (max_value - min_value) 
    # plt.imshow(image[100,:,:])
    # plt.show()
    # embed()
    return image_x

def seg_3d_image(image_nii, forsee=None):

    data = np.array(image_nii.data[0, :, :, :])

    process_data = window_image(data, -600., 1100.)
    result = np.zeros_like(data).astype(float)

    for i in range(image_nii.shape[3]):
        im = get_segmented_lungs(data[:,:,i], process_data[:,:,i])
        result[:,:,i] = im
    if forsee != None:
        data_1 = torch.from_numpy(result)
        data_nii = tio.ScalarImage(tensor=data_1.unsqueeze(0), affine=np.eye(4))
        data_nii.save('./nii_temp/seg-image.nii')
    # if forsee != None:
    #     npy2nii(name, data, suffix='seg', resample=True, affine=affine)
    return result

def get_lung_coordinates(data):
    # 获取肺部区域的坐标
    lung_coords = np.where(data != 0)
    lung_coord_range = [(np.min(coord), np.max(coord)) for coord in lung_coords]
    
    return lung_coord_range


def crop_lung(data):
    # 获取肺部区域的坐标
    data = np.asarray(data)
    lung_coords = np.where(data != 0)
    
    # 计算肺部区域的坐标范围
    lung_coord_range = [np.min(coord) for coord in lung_coords] + [np.max(coord) + 1 for coord in lung_coords]
    
    # 裁剪出肺部区域
    cropped_lung = np.copy(data[lung_coord_range[0]:lung_coord_range[3], lung_coord_range[1]:lung_coord_range[4], lung_coord_range[2]:lung_coord_range[5]])
    
    # 输出裁剪后的肺部区域
    return cropped_lung

def write_to_csv(name, path, coordinates, dimensions, x_scale, y_scale, z_scale):
    x, y, z = coordinates
    w, h, d = dimensions
    x_start, x_end = x_scale
    y_start, y_end = y_scale
    z_start, z_end = z_scale
    new_csv_dir = 'D:\Work_file\det_LUNA16_data\AT_afterlungcrop.csv'
    with open(new_csv_dir, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, path, x, y, z, w, h, d, x_start, x_end, y_start, y_end, z_start, z_end])



def data_propre():
    csv_dir = 'D:\Work_file\det_LUNA16_data\\annotations_pathcoord_noras.csv'
    # csv_dir = 'D:\Work_file\det_LUNA16_data\\annotations_pathcoord_noras.csv'
    name_list = read_names_from_csv(csv_dir)
    # shape_x = []
    # shape_y = []
    # shape_z = []
    for name in tqdm(name_list):
        new_spacing = (0.7, 0.7, 1.2)
        resampled_image, new_coord, new_whd, affine = resample_image_coord(name, new_spacing, forsee=None, root='')


        data = seg_3d_image(resampled_image)
        lung_coord_range = get_lung_coordinates(data)
        cropped_lung = crop_lung(data)

        # lung_shape = cropped_lung.shape
        # x = lung_shape[0]
        # y = lung_shape[1]
        # z = lung_shape[2]
        
        # with open('./txt/xshape.txt', mode='a') as f:
        #     f.write(str(x))
        #     f.write('\n')
        # with open('./txt/yshape.txt', mode='a') as f:
        #     f.write(str(y))
        #     f.write('\n')
        # with open('./txt/zshape.txt', mode='a') as f:
        #     f.write(str(z))
        #     f.write('\n')

        # the finish coord
        crop_coord = np.array((new_coord[0] - lung_coord_range[0][0], new_coord[1] - lung_coord_range[1][0], new_coord[2] - lung_coord_range[2][0]))
        path = 'D:\Work_file\det\\nii_data_resample_seg_crop\{}_croplung.nii.gz'.format(name)
        npy2nii(name, cropped_lung, suffix='croplung', resample=True, affine=affine)
        # write_to_csv(name, path, crop_coord, new_whd, lung_coord_range[0], lung_coord_range[1], lung_coord_range[2])


def plot_loss(log_file_path, log_file_path_5101, save_frequency, contrast=None):
    train_loss = []
    valid_loss = []
    whd_Loss = []
    r_Loss = []
    hmap_Loss = []

    train_loss_5101 = []
    valid_loss_5101 = []
    whd_Loss_5101 = []
    r_Loss_5101 = []
    hmap_Loss_5101 = []
    with open(log_file_path, 'r') as f:
        for line in f:
            if 'train_Loss' in line:
                train_loss.append(float(line.split('train_Loss: ')[1]))
            elif 'valid_Loss' in line :
                valid_loss.append(float(line.split('valid_Loss: ')[1]))
            elif 'whd_Loss' in line and 'valid' not in line:
                whd_Loss.append(float(line.split('whd_Loss: ')[1]))
            elif 'r_Loss' in line and 'valid' not in line:
                r_Loss.append(float(line.split('r_Loss: ')[1]))
            elif 'hmap_Loss' in line and 'valid' not in line:
                hmap_Loss.append(float(line.split('hmap_Loss: ')[1]))

    if contrast != None:
        with open(log_file_path_5101, 'r') as f_5101:
            for line in f_5101:
                if 'train_Loss' in line:
                    train_loss_5101.append(float(line.split('train_Loss: ')[1]))
                elif 'valid_Loss' in line :
                    valid_loss_5101.append(float(line.split('valid_Loss: ')[1]))
                elif 'whd_Loss' in line and 'valid' not in line:
                    whd_Loss_5101.append(float(line.split('whd_Loss: ')[1]))
                elif 'r_Loss' in line and 'valid' not in line:
                    r_Loss_5101.append(float(line.split('r_Loss: ')[1]))
                elif 'hmap_Loss' in line and 'valid' not in line:
                    hmap_Loss_5101.append(float(line.split('hmap_Loss: ')[1]) / 5.)

    # embed()
    plt.plot(train_loss, 'b-')
    if contrast != None:
        plt.plot(train_loss_5101, 'r--')
    plt.title('Train Loss')
    plt.show()
    
    plt.plot(valid_loss, 'b-')
    if contrast != None:
        plt.plot(valid_loss_5101, 'r--')
    plt.title('Valid Loss')
    plt.show()

    plt.plot(hmap_Loss, 'b-')
    if contrast != None:
        plt.plot(hmap_Loss_5101, 'r--')
    plt.title('hmap_Loss')
    plt.show()

    plt.plot(whd_Loss, 'b-')
    if contrast != None:
        plt.plot(whd_Loss_5101, 'r--')
    plt.title('whd_Loss')
    plt.show()

    plt.plot(r_Loss, 'b-')
    if contrast != None:
        plt.plot(r_Loss_5101, 'r--')
    plt.title('r_Loss')
    plt.show()
    
    plt.plot(train_loss, 'g-', label='Train Loss')
    plt.plot([i * save_frequency for i in range(1, len(valid_loss) + 1)], valid_loss, 'k-',label='Valid Loss')
    if contrast != None:
        plt.plot(train_loss_5101,  'b--',label='Train Loss 5-1-01')
        plt.plot([i * save_frequency for i in range(1, len(valid_loss_5101) + 1)], valid_loss_5101, 'r--', label='Valid Loss 5-1-01')
    plt.legend()
    plt.title('Train and Valid Loss')
    plt.show()


if __name__ == '__main__':
    # csv_dir = 'D:\Work_file\det_LUNA16_data\\annotations_pathcoord_noras.csv'
    # name_list = read_names_from_csv(csv_dir)
    # data_propre()
    # embed()
    plot_loss('training.log', 'training_v2.log', 10, contrast=True)
    # csv_dir = 'D:\Work_file\det_LUNA16_data\\annotations_pathcoord_noras.csv'
    # # csv_dir = 'D:\Work_file\det_LUNA16_data\\annotations_pathcoord_noras.csv'
    # name_list = read_names_from_csv(csv_dir)
    # for name in tqdm(name_list):
    #     # print(name)
    #     path = name2path(name)

