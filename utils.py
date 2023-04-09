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
from time import time


def get_mhd_files(path):
    mhd_files = []
    for i in range(10):
        subset = 'subset' + str(i)
        subset_path = os.path.join(path, subset)
        if os.path.exists(subset_path):
            for file in os.listdir(subset_path):
                if file.endswith('.mhd'):
                    filename = os.path.splitext(file)[0]
                    mhd_files.append({'name': filename, 'path': os.path.join(subset_path, file)})
    return mhd_files

def find_the_orientation(mhd_files):
    for i in range(len(mhd_files)):
        name = mhd_files[i]['name']
        path = mhd_files[i]['path']
        image = tio.ScalarImage(path)
        if image.orientation == ('R', 'A', 'S'):
            print(name)
        else:
            pass
    return print('over')




def _map_data(data, max, min):
    '''
    data: the data to map
    max : map to the max
    min : map to the min
    '''
    data_max = data.max()
    data_min = data.min()

    data = min + (max - min) / (data_max - data_min) * (data - data_min)
    return data

def generate_heatmap(name, save=None, save_nii=None):
    # 从csv文件中读取数据
    csv_path = 'D:\\Work_file\\det_LUNA16_data\\annotations_pathcoord.csv'
    data = pd.read_csv(csv_path)
    # 筛选出与name对应的数据
    data = data[data['seriesuid'] == name]
    # 获取mhd文件的路径
    mhd_path = data['path'].iloc[0]
    # 读取mhd文件
    image = tio.ScalarImage(mhd_path)
    # 获取图像大小
    shape = image.shape[1:]
    # 创建一个与图像大小相同的数组
    heatmap = np.zeros(shape)
    # 遍历每个点
    for index, row in data.iterrows():
        x, y, z = row['coordX'], row['coordY'], row['coordZ']
        diameter = row['diameter_mm']
        # 将坐标转换为整数
        x, y, z = int(x), int(y), int(z)
        # embed()
        # 计算高斯衰减的范围
        radius = int(diameter / 2)
        # 在heatmap上进行高斯衰减
        anchor = np.zeros((2 * radius + 1, 2 * radius + 1, 2 * radius + 1), dtype=np.float32)
        anchor[int(radius)][int(radius)][int(radius)] = 1
        anchor_hm = gaussian_filter(anchor, sigma=(2, 2, 2))
        anchor_hm = _map_data(anchor_hm, max=1, min=1e-4)
        heatmap[int(x) - radius - 1 : int(x) + radius, 
                int(y) - radius - 1 : int(y) + radius, 
                int(z) - radius - 1 : int(z) + radius] += anchor_hm[:, :, :]
    
    if save != None:
        np.save('D:\\Work_file\\det\\npy_data\\{}_hmap.npy'.format(name), heatmap)
        # np.save('D:\\Work_file\\det\\npy_data\\{}_image.npy'.format(name), image.data[0, :, :, :])
    if save_nii != None:
        # Get the affine from the WHD file
        affine = image.affine
        heatmap = torch.from_numpy(heatmap).unsqueeze(0)
        # Create a torchio Image from the heatmap and affine
        heatmap_image = tio.ScalarImage(tensor=heatmap, affine=affine)
        # Save the heatmap as a NIfTI file
        heatmap_image.save('./nii_temp/heatmap.nii.gz')

    return heatmap


def find_name_in_csv(mhd_name):
    # * 输入name，输出这个name所对应着的gt坐标信息
    result = []
    csv_file_dir = 'D:\\Work_file\\det_LUNA16_data\\annotations_pathcoord.csv'
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


def read_names_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        names = [row[0] for row in reader]
    return list(set(names))




# * to get the W H D map very preprocess data 
def get_WHD_offset_mask(name, whd=None, offset=None, mask=None):

    # * the resampled image
    csv_dir = 'D:\\Work_file\\det_LUNA16_data\\annotations_pathcoord.csv'
    df = pd.read_csv(csv_dir)
    df = df[df['seriesuid'] == name]

    coords = df[['coordX','coordY','coordZ']].values
    diameter_mm = df[['diameter_mm']].values
    # embed()
    mhd_path = str(df[['path']].values[0])[2:-2]

    data_npy_dir = 'D:\\Work_file\\det\\npy_data\\{}_image.npy'.format(name)
    if os.path.isfile(data_npy_dir):
        data_npy = np.load(data_npy_dir)
    else:
        image = tio.ScalarImage(mhd_path)
        data_npy = image.data[0, :, :, :]

    w_image = np.zeros(data_npy.shape)
    h_image = np.zeros(data_npy.shape)
    d_image = np.zeros(data_npy.shape)
    whd_image = []
    offset_image_w = np.zeros(data_npy.shape)
    offset_image_h = np.zeros(data_npy.shape)
    offset_image_d = np.zeros(data_npy.shape)
    offset_image = []
    mask_image = np.zeros(data_npy.shape)


    if whd == True:
        for i in range(len(coords)):
            coord_int = coords[i].astype(np.int32)
            # embed()
            # w_image[coord_int] = coords[i]
            w_image[(coord_int[0]-1, coord_int[1]-1, coord_int[2]-1)] = diameter_mm[i] / 2
            h_image[(coord_int[0]-1, coord_int[1]-1, coord_int[2]-1)] = diameter_mm[i] / 2
            d_image[(coord_int[0]-1, coord_int[1]-1, coord_int[2]-1)] = diameter_mm[i] / 2

        whd_image = np.stack([w_image, h_image, d_image])
        save_whd_dir = 'D:\\Work_file\\det\\npy_data\\{}_whd.npy'.format(name)
        # np.save(save_whd_dir, whd_image)

    if offset == True:
        for i in range(len(coords)):
            coord_int = coords[i].astype(np.int32)
            offset_image_w[(coord_int[0]-1, coord_int[1]-1, coord_int[2]-1)] = coords[i][0] - coord_int[0]
            offset_image_h[(coord_int[0]-1, coord_int[1]-1, coord_int[2]-1)] = coords[i][1] - coord_int[1]
            offset_image_d[(coord_int[0]-1, coord_int[1]-1, coord_int[2]-1)] = coords[i][2] - coord_int[2]

        offset_image = np.stack([offset_image_w, offset_image_h, offset_image_d])

        save_offset_dir = 'D:\\Work_file\\det\\npy_data\\{}_offset.npy'.format(name)
        # np.save(save_offset_dir, offset_image)

    if mask == True:
        for i in range(len(coords)):
            coord_int = coords[i].astype(np.int32)
            mask_image[(coord_int[0]-1, coord_int[1]-1, coord_int[2]-1)] = 1

        save_mask_dir = 'D:\\Work_file\\det\\npy_data\\{}_mask.npy'.format(name)
        # np.save(save_mask_dir, mask_image)
    
    info_dict = {}
    info_dict['whd'] = whd_image
    info_dict['offset'] = offset_image
    info_dict['mask'] = mask_image

    return info_dict


def crop_data(name):

    image_dir = 'D:\\Work_file\\det\\npy_data\\{}_image.npy'.format(name)
    if os.path.isfile(image_dir):
        image_data = np.load(image_dir)
    else:
        csv_path = 'D:\\Work_file\\det_LUNA16_data\\annotations_pathcoord.csv'
        data = pd.read_csv(csv_path)
        # embed()
        data = data[data['seriesuid'] == name]
        mhd_path = data['path'].iloc[0]
        # embed()
        image = tio.ScalarImage(mhd_path)
        image_data = image.data[0, :, :, :]

    hmap_dir = 'D:\\Work_file\\det\\npy_data\\{}_hmap.npy'.format(name)
    if os.path.isfile(hmap_dir):
        hmap_data = np.load(hmap_dir)
    else:
        time_hmap = time()
        hmap_data = generate_heatmap(name=name, save=True)
        print('hmap_data_generate : {}'.format(time() - time_hmap))

    whd_dir = 'D:\\Work_file\\det\\npy_data\\{}_whd.npy'.format(name)
    if os.path.isfile(whd_dir):
        whd_data = np.load(whd_dir)
    else:
        time_whd = time()
        whd_data = get_WHD_offset_mask(name, whd=True)['whd'] 
        print('whd_data_generate : {}'.format(time() - time_whd))

    offset_dir = 'D:\\Work_file\\det\\npy_data\\{}_offset.npy'.format(name)
    if os.path.isfile(offset_dir):
        offset_data = np.load(offset_dir)
    else:
        time_offset = time()
        offset_data = get_WHD_offset_mask(name, offset=True)['offset']
        print('offset_data_generate : {}'.format(time() - time_offset))

    mask_dir = 'D:\\Work_file\\det\\npy_data\\{}_mask.npy'.format(name)
    if os.path.isfile(mask_dir):
        mask_data = np.load(mask_dir)
    else:
        time_mask = time()
        mask_data = get_WHD_offset_mask(name, mask=True)['mask']
        print('mask_data_generate : {}'.format(time() - time_mask))

    coords = find_name_in_csv(name)
    x, y, z = image_data.shape[:]

    if x < 128 or y < 128 or z < 128:
        input_array = np.pad(input_array, ((0, max(0, 128-x)), (0, max(0, 128-y)), (0, max(0, 128-z))), mode='reflect')
    if random.random() < 0.9:
        px, py, pz, _ = random.choice(coords)
        px = int(px)
        py = int(py)
        pz = int(pz)
        if px <= x - 64:
            x_start = random.randint(px - 64, px)
        else:
            x_start = random.randint(x - 192, x - 128)
        if py <= y - 64:
            y_start = random.randint(py - 64, py)
        else:
            y_start = random.randint(y - 192, y - 128)
        if pz <= z - 64:
            z_start = random.randint(pz - 64, pz)
        else:
            z_start = random.randint(z - 192, z - 128)
    else:
        x_start = random.randint(0, min(0, x - 128))
        y_start = random.randint(0, min(0, y - 128))
        z_start = random.randint(0, min(0, z - 128))

    image_crop = image_data[x_start : x_start + 128,
                            y_start : y_start + 128,
                            z_start : z_start + 128]

    hmap_crop = hmap_data[x_start : x_start + 128,
                          y_start : y_start + 128,
                          z_start : z_start + 128]
    
    offset_crop = offset_data[:, x_start : x_start + 128,
                              y_start : y_start + 128,
                              z_start : z_start + 128]
    
    whd_crop = whd_data[:, x_start : x_start + 128,
                        y_start : y_start + 128,
                        z_start : z_start + 128]
    
    mask_crop = mask_data[x_start : x_start + 128,
                          y_start : y_start + 128,
                          z_start : z_start + 128]
    # embed()
    image_crop = image_crop.type(torch.float32)
    hmap_crop = torch.from_numpy(hmap_crop).type(torch.float32)
    whd_crop = torch.from_numpy(whd_crop).type(torch.float32)
    offset_crop = torch.from_numpy(offset_crop).type(torch.float32)
    mask_crop = torch.from_numpy(mask_crop).type(torch.float32)
    # return image_crop, hmap_crop, bbox_crop
    # embed()
    return image_crop, hmap_crop, whd_crop, offset_crop, mask_crop


# -*- coding:utf-8 -*-
'''
此脚本用于数据科学碗中的lung 2017基本流程
'''
import SimpleITK as sitk
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label, regionprops
from skimage.filters import roberts, sobel
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
 
 
# numpyImage[numpyImage > -600] = 1
# numpyImage[numpyImage <= -600] = 0
 
def get_segmented_lungs(im, plot=False):
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
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone)
 
    plt.show()


 
    return im
 
 

def npy2nii(name, image_npy, suffix='', resample=None, affine=''):
    csv_dir = 'D:\\Work_file\\det_LUNA16_data\\annotations_pathcoord.csv'
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
        image_nii.save('./nii_temp/{}_{}.nii'.format(name, suffix))
    elif len(image_npy.shape) == 4:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy, affine=affine)
        image_nii.save('./nii_temp/{}_{}.nii'.format(name, suffix))
    else: 
        print('DIM ERROR : npy.dim != 3 or 4')

    # image_nii.save('./nii_temp/{}_image.nii'.format(name))
    return print('save done')


def name2path(name):
    # * the resampled image
    csv_dir = 'D:\\Work_file\\det_LUNA16_data\\annotations_pathcoord.csv'
    df = pd.read_csv(csv_dir)
    df = df[df['seriesuid'] == name]
    mhd_path = str(df[['path']].values[0])[2:-2]
    return mhd_path



def seg_3d(name, forsee=None):

    mhd_path = name2path(name)
    image_nii = tio.ScalarImage(mhd_path)
    data = np.array(image_nii.data[0, :, :, :])
    # pre_data = 
    for i in range(image_nii.shape[3]):
        im = get_segmented_lungs(data[:,:,i])
        data[:,:,i] = im
    if forsee != None:
        npy2nii(name, data, suffix='seg')
    return data

    

def resize_data(name, new_shape=(514, 514, 277)):
    # time_1 = time()
    path = name2path(name)
    coords = find_name_in_csv(name)
    image = tio.ScalarImage(path)
    scale = np.array(new_shape) / np.array(image.shape[1:])
    new_spacing = (np.array(image.spacing) * np.array(new_shape)) / np.array(image.shape[1:])
    new_affine = image.affine * np.array([[scale[0], 0, 0, 0], [0, scale[1], 0, 0], [0, 0, scale[2], 0], [0, 0, 0, 1]])
    new_coords = []
    new_whd = []
    for coord in coords:
        new_coords.append(coord[: -1] * scale)
        new_whd.append((coord[-1], coord[-1], coord[-1]) * scale)
    # embed()
    
    # create the 1.mask and 2.whd and 3.offset and the 4.image
    mask = create_mask(new_coords, new_shape) # 0.0s no save is so fast
    whd = create_whd(coordinates=new_coords, whd=new_whd, shape=new_shape)
    offset = create_offset(coordinates=new_coords, shape=new_shape)
    # npy2nii(name, mask, suffix='mask', resample=True, affine=new_affine)
    # npy2nii(name, whd, suffix='whd', resample=True, affine=new_affine)
    # npy2nii(name, offset, suffix='offset', resample=True, affine=new_affine)
    
    hmap_dir = 'D:\Work_file\det\\npy_data\\{}_hmap.npy'.format(name)
    if os.path.isfile(hmap_dir):
        hmap = np.load(hmap_dir)
    else:
        hmap = create_hmap(coordinates=new_coords, shape=new_shape, save=True, name=name)
    # npy2nii(name, hmap, suffix='hmap', resample=True, affine=new_affine)
    
    input_data = seg_3d(name)
    input_data = torch.from_numpy(input_data).unsqueeze(0).unsqueeze(0).float()
    input_resize = F.interpolate(input_data, size=new_shape).squeeze(0).squeeze(0).numpy()
    input_resize = (input_resize - input_data.numpy().min()) / (input_data.numpy().max() - input_data.numpy().min() + 1e-8)
    # npy2nii(name, input_resize, suffix='resize_image', resample=True, affine=new_affine)
    # embed()
    dict = {}
    dict['hmap'] = hmap
    dict['offset'] = offset
    dict['mask'] = mask
    dict['input'] = input_resize
    dict['whd'] = whd
    return dict



def create_mask(coordinates, shape, save=False, name=''):
    arr = np.zeros(shape)
    for coord in coordinates:
        x, y, z = coord
        arr[int(x) - 1][int(y) - 1][int(z) - 1] = 1
    if save:
        np.save('D:\Work_file\det\\npy_data\\{}_mask.npy'.format(name), arr)
    return arr


def create_whd(coordinates, whd, shape, save=False):
    arr = np.zeros((3,) + shape)
    for i in range(len(coordinates)):
        x, y, z = coordinates[i]
        arr[0][int(x) - 1][int(y) - 1][int(z) - 1] = whd[i][0]
        arr[1][int(x) - 1][int(y) - 1][int(z) - 1] = whd[i][1]
        arr[2][int(x) - 1][int(y) - 1][int(z) - 1] = whd[i][2]
    if save:
        np.save('array.npy', arr)
    return arr


def create_offset(coordinates, shape, save=False):
    arr = np.zeros((3,) + shape)
    for coord in coordinates:
        x, y, z = coord
        arr[0][int(x) - 1][int(y) - 1][int(z) - 1] = x - int(x)
        arr[1][int(x) - 1][int(y) - 1][int(z) - 1] = y - int(y)
        arr[2][int(x) - 1][int(y) - 1][int(z) - 1] = z - int(z)
    if save:
        np.save('array.npy', arr)
    return arr

# * load time is 0.09s
def create_hmap(coordinates, shape, save=None, name=''): # 1.37s, if save :4.33s
    arr = np.zeros(shape)
    for coord in coordinates:
        x, y, z = coord
        arr[int(x) - 1][int(y) - 1][int(z) - 1] = 1
    # time_si = time()
    arr = gaussian_filter(arr, sigma=2)
    # print('time of si is {}'.format(time() - time_si))
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    if save != None:
        np.save('D:\Work_file\det\\npy_data\\{}_hmap.npy'.format(name), arr)
    return arr



if __name__ == '__main__':
#     # filename = 'D:\Work_file\det_LUNA16_data\subset2\\1.3.6.1.4.1.14519.5.2.1.6279.6001.964952370561266624992539111877.mhd'
#     # # itkimage = sitk.ReadImage(filename)  # 读取.mhd文件
#     # # numpyImage = sitk.GetArrayFromImage(itkimage)  # 获取数据，自动从同名的.raw文件读取
#     # image_nii = tio.ScalarImage(filename)
#     # im = image_nii.data[0, :, :, :]
#     # im = np.array(im)
#     # # embed()
#     # # data = numpyImage[50]
#     # # data = numpyImage
#     # plt.figure(300)
#     # plt.imshow(im[:, :, 100], cmap='gray')
#     # name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.964952370561266624992539111877'
    name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.100953483028192176989979435275'
    new_coords = resize_data(name, new_shape=(514, 514, 277))
    embed()
#     # im = get_segmented_lungs(im[:, :, 100], plot=True)
#     # # npy2nii(name, im)
#     # plt.figure(200)
#     # data = mask_seg_3d(name, forsee=True)
#     # plt.imshow(im, cmap='gray')
#     # plt.show()
#     # time_load = time()
#     # npy = np.load('D:\Work_file\det\\npy_data\\1.3.6.1.4.1.14519.5.2.1.6279.6001.106379658920626694402549886949_hmap.npy')
#     # print('load time is {}'.format(time() - time_load))



#     new_coords = resize_coords(name, new_shape=(514, 514, 277))
    # embed()















# if __name__ == '__main__':

    # mhd_files = get_mhd_files('D:\Work_file\det_LUNA16_data')
#     example_path = mhd_files[0]['path']
    # find_the_orientation(mhd_files)
#     # min_shape, min_name = get_min_shape_with_torchio(mhd_files)
#     mhd_path = 'D:\Work_file\det_LUNA16_data\subset8\\1.3.6.1.4.1.14519.5.2.1.6279.6001.997611074084993415992563148335.mhd'
    
    # name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.997611074084993415992563148335'
    # image_npy = np.load('D:\Work_file\det\\npy_data\\1.3.6.1.4.1.14519.5.2.1.6279.6001.997611074084993415992563148335_image.npy')

    # npy2nii(name, image_npy)
#     name = 'D:\Work_file\det_LUNA16_data\subset2\\1.3.6.1.4.1.14519.5.2.1.6279.6001.964952370561266624992539111877.mhd'
#     image_a = tio.ScalarImage(mhd_path)
#     image_b = tio.ScalarImage(name)
#     result = find_name_in_csv(mhd_name)
#     heatmap = generate_heatmap(mhd_name, save=True)
#     # visualize_3d(heatmap)
#     print(result)


#     info_dict = get_WHD_offset_mask(mhd_name, whd=True, offset=True, mask=True)
#     embed()
#     print(mhd_files)
#     file_path = 'D:\\Work_file\\det_LUNA16_data\\annotations_pathcoord.csv'
#     names = read_names_from_csv(file_path)

#     image_crop, hmap_crop, whd_crop, offset_crop, mask_crop = crop_data(mhd_name)
#     embed()
#     print(names)