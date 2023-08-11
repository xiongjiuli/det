import os
import matplotlib.pyplot as plt
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




def name2coord(mhd_name, root_dir='D:\Work_file\det'):
    # * 输入name，输出这个name所对应着的gt坐标信息
    xyz = []
    whd = []
    # csv_file_dir = 'D:\\Work_file\\det_LUNA16_data\\annotations_pathcoord.csv'
    csv_file_dir = os.path.join(root_dir, 'annotations_resmaple_segcrop_guanfang.csv')
    with open(csv_file_dir, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == mhd_name:
                # print(row[0])
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
                # radius = float(row[5])
                # result.append((x, y, z, radius))
                w = float(row[4])
                h = float(row[5])
                d = float(row[6])
                xyz.append((x, y, z))
                whd.append((w, h, d))
                # print(xyz)
                # print(whd)
    # print(f'xyz : {xyz}, whd : {whd}  in name2coord func')
    # embed()
    return xyz, whd



def create_hmap(coordinates, shape, reduce=4, save=None, hmap_dir=''): # 1.37s, if save :4.33s
    arr = np.zeros(tuple(np.array(shape) // reduce))
    for coord in coordinates:
        x, y, z = coord
        arr[int(x / reduce)][int(y / reduce)][int(z / reduce)] = 1
    # time_si = time()
    arr = gaussian_filter(arr, sigma=3)
    # print('time of si is {}'.format(time() - time_si))
    if arr.max() == arr.min():
        if save != None:
            np.save(hmap_dir, arr)
        return arr
    else:
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        if save != None:
            np.save(hmap_dir, arr)
        return arr


def crop_data(name, root_dir, crop_length=128, mode='train'):
    path = 'D:\Work_file\det\data_seg_crop\{}_croplung.nii.gz'.format(name)
    image = tio.ScalarImage(path)
    new_shape = (crop_length, crop_length, crop_length)
    origin_coords, origin_whd = name2coord(name)
    # 生成一个随机数
    p = random.random()
    image_shape = image.shape[1:]
    idx = random.randint(0, len(origin_coords) - 1)
    label = origin_coords[idx]
    whd_choose = origin_whd[idx]
    random.seed(0)

    if p < 0.8:
        # 有选择的裁剪

        x, y, z = label
        w, h, d = whd_choose
        x_min = int(x - w / 2)
        y_min = int(y - h / 2)
        z_min = int(z - d / 2)
        # logger.info(f'x_crop between {(max(0, x_min - crop_length), min(x_min, image_shape[0] - crop_length))},\n\
        #               y_crop between {(max(0, y_min - crop_length), min(y_min, image_shape[1] - crop_length))},\n\
        #               z_crop between {(max(0, z_min - crop_length), min(z_min, image_shape[2] - crop_length))},\n\
        #               label is {label}, x_min is {x_min}, y_min is {y_min}, z_min is {z_min}, image shape is {image.shape}')
        # 考虑到万一边截止的范围比开始的要小的话，就去强行变化范围
        x_sta = max(0, x_min - crop_length)
        x_stop = min(x_min, image_shape[0] - crop_length)
        y_sta = max(0, y_min - crop_length)
        y_stop = min(y_min, image_shape[1] - crop_length)
        z_sta = max(0, y_min - crop_length)
        z_stop = min(z_min, image_shape[2] - crop_length)
        if x_sta >= x_stop:
            x_sta = x_stop - 10
        if y_sta >= y_stop:
            y_sta = y_stop - 10
        if z_sta >= z_stop:
            z_sta = z_stop - 10
        x_crop = random.randint(x_sta, x_stop)
        y_crop = random.randint(y_sta, y_stop)
        z_crop = random.randint(z_sta, z_stop)
    else:
        if image_shape[0] - crop_length <= 0:
            x_crop = 0
        else:
            x_crop = random.randint(0, image_shape[0] - crop_length)
        if image_shape[1] - crop_length <= 0:
            y_crop = 0
        else:
            y_crop = random.randint(0, image_shape[1] - crop_length)
        if image_shape[2] - crop_length <= 0:
            z_crop = 0
        else:
            z_crop = random.randint(0, image_shape[2] - crop_length)


    # 考虑到万一整个的image最短边小于被crop的长度的话，就去padding
    if (image_shape[0] - crop_length) <= 0 or x_crop < 0:
        x_crop = 0
    elif (image_shape[1] - crop_length) <= 0 or y_crop < 0:
        y_crop = 0
    elif (image_shape[2] - crop_length) <= 0 or z_crop < 0:
        z_crop = 0
    
    # 确定这个被crop图像的start point
    start_point = (x_crop, y_crop, z_crop)
    # print(f'the start point is {start_point}')
    # print(f'the image shape is {image.shape}')

    if (image_shape[0] - crop_length) < 0 or (image_shape[1] - crop_length) < 0 or (image_shape[2] - crop_length) < 0:
        image_crop = crop_padding(image.data[0, :, :, :], start_point, size=(crop_length, crop_length, crop_length))
    else:
        image_crop = image.data[0, x_crop : x_crop + crop_length,\
                                   y_crop : y_crop + crop_length,\
                                   z_crop : z_crop + crop_length,]

    new_coords = process_boxes(origin_coords, origin_whd, (x_crop, y_crop, z_crop))
    print(f'the new coords is {new_coords}, the origin_croods is {origin_coords}, the start point is {(x_crop, y_crop, z_crop)}')
    #* bulid the other label
    mask = create_mask(new_coords, new_shape, reduce=1) # 0.0s no save is so fast
    whd = create_whd(coordinates=new_coords, whd=origin_whd, shape=new_shape, reduce=1)
    offset = create_offset(coordinates=new_coords, shape=new_shape, reduce=1)
    hmap = create_hmap(coordinates=new_coords, shape=new_shape, reduce=1)

    hmap = torch.from_numpy(hmap).unsqueeze(0).unsqueeze(0)
    offset = torch.from_numpy(offset).unsqueeze(0)
    mask = torch.from_numpy(mask)
    whd = torch.from_numpy(whd).unsqueeze(0)
    
    dict = {}
    dict['hmap'] = hmap
    dict['offset'] = offset
    dict['mask'] = mask
    dict['input'] = image_crop
    dict['new_coords'] = new_coords
    dict['name'] = name
    dict['origin_whd'] = origin_whd
    dict['origin_coords'] = origin_coords
    dict['whd'] = whd

    return dict


def process_boxes(boxes, origin_whd, coord):
    result = []
    for i in range(len(boxes)):
        x, y, z = boxes[i]
        w, h, d = origin_whd[i]
        x -= coord[0]
        y -= coord[1]
        z -= coord[2]
        if (x - w/2 < 0 or y - h/2 < 0 or z - d/2 < 0) or (x + w/2 >= 128 or y + h/2 >= 128 or z + d/2 >= 128):
            continue
        result.append([x, y, z])
    return result


def crop_padding(image, start_point, size):
    # 计算裁剪区域的坐标范围
    x_min = start_point[0] - size[0] // 2
    x_max = start_point[0] + size[0] // 2
    y_min = start_point[1] - size[1] // 2
    y_max = start_point[1] + size[1] // 2
    z_min = start_point[2] - size[2] // 2
    z_max = start_point[2] + size[2] // 2
    
    # 计算需要填充的大小
    pad_x_min = max(0, -x_min)
    pad_x_max = max(0, x_max - image.shape[0])
    pad_y_min = max(0, -y_min)
    pad_y_max = max(0, y_max - image.shape[1])
    pad_z_min = max(0, -z_min)
    pad_z_max = max(0, z_max - image.shape[2])
    
    # 对图像进行填充
    padded_image = np.pad(image, ((pad_x_min, pad_x_max), (pad_y_min, pad_y_max), (pad_z_min, pad_z_max)), mode='constant', constant_values=0)
    
    # 裁剪图像
    cropped_image = padded_image[x_min+pad_x_min:x_max+pad_x_min, y_min+pad_y_min:y_max+pad_y_min, z_min+pad_z_min:z_max+pad_z_min]
    
    return cropped_image


def resize_data(name, root_dir, new_shape=(512, 512, 256)):
    # time_1 = time()
    path = 'D:\Work_file\det\\data_seg_crop\\{}_croplung.nii.gz'.format(name)
    origin_coords, origin_whd = name2coord(name)
    # embed()
    image = tio.ScalarImage(path)
    affine = np.eye(4)
    shape = image.shape[1:]
    # affine = image.affine
    scale = np.array(new_shape) / np.array(image.shape[1:])
    print(f'the new_shape.shape is {new_shape}, the origin shape is {image.shape[1:]}, and the scale is {scale}')
    new_coords = []
    new_whd = []
    for i in range(len(origin_coords)):
        new_coords.append(origin_coords[i] * scale)
        # new_whd.append((coord[-1], coord[-1], coord[-1]) * scale)
        new_whd.append(origin_whd[i] * scale)

    # create the 1.mask and 2.whd and 3.offset and the 4.image
    mask = create_mask(new_coords, new_shape) # 0.0s no save is so fast
    # mask = resize_and_normalize(mask, new_size=(tuple(np.array(new_shape) // 4)))
    whd = create_whd(coordinates=new_coords, whd=new_whd, shape=new_shape)
    # whd = resize_and_normalize(whd, new_size=(tuple(np.array(new_shape) // 4)))
    offset = create_offset(coordinates=new_coords, shape=new_shape)
    # offset = resize_and_normalize(offset, new_size=(tuple(np.array(new_shape) // 4)))
    # npy2nii(name, mask, suffix='resize_mask', resample=True, affine=affine)
    # npy2nii(name, whd, suffix='resize_whd', resample=True, affine=affine)
    # npy2nii(name, offset, suffix='resize_offset', resample=True, affine=affine)
    
    # hmap_dir = 'D:\Work_file\det\\npy_data\\{}_hmap.npy'.format(name)
    hmap_dir = os.path.join(root_dir, 'npy_data', '{}_hmap.npy'.format(name))
    # hmap_dir = '/public_bme/data/xiongjl/npy_data/{}_hmap.npy'.format(name)
    if os.path.isfile(hmap_dir):
        hmap = np.load(hmap_dir)
    else:
        hmap = create_hmap(coordinates=new_coords, shape=new_shape, save=True, hmap_dir=hmap_dir)
        # hmap = resize_and_normalize(hmap, new_size=(tuple(np.array(new_shape) // 4)))
        # hmap = generate_heatmap_ing(coordinates=new_coords, shape=new_shape, whd=new_whd, reduce=4, save=None, hmap_dir=hmap_dir)

    # npy2nii(name, hmap, suffix='resize_hmap', resample=True, affine=affine)
    # input_data = seg_3d_name(name, root_dir)
    # input_data = torch.from_numpy(input_data).unsqueeze(0).unsqueeze(0).float()
    input_data = image.data.unsqueeze(0).float()
    input_resize = F.interpolate(input_data, size=new_shape).squeeze(0).squeeze(0).numpy()
    input_resize = (input_resize - input_data.numpy().min()) / (input_data.numpy().max() - input_data.numpy().min() + 1e-8)
    # npy2nii(name, input_resize, suffix='resize_image', resample=True, affine=affine)
    
    dict = {}
    dict['hmap'] = hmap
    dict['offset'] = offset
    dict['mask'] = mask
    dict['input'] = input_resize
    dict['new_whd'] = new_whd
    dict['new_coords'] = new_coords
    dict['name'] = name
    dict['scale'] = scale
    dict['origin_whd'] = origin_whd
    dict['origin_coords'] = origin_coords
    dict['whd'] = whd
    return dict

def resize_and_normalize(input_array, new_size):
    # 将输入数组转换为张量
    tensor = torch.tensor(input_array)
    # tensor = input_array.clone().detach()
    
    # 为张量添加批处理和通道维度
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif len(tensor.shape) == 4:
        tensor = tensor.unsqueeze(0)
    else:
        print('the numpy dim != 3 or 4')
    
    # 调整张量大小
    resized_tensor = F.interpolate(tensor, size=new_size, mode='trilinear', align_corners=True)
    
    # 删除批处理和通道维度
    resized_tensor = resized_tensor.squeeze(0).squeeze(0)
    
    # 归一化张量到0和1之间
    normalized_tensor = (resized_tensor - resized_tensor.min()) / (resized_tensor.max() - resized_tensor.min() + 1e-8)
    
    # 将张量转换为numpy数组并返回
    return normalized_tensor.numpy()

def create_mask(coordinates, shape, reduce=4, save=False, name=''):
    
    arr = np.zeros(tuple(np.array(shape) // reduce)) 
    for coord in coordinates:
        x, y, z = coord
        x = x / reduce
        y = y / reduce
        z = z / reduce 
        arr[int(x)][int(y)][int(z)] = 1
    if save:
        np.save('D:\Work_file\det\\npy_data\\{}_mask.npy'.format(name), arr)
    
    return arr

def create_whd(coordinates, whd, shape, reduce=4, save=False):
    
    arr = np.zeros(tuple(np.insert(np.array(shape) // reduce, 0, 3)))
    for i in range(len(coordinates)):
        x, y, z = coordinates[i]
        x = x / reduce
        y = y / reduce
        z = z / reduce 
        arr[0][int(x)][int(y)][int(z)] = whd[i][0]
        arr[1][int(x)][int(y)][int(z)] = whd[i][1]
        arr[2][int(x)][int(y)][int(z)] = whd[i][2]
    if save:
        np.save('array.npy', arr)
    
    return arr

def create_offset(coordinates, shape, reduce=4, save=False):
    arr = np.zeros(tuple(np.insert(np.array(shape) // reduce, 0, 3)))
    for coord in coordinates:
        x, y, z = coord
        x = x / reduce
        y = y / reduce
        z = z / reduce 
        arr[0][int(x)][int(y)][int(z)] = x - int(x)
        arr[1][int(x)][int(y)][int(z)] = y - int(y)
        arr[2][int(x)][int(y)][int(z)] = z - int(z)
    if save:
        np.save('array.npy', arr)
    return arr



def npy2nii(image_npy, suffix=''):
    image_npy = image_npy
    affine = np.array([[0.7, 0, 0, 0], [0, 0.7, 0, 0], [0, 0, 1.2, 0], [0, 0, 0, -1]])
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




from PIL import Image
def save_images(name, coords_list, new_coords_list, output_folder, reduce=4):
    # 读取nii文件
    nii_path = 'D:\Work_file\det\\data_seg_crop\\{}_croplung.nii.gz'.format(name)
    npy_path = 'D:\Work_file\det\\npy_data\\{}_hmap.npy'.format(name)
    nii_data = tio.ScalarImage(nii_path).data.squeeze(0)
    npy_data = np.load(npy_path)
    
    # 遍历坐标列表
    for i in range(len(coords_list)):
        x, y, z = coords_list[i]
        x_h, y_h, z_h = new_coords_list[i]

        x_nii = int(x)
        y_nii = int(y)
        z_nii = int(z)
        x_hmap = int(x_h / reduce)
        y_hmap = int(y_h / reduce)
        z_hmap = int(z_h / reduce)

        # 获取切片数据
        slice_x = nii_data[x_nii, :, :].numpy()
        slice_x[y_nii, :] = 1
        slice_x[:, z_nii] = 1
        slice_y = nii_data[:,  y_nii, :].numpy()
        slice_y[x_nii, :] = 1
        slice_y[:, z_nii] = 1
        slice_z = nii_data[:, :, z_nii].numpy()
        slice_z[x_nii, :] = 1
        slice_z[:, y_nii] = 1
        slice_x_hmap = npy_data[x_hmap, :, :]
        slice_x_hmap[y_hmap, :] = 1
        slice_x_hmap[:, z_hmap] = 1
        slice_y_hmap = npy_data[:, y_hmap, :]
        slice_y_hmap[x_hmap, :] = 1
        slice_y_hmap[:, z_hmap] = 1
        slice_z_hmap = npy_data[:, :, z_hmap]
        slice_z_hmap[x_hmap, :] = 1
        slice_z_hmap[:, y_hmap] = 1

        arrays = [slice_x, slice_y, slice_z, slice_x_hmap, slice_y_hmap, slice_z_hmap]

        # 将数组转换为图像
        images = [Image.fromarray((arr * 255 ).astype(np.uint8)) for arr in arrays] #!

        # 将图片缩放到相同的大小
        new_size = (400, 400)
        images = [image.resize(new_size) for image in images]

        # 计算新图片的大小
        width, height = new_size
        gap = 10
        merged_width = width * 3 + gap * 2
        merged_height = height * 2 + gap

        # 创建一个新的空白图片，用于存放合并后的图片
        merged_image = Image.new('RGB', (merged_width, merged_height), (255, 255, 255))

        # 将六张图片合并到新图片中
        for i in range(6):
            x = i % 3 * (width + gap)
            y = i // 3 * (height + gap)
            merged_image.paste(images[i], (x, y))

        # 保存合并后的图片
        merged_image.save(os.path.join(output_folder, f"{name}_{x_nii}_{y_nii}_{z_nii}.png"))



def get_filenames(path):
    filenames = []
    for filename in os.listdir(path):
        filenames.append(filename.split('_')[0])
        # embed()
    return filenames



import random
import numpy as np

def random_crop_3d(name, crop_size, p=0.8):
    path = 'D:\Work_file\det\data_seg_crop\{}_croplung.nii.gz'.format(name)
    image = tio.ScalarImage(path)
    image = image.data[0, :, :, :]

    # new_shape = (crop_size, crop_size, crop_size)
    origin_coords, origin_whd = name2coord(name)
    width, height, depth = image.shape[:]

    crop_width, crop_height, crop_depth = crop_size
    
    # pad the image if it's smaller than the desired crop size
    pad_width = max(0, crop_width - width)
    pad_height = max(0, crop_height - height)
    pad_depth = max(0, crop_depth - depth)
    if pad_height > 0 or pad_width > 0 or pad_depth > 0:
        # print('padding!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        image = np.pad(image, ((0, pad_width), (0, pad_height), (0, pad_depth)), mode='constant')
        width, height, depth = image.shape[:]
    # import pdb
    # pdb.set_trace()
    if random.random() < p:
        print('have the points')
        # 80% chance to have one or some points in the cropped image
        point = random.choice(origin_coords)
        x, y, z = point
        print(f'the xyz is {x, y, z}')
        # 考虑到要是结束的范围比开始的范围要小的话就去强行变化范围
        x_sta = int(max(0, x - crop_width + 1))
        x_stop = int(min(x + 1, width - crop_width))
        y_sta = int(max(0, y - crop_height + 1))
        y_stop = int(min(y + 1, height - crop_height))
        z_sta = int(max(0, z - crop_depth + 1))
        z_stop = int(min(z + 1, depth - crop_depth))
        if x_sta > x_stop:
            # print('x_sta >= x_stop')
            x_sta = x_stop - 10
        if y_sta > y_stop:
            # print('y_sta >= y_stop')
            y_sta = y_stop - 10
        if z_sta > z_stop:
            # print('z_sta >= z_stop')
            z_sta = z_stop - 10
        x1 = random.randint(x_sta, x_stop)
        x2 = x1 + crop_width
        y1 = random.randint(y_sta, y_stop)
        y2 = y1 + crop_height
        z1 = random.randint(z_sta, z_stop)
        z2 = z1 + crop_depth

    else:
        # 20% chance to randomly crop the image
        x1 = random.randint(0, width - crop_width)
        x2 = x1 + crop_width
        y1 = random.randint(0, height - crop_height)
        y2 = y1 + crop_height
        z1 = random.randint(0, depth - crop_depth)
        z2 = z1 + crop_depth
    
    cropped_image = image[x1:x2, y1:y2, z1:z2]
    # print(f'the start point is {(x1, y1, z1)}')
    cropped_points = [(x-x1,y-y1,z-z1) for (x,y,z) in origin_coords if x1 <= x < x2 and y1 <= y < y2 and z1 <= z < z2]
    # print(f'the origin points is {origin_coords}')
    # print(f'the cropped points is {cropped_points}')
    # new_coords = process_boxes(origin_coords, origin_whd, (x1, y1, z1))
    # print(f'the new coords is {new_coords}, the origin_croods is {origin_coords}, the start point is {(x1, y1, z1)}')
    #* bulid the other label
    mask = create_mask(cropped_points, crop_size, reduce=1) # 0.0s no save is so fast
    whd = create_whd(coordinates=cropped_points, whd=origin_whd, shape=crop_size, reduce=1)
    offset = create_offset(coordinates=cropped_points, shape=crop_size, reduce=1)
    hmap = create_hmap(coordinates=cropped_points, shape=crop_size, reduce=1)

    hmap = torch.from_numpy(hmap).unsqueeze(0).unsqueeze(0)
    offset = torch.from_numpy(offset).unsqueeze(0)
    mask = torch.from_numpy(mask)
    whd = torch.from_numpy(whd).unsqueeze(0)
    
    dict = {}
    dict['hmap'] = hmap
    dict['offset'] = offset
    dict['mask'] = mask
    dict['input'] = cropped_image
    dict['new_coords'] = cropped_points
    dict['name'] = name
    dict['origin_whd'] = origin_whd
    dict['origin_coords'] = origin_coords
    dict['whd'] = whd

    return dict










if __name__ == '__main__':
    name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.124822907934319930841506266464'
    for i in range(10):
        # train_batch = crop_data(name=name, root_dir='', crop_length=256)
        train_batch = random_crop_3d(name, crop_size=(256, 256, 256), p=0.8)
        # npy2nii(cropped_image, f'input_image_forsee_{name[-6:]}')

        image = train_batch['input']
        heatmap = train_batch['hmap']
        wh_size = train_batch['whd']
        regression = train_batch['offset']
        masks = train_batch['mask']
        name = train_batch['name']
        import pdb
        pdb.set_trace()
        from time import time 
        time_1 = time()
        image_savetrain = resize_and_normalize(image, new_size=(256, 256, 256))
        print(time() - time_1)
        npy2nii(image_savetrain, f'input_image_forsee_{name[-6:]}')

        npy2nii(heatmap, f'input_hmap_forseee_{name[-6:]}')
        npy2nii(wh_size, f'input_whd_forseee_{name[-6:]}')
        npy2nii(regression, f'input_offset_forseee_{name[-6:]}')
        npy2nii(masks, f'input_mask_forseee_{name[-6:]}')
    # embed()
    # file_path = 'D:\Work_file\det\\data_seg_crop'
    # names = get_filenames(file_path)
    # output_folder = 'D:\Work_file\det\image_data'
    # for name in tqdm(names):
    #     # 查看原始数据的样子

    #     dict = resize_data(name, '')
    #     coords = dict['origin_coords']
    #     new_coords = dict['new_coords']
    #     # print(f"{name[-8:]}, the coords is {coords}")
    #     save_images(name, coords, new_coords, output_folder)

