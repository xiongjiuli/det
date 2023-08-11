import csv
from utils_v2 import resize_data
from model.resnet import CenterNet
# from model.resnet_v0 import CenterNet
import torch
import numpy as np
import torch
from torch import nn
from torchvision.ops import nms
import torchio as tio
from IPython import embed
import csv
import torch.nn.functional as F
import os
from tqdm import tqdm

from model.SwinUNet import SwinTransformerSys3D
from model.hourglass3d_v1 import Hourglass
from model.hourglass3d_meilan import get_hourglass
from model.swinunet3d_v1 import swinUnet_p_3D




def npy2nii(name, image_npy, suffix=''):
    image_npy = image_npy.cpu()
    affine = np.array([[0.7, 0, 0, 0], [0, 0.7, 0, 0], [0, 0, 1.2, 0], [0, 0, 0, 1]])
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
    elif len(image_npy.shape) == 5:
        # image_npy = torch.from_numpy(image_npy)
        image_nii = tio.ScalarImage(tensor=image_npy[0, :, :, :,:], affine=affine)
        image_nii.save('./nii_temp/{}_{}.nii'.format(name, suffix))
    else: 
        print('DIM ERROR : npy.dim != 3 or 4 or 5')


def centerwhd_2nodes(centers, dimensions_list, point, hmap_scores=None):
    if hmap_scores != None:
        result = []
        x_sta, y_sta, z_sta = point
        for center, dimensions, hmap_score in zip(centers, dimensions_list, hmap_scores):

            x, y, z = center
            # embed()
            length, width, height = dimensions
            x1 = x - length/2.0
            y1 = y - width/2.0
            z1 = z - height/2.0
            x2 = x + length/2.0
            y2 = y + width/2.0
            z2 = z + height/2.0
            x1 += x_sta
            x2 += x_sta
            y1 += y_sta
            y2 += y_sta
            z1 += z_sta
            z2 += z_sta
            result.append([hmap_score, x1, y1, z1, x2, y2, z2])
        return result
    
    else:
        result = []
        x_sta, y_sta, z_sta = point
        for center, dimensions in zip(centers, dimensions_list):

            x, y, z = center
            # embed()
            length, width, height = dimensions
            x1 = x - length / 2.0
            y1 = y - width / 2.0
            z1 = z - height / 2.0
            x2 = x + length / 2.0
            y2 = y + width / 2.0
            z2 = z + height / 2.0
            x1 += x_sta
            x2 += x_sta
            y1 += y_sta
            y2 += y_sta
            z1 += z_sta
            z2 += z_sta
            result.append([x1, y1, z1, x2, y2, z2])

        return result




def calculate_iou(box1, box2):
    # 计算两个框的交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    z1 = max(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    y2 = min(box1[4], box2[4])
    z2 = min(box1[5], box2[5])

    x = ((box1[0] + box1[3]) / 2. - (box2[0] + box2[3]) / 2.).cpu()
    y = ((box1[1] + box1[4]) / 2. - (box2[1] + box2[4]) / 2.).cpu()
    z = ((box1[2] + box1[5]) / 2. - (box2[2] + box2[5]) / 2.).cpu()
    
    offset = np.sqrt((x**2 + y**2 + z**2))
    intersection = max(0, x2 - x1) * max(0, y2 - y1) * max(0, z2 - z1)
    
    
    # 计算两个框的并集
    box1_volume = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    box2_volume = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    union = box1_volume + box2_volume - intersection
    
    # 计算IoU
    iou = intersection / union
    return iou, offset



def pool_nms(heat, kernel = 5):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool3d(heat, (kernel, kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep



def decode_bbox(pred_hms, pred_whds, pred_offsets, scale, confidence, reduce, point, cuda):

    pred_hms    = pool_nms(pred_hms)
    heat_map    = pred_hms[0, :, :, :].cpu()
    pred_whd    = pred_whds[0, :, :, :].cpu()
    pred_offset = pred_offsets[0, :, :, :].cpu()

    mask = torch.from_numpy(np.where(heat_map > confidence, 1, 0)).squeeze(1).bool()
    # mask[0, 50, 60, 30] = 1
    indices = np.argwhere(mask == 1)

    centers = []
    whds = []
    hmap_scores = []
    for i in range(indices.shape[1]):
        coord = indices[1 :, i]
        x = coord[0].cpu()
        y = coord[1].cpu()
        z = coord[2].cpu()
        
        offset_x = pred_offset[0, x, y, z]
        offset_y = pred_offset[1, x, y, z]
        offset_z = pred_offset[2, x, y, z]
        # embed()
        hmap_score = heat_map[0, x, y, z]
        # print(f'--x y z -- : {x}, {y}, {z}')
        w = pred_whd[0, x, y, z] / scale[0]
        h = pred_whd[1, x, y, z] / scale[1]
        d = pred_whd[2, x, y, z] / scale[2]

        center = ((x + offset_x) * reduce, (y + offset_y) * reduce, (z + offset_z) * reduce)
        center = [a / b for a, b in zip(center, scale)]
        origin_whd = (w, h, d) 
        # print(f'center: {center}, origin_whd : {origin_whd}')

        centers.append(center)
        whds.append(origin_whd)
        hmap_scores.append(hmap_score)
    predicted_boxes = centerwhd_2nodes(centers, dimensions_list=whds, point=point, hmap_scores=hmap_scores)
    # print(f'predicted_boxes is : {predicted_boxes}')

    return predicted_boxes


def pad_image(image, target_size):
    # 计算每个维度需要填充的数量
    padding = [(0, max(0, target_size - size)) for size in image.shape]
    # 使用pad函数进行填充
    padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    # 返回填充后的图像
    return padded_image


def sliding_window_3d_volume_padded(arr, patch_size, stride, padding_value=0):
    """
    This function takes a 3D numpy array representing a 3D volume and returns a 4D array of patches extracted using a sliding window approach.
    The input array is padded to ensure that its dimensions are divisible by the patch size.
    :param arr: 3D numpy array representing a 3D volume
    :param patch_size: size of the cubic patches to be extracted
    :param stride: stride of the sliding window
    :param padding_value: value to use for padding
    :return: 4D numpy array of shape (num_patches, patch_size, patch_size, patch_size)
    """
    # regular the shape
    if len(arr.shape) != 3:
        arr = arr.squeeze(0)

    patch_size_x = patch_size[0]
    patch_size_y = patch_size[1]
    patch_size_z = patch_size[2]
    stride_x = stride[0]
    stride_y = stride[1]
    stride_z = stride[2]
    # Compute the padding size for each dimension
    pad_size_x = (patch_size_x - (arr.shape[0] % patch_size_x)) % patch_size_x
    pad_size_y = (patch_size_y - (arr.shape[1] % patch_size_y)) % patch_size_y
    pad_size_z = (patch_size_z - (arr.shape[2] % patch_size_z)) % patch_size_z

    # Pad the array
    arr_padded = np.pad(arr, ((0, pad_size_x), (0, pad_size_y), (0, pad_size_z)), mode='constant', constant_values=padding_value)

    # Extract patches using a sliding window approach
    patches = []
    order = 0
    for i in range(0, arr_padded.shape[0] - patch_size_x + 1, stride_x):
        for j in range(0, arr_padded.shape[1] - patch_size_y + 1, stride_y):
            for k in range(0, arr_padded.shape[2] - patch_size_z + 1, stride_z):
                patch = arr_padded[i:i + patch_size_x, j:j + patch_size_y, k:k + patch_size_z]
                if isinstance(patch, np.ndarray):
                    patch = torch.from_numpy(patch).unsqueeze(0)
                else:
                    patch = patch.unsqueeze(0)
                start_point = torch.tensor([order, i, j, k])
                add = {'image': patch, 'point': start_point}
                patches.append(add)
                order += 1
    # return np.array(patches)
    return patches


def test_crop(model_path, patch_size, overlap=15, confidence=0.15, iou_threshold=0.01):

    test_names = []
    with open('D:\Work_file\det\csv_file\\test_names_server.csv') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            # print(row)
            test_names.append(row[0])

    # model = CenterNet('resnet101', 1)
    # model = CenterNet('resnet101', 1)
    # model = SwinTransformerSys3D()
    # model = Hourglass()
    # model = get_hourglass['large_hourglass']


    #* the swinunet3d config
    x = torch.randn((1, 1, 224, 224, 160))
    window_size = [i // 32 for i in x.shape[2:]]
    model = swinUnet_p_3D(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24),
                    window_size=window_size, in_channel=1, num_classes=64
                    )




    model.cuda()
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    scale = [1., 1., 1.]
    step = [pahs - ovlap for pahs, ovlap in zip(patch_size, overlap)]
    no_pred_dia = []
    for name in tqdm(test_names):
        image = tio.ScalarImage(f'D:\Work_file\det\data_seg_crop\\{name}_croplung.nii.gz')
        image_data = image.data
        shape = image.shape[1:]
        image_patches = sliding_window_3d_volume_padded(image_data, patch_size=patch_size, stride=step)
        # embed() 
        #* the image_patch is a list consist of the all patch of a whole image
        #* each element in the list is a dict consist of start point and tensor(input)
        label_xyz, label_whd = name2coord(name)
        print(f'========================{name}========================')
        pred_bboxes = []
        for image_patch in image_patches:
            with torch.no_grad():
                image_input = image_patch['image'].unsqueeze(0)
                point = image_patch['point'][1:]
                order = image_patch['point'][0]
                image_input = image_input.cuda()
                # embed()
                # print(f'the input image shape is {image_input.shape}')
                pred_hmap, pred_whd, pred_offset = model(image_input)

                pred_bbox = decode_bbox(pred_hmap, pred_whd, pred_offset, scale, confidence, reduce=1., cuda=True, point=point)
                if len(pred_bbox) > 0:
                    pass
                    # npy2nii(name, image_input, suffix=f'{order}-{len(pred_bbox)}-image.nii')
                    # npy2nii(name, pred_hmap, suffix=f'{order}-{len(pred_bbox)}-hmap.nii')
                pred_bboxes.append(pred_bbox)

        ground_truth_boxes = centerwhd_2nodes(centers=label_xyz, dimensions_list=label_whd, point=(0, 0, 0))
        
        # do the nms
        pred_bboxes = normal_list(pred_bboxes)
        pred_bboxes = non_overlapping_boxes(pred_bboxes)
        # pred_bboxes = nms(pred_bboxes, thres=0.5)
        print(f'the gt bbox is {ground_truth_boxes}')
        print(f'the pred bbox is {pred_bboxes}')
        draw_boxes_on_nii(name, ground_truth_boxes, pred_bboxes)
        no_predbox = filter_boxes(ground_truth_boxes, pred_bboxes)
        no_pred_dia.extend(no_predbox)
        for bbox in pred_bboxes:
            hmap_score, x1, y1, z1, x2, y2, z2 = bbox
            w = x2 - x1
            h = y2 - y1
            d = z2 - z1
            with open(f'D:\Work_file\Object-Detection-Metrics-3D\samples\sample_2\detections\\{name}.txt', 'a') as f:
                f.write(f'nodule {hmap_score} {x1} {y1} {z1} {x2} {y2} {z2} {shape}\n')

    print(f'the no pred dia is {no_pred_dia}')
    return print('Done')


def normal_list(list):
    new_list = []
    for lit in list:
        if lit == []:
            continue
        else:
            for l in lit:
                new_list.append(l)
    return new_list

def filter_boxes(gt, pred):
    result = []
    for box in gt:
        overlap = False
        for p_box in pred:
            if (box[0] < p_box[4] and box[3] > p_box[1]) and (box[1] < p_box[5] and box[4] > p_box[2]) and (box[2] < p_box[6] and box[5] > p_box[3]):
                overlap = True
                break
        if not overlap:
            result.append(min(box[3]-box[0], box[4]-box[1], box[5]-box[2]))
    return result

def non_overlapping_boxes(boxes):
    non_overlapping = []
    for i, box1 in enumerate(boxes):
        overlapping = False
        for j, box2 in enumerate(non_overlapping):
            if boxes_overlap(box1, box2):
                overlapping = True
                if box_area(box1) > box_area(box2):
                    non_overlapping.remove(box2)
                    non_overlapping.append(box1)
                break
        if not overlapping:
            non_overlapping.append(box1)
    return non_overlapping

def nms(dets, thres):
    '''
    https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    :param dets:  [[x1,y1,x2,y2,score], [x1,y1,x2,y2,score],,,]
    :param thres: for example 0.5
    :return: the rest ids of dets
    '''
    x1 = [det[1] for det in dets]
    y1 = [det[2] for det in dets]
    z1 = [det[3] for det in dets]
    x2 = [det[4] for det in dets]
    y2 = [det[5] for det in dets]
    z2 = [det[6] for det in dets]
    areas = [(x2[i] - x1[i]) * (y2[i] - y1[i]) * (z2[i] - z1[i]) for i in range(len(x1))]
    scores = [det[0] for det in dets]
    order = order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = [max(x1[i], x1[j]) for j in order[1:]]
        xx2 = [min(x2[i], x2[j]) for j in order[1:]]
        yy1 = [max(y1[i], y1[j]) for j in order[1:]]
        yy2 = [min(y2[i], y2[j]) for j in order[1:]]
        zz1 = [max(z1[i], z1[j]) for j in order[1:]]
        zz2 = [min(z2[i], z2[j]) for j in order[1:]]

        w = [max(xx2[i] - xx1[i] + 1.0, 0.0) for i in range(len(xx1))]
        h = [max(yy2[i] - yy1[i] + 1.0, 0.0) for i in range(len(yy1))]
        d = [max(zz2[i] - zz1[i] + 1.0, 0.0) for i in range(len(zz1))]

        inters = [w[i] * h[i] * d[i] for i in range(len(w))]
        unis = [areas[i] + areas[j] - inters[k] for k, j in enumerate(order[1:])]
        ious = [inters[i] / unis[i] for i in range(len(inters))]

        inds = [i for i, val in enumerate(ious) if val <= thres]
         # return the rest boxxes whose iou<=thres

        order = [order[i + 1] for i in inds]

            # inds + 1]  # for exmaple, [1,0,2,3,4] compare '1', the rest is 0,2 who is the id, then oder id is 1,3
    result = [dets[i] for i in keep]

    return result


def boxes_overlap(box1, box2):
    x1, y1, z1, x2, y2, z2 = [np.float16(x) for x in box1[1:]]
    a1, b1, c1, a2, b2, c2 = [np.float16(x) for x in box2[1:]]
    return not (x2 < a1 or a2 < x1 or y2 < b1 or b2 < y1 or z2 < c1 or c2 < z1)

def box_area(box):
    _, x1, y1, z1, x2, y2, z2 = box
    return (x2 - x1) * (y2 - y1) * (z2 - z1)


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


from PIL import Image
def draw_boxes_on_nii(name, ground_truth_boxes, predicted_boxes):
    # 将nii图像转换为numpy数组
    nii_image = tio.ScalarImage(f'D:\Work_file\det\data_seg_crop\{name}_croplung.nii.gz')
    nii_image = nii_image.data.squeeze(0)
    nii_data = np.array(nii_image)
    nii_data_p = np.array(nii_image)

    # 创建一个空白的四维数组
    color_image = np.zeros((nii_image.shape[0], nii_image.shape[1], nii_image.shape[2], 3))
    # 将灰度图像复制到新数组的每个颜色通道中
    color_image[..., 0] = nii_data
    color_image[..., 1] = nii_data
    color_image[..., 2] = nii_data

    # 创建一个空白的四维数组
    color_image_p = np.zeros((nii_image.shape[0], nii_image.shape[1], nii_image.shape[2], 3))
    # 将灰度图像复制到新数组的每个颜色通道中
    color_image_p[..., 0] = nii_data_p
    color_image_p[..., 1] = nii_data_p
    color_image_p[..., 2] = nii_data_p


    # 遍历 ground_truth_boxes中的每个框
    for box in ground_truth_boxes:
        # 获取框的坐标
        # x1, y1, z1, x2, y2, z2 = box
        x1, y1, z1, x2, y2, z2 = map(int, box)
        # embed()
        # 在nii数据上绘制框
        color_image[x1:x2+1, y1:y2+1, z1, 0] = 1.0
        color_image[x1:x2+1, y1:y2+1, z2, 0] = 1.0
        color_image[x1:x2+1, y1, z1:z2+1, 0] = 1.0
        color_image[x1:x2+1, y2, z1:z2+1, 0] = 1.0
        color_image[x1, y1:y2+1, z1:z2+1, 0] = 1.0
        color_image[x2, y1:y2+1, z1:z2+1, 0] = 1.0

    slice_gt_array = []
    for box in ground_truth_boxes:
        # 获取框的坐标
        # x1, y1, z1, x2, y2, z2 = box
        x1, y1, z1, x2, y2, z2 = map(int, box)
        c_x = int((x1 + x2) / 2)
        c_y = int((y1 + y2) / 2)
        c_z = int((z1 + z2) / 2)
        # embed()
        slice_x = color_image[c_x, :, :, :]
        slice_y = color_image[:, c_y, :, :]
        slice_z = color_image[:, :, c_z, :]

        slice_gt_array.append(slice_x)
        slice_gt_array.append(slice_y)
        slice_gt_array.append(slice_z)

    # 遍历predicted_boxes中的每个框
    for box_p in predicted_boxes:
        # 获取框的坐标
        # x1, y1, z1, x2, y2, z2 = box_p
        x1, y1, z1, x2, y2, z2 = map(int, box_p[1:])
        
        # 在nii数据上绘制框
        color_image_p[x1:x2+1, y1:y2+1, z1, 0] = 1.0
        color_image_p[x1:x2+1, y1:y2+1, z2, 0] = 1.0
        color_image_p[x1:x2+1, y1, z1:z2+1, 0] = 1.0
        color_image_p[x1:x2+1, y2, z1:z2+1, 0] = 1.0
        color_image_p[x1, y1:y2+1, z1:z2+1, 0] = 1.0
        color_image_p[x2, y1:y2+1, z1:z2+1, 0] = 1.0

    slice_gt_array_p = []
    for box_p in predicted_boxes:
        # 获取框的坐标
        # x1, y1, z1, x2, y2, z2 = box_p
        x1, y1, z1, x2, y2, z2 = map(int, box_p[1:])
        c_x = int((x1 + x2) / 2)
        c_y = int((y1 + y2) / 2)
        c_z = int((z1 + z2) / 2)

        slice_x = color_image_p[c_x, :, :, :]
        slice_y = color_image_p[:, c_y, :, :]
        slice_z = color_image_p[:, :, c_z, :]

        slice_gt_array_p.append(slice_x)
        slice_gt_array_p.append(slice_y)
        slice_gt_array_p.append(slice_z)

    # 将数组转换为图像
    images = [Image.fromarray((arr * 255 ).astype(np.uint8)) for arr in slice_gt_array] #!
    images_p = [Image.fromarray((arr * 255 ).astype(np.uint8)) for arr in slice_gt_array_p]

    # 将图片缩放到相同的大小
    new_size = (400, 400)
    images = [image.resize(new_size) for image in images]
    images_p = [image.resize(new_size) for image in images_p]

    # 计算新图片的大小
    width, height = new_size
    gap = 10
    number = max(len(images), len(images_p))
    merged_width = (width * 3 + gap * 2 + gap + 10) * number / 3
    merged_height = height * 2 + gap + 10

    # 创建一个新的空白图片，用于存放合并后的图片
    # embed()
    merged_image = Image.new('RGB', (int(merged_width), int(merged_height)), (255, 255, 255))

    # 将六张图片合并到新图片中
    x, y = 0, 0
    for i in range(len(images)):
        merged_image.paste(images[i], (x, y))
        if (i+1) % 3 == 0: 
            x += 420 
        else:
            x += 410

    a, b = 0, 420
    for i in range(len(images_p)):
        merged_image.paste(images_p[i], (a, b))
        if (i+1) % 3 == 0: 
            a += 420
        else:
            a += 410
            

        # 保存合并后的图片
    merged_image.save(os.path.join('D:\Work_file\det\image_data', f"{name[-9:]}.png"))



if __name__ == '__main__':


    model_path = 'D:\Work_file\det\save\\best_model_swin3d_crop224160_0728-310.pt'
    patch_size = [224, 224, 160]
    overlap = [15, 15, 15]
    test_crop(model_path, patch_size, overlap, confidence=0.15, iou_threshold=0.01)



