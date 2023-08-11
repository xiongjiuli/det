import csv
from utils_v2 import resize_data
from model.resnet_v0 import CenterNet
import torch
import numpy as np
import torch
from torch import nn
from torchvision.ops import nms
import torchio as tio
from IPython import embed
import csv
import torch.nn.functional as F
from collections import defaultdict
import os


def expand_tensor(input_tensor):
    # 压缩第一个维度
    squeezed_tensor = torch.squeeze(input_tensor, 0)
    # 获取原始大小
    # original_size = squeezed_tensor.size()
    # 计算新大小
    # new_size = [dim * 4 for dim in original_size]
    # 使用 interpolate 函数将原始张量扩大到新大小
    expanded_tensor = F.interpolate(squeezed_tensor.unsqueeze(0), size=[512, 512, 256], mode='nearest').squeeze(0)
    # 在第一个维度上添加一个新维度
    expanded_tensor = torch.unsqueeze(expanded_tensor, 0)
    return expanded_tensor


def npy2nii(name, image_npy, suffix=''):
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



def centerwhd_2nodes(centers, dimensions_list, hmap_scores=None):
    if hmap_scores != None:
        result = []
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
            result.append([hmap_score, x1, y1, z1, x2, y2, z2])
        return result
    
    else:
        result = []
        for center, dimensions in zip(centers, dimensions_list):

            x, y, z = center
            # embed()
            length, width, height = dimensions
            x1 = x - length/2.0
            y1 = y - width/2.0
            z1 = z - height/2.0
            x2 = x + length/2.0
            y2 = y + width/2.0
            z2 = z + height/2.0
            result.append([x1, y1, z1, x2, y2, z2])

        return result


def read_names_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过第一行
        names = [row[0] for row in reader]
    return list(names)


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
    # print(f'x : {x}, y : {y}, z : {z}, and the type of the data is {type(x)}')
    
    offset = np.sqrt((x**2 + y**2 + z**2))

    intersection = max(0, x2 - x1) * max(0, y2 - y1) * max(0, z2 - z1)
    
    # 计算两个框的并集
    box1_volume = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    box2_volume = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    union = box1_volume + box2_volume - intersection
    
    # 计算IoU
    iou = intersection / union
    return iou, offset


def calculate_accuracy_and_recall(predicted_boxes, ground_truth_boxes, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # for image_index in range(len(predicted_boxes)):
        # in a image 
    image_predicted_boxes = predicted_boxes#[image_index]
    image_ground_truth_boxes = ground_truth_boxes#[image_index]
    matched_ground_truth_boxes = [False] * len(image_ground_truth_boxes)

    iou_data = []
    offsets = []
    for predicted_box in image_predicted_boxes:
        max_iou = 0
        max_iou_index = -1
        for ground_truth_index, ground_truth_box in enumerate(image_ground_truth_boxes):
            # embed()
            iou, offset = calculate_iou(predicted_box, ground_truth_box)
            if iou > 0:
                iou_data.append(iou)
                offsets.append(offset)

            if iou > max_iou:
                max_iou = iou
                max_iou_index = ground_truth_index
        
        if max_iou >= iou_threshold:
            if not matched_ground_truth_boxes[max_iou_index]:
                true_positives += 1
                matched_ground_truth_boxes[max_iou_index] = True
            else:
                false_positives += 1
        else:
            false_positives += 1
    
    false_negatives += len(image_ground_truth_boxes) - sum(matched_ground_truth_boxes)

    if true_positives == 0:
        precision = 0
        recall = 0
    else:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
    if len(iou_data) == 0:
        iou_mean = 0
    else:
        iou_mean = np.mean(iou_data)
    if len(offsets) == 0:
        offsets_mean = 0
    else:
        offsets_mean = np.mean(offsets)

    return precision, recall, iou_mean, offsets_mean



def pool_nms(heat, kernel = 3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool3d(heat, (kernel, kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def decode_bbox(pred_hms, pred_whds, pred_offsets, scale, confidence, reduce, cuda):

    pred_hms = pool_nms(pred_hms)

    heat_map    = pred_hms[0, :, :, :]
    pred_whd     = pred_whds[0, :, :, :]
    pred_offset = pred_offsets[0, :, :, :]

    mask = torch.from_numpy(np.where(heat_map > confidence, 1, 0)).squeeze(1).bool()
    # mask[0, 50, 60, 30] = 1
    indices = np.argwhere(mask == 1)
    # indices += 1
    # print(indices)
    # embed()
    
    centers = []
    whds = []
    hmap_scores = []
    for i in range(indices.shape[1]):
        coord = indices[1 :, i]
        x = coord[0]
        y = coord[1]
        z = coord[2]
        
        offset_x = pred_offset[0, x, y, z]
        offset_y = pred_offset[1, x, y, z]
        offset_z = pred_offset[2, x, y, z]

        hmap_score = heat_map[0, x, y, z]

        w = pred_whd[0, x, y, z] / scale[0]
        h = pred_whd[1, x, y, z] / scale[1]
        d = pred_whd[2, x, y, z] / scale[2]
        # embed()
        center = ((x + offset_x) * reduce, (y + offset_y) * reduce, (z + offset_z) * reduce) / scale
        origin_whd = (w, h, d) 
        # print(f'center: {center}, origin_whd : {origin_whd}')

        centers.append(center)
        whds.append(origin_whd)
        hmap_scores.append(hmap_score)
    predicted_boxes = centerwhd_2nodes(centers, dimensions_list=whds, hmap_scores=hmap_scores)
    # print(f'predicted_boxes is : {predicted_boxes}')

    return predicted_boxes


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


def test_luna16(name, model_path):

    # file_path = ''
    root_dir = 'D:\Work_file\det'
    # names = read_names_from_csv(file_path)
    test_batch = resize_data(name, root_dir, new_shape=(512, 512, 256))

    model = CenterNet('resnet101', 1)
    # model = CenterNet(config.backbone_name, 3)
    # model.cuda()

    model.load_state_dict(torch.load(model_path)['model'])

    with torch.no_grad():
        # 清理缓存
        # torch.cuda.empty_cache()
        image = test_batch['input']
        image = torch.from_numpy(image)
        image = image.unsqueeze(0).unsqueeze(0)
        name = test_batch['name']
        coords = test_batch['origin_coords']
        whd = test_batch['origin_whd']
        scale = test_batch['scale']
        # embed()
        pred_hmap, pred_whd, pred_offset = model(image)
        # image_savetrain = resize_and_normalize(image, new_size=(128, 128, 64))
        # npy2nii(name=name[-4 :], image_npy=pred_hmap, suffix='pred_hmap')
        # npy2nii(name=name[-4 :], image_npy=image_savetrain, suffix='input_image')
        shape = [512, 512, 256] / scale
        predicted_boxes = decode_bbox(pred_hmap, pred_whd, pred_offset, scale, confidence=0.25, reduce=4, cuda=None)

        # print(f'predicted boxes is {predicted_boxes}')
        # pred_bboxes = normal_list(pred_bboxes)
        # pred_bboxes = non_overlapping_boxes(pred_bboxes)
        # no_predbox = filter_boxes(ground_truth_boxes, pred_bboxes)
        # no_pred_dia.extend(no_predbox)
        if predicted_boxes == []:
            pass
        else:
            for bbox in predicted_boxes:
                hmap_score, x1, y1, z1, x2, y2, z2 = bbox
                w = x2 - x1
                h = y2 - y1
                d = z2 - z1
                with open(f'D:\Work_file\Object-Detection-Metrics-3D\samples\sample_2\detections\\{name}.txt', 'a') as f:

                    f.write(f'nodule {hmap_score} {x1} {y1} {z1} {x2} {y2} {z2} {shape}\n')
        # # ground_truth_boxes = read_and_convert_data(name)
        # # embed()
        # ground_truth_boxes = centerwhd_2nodes(centers=coords, dimensions_list=whd)
        # # embed()
        # precision, recall, iou, offset = calculate_accuracy_and_recall(predicted_boxes, ground_truth_boxes, iou_threshold=0.3)
        
        # print(f'accuracy : {precision}, recall : {recall}, iou : {iou}, offset : {offset}\n')
        # print(f'name : {name}\n')
        # print(f'predicted_boxes : {predicted_boxes}\n')
        # print(f'ground_truth_boxes : {ground_truth_boxes}\n')
        # print('==================================\n')
        # # logger.info('Epoch: %d, accuracy: %d, recall: %.4f', step, accuracy, recall)/
        # # embed()
    # return precision, recall, iou, offset
    # return print('done')

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

def boxes_overlap(box1, box2):
    x1, y1, z1, x2, y2, z2 = [np.float16(x) for x in box1[1:]]
    a1, b1, c1, a2, b2, c2 = [np.float16(x) for x in box2[1:]]
    return not (x2 < a1 or a2 < x1 or y2 < b1 or b2 < y1 or z2 < c1 or c2 < z1)

def box_area(box):
    _, x1, y1, z1, x2, y2, z2 = box
    return (x2 - x1) * (y2 - y1) * (z2 - z1)


from PIL import Image
def draw_boxes_on_nii(name, ground_truth_boxes, predicted_boxes):
    # 将nii图像转换为numpy数组
    nii_image = tio.ScalarImage(f'D:\Work_file\det\data_seg_crop\{name}_croplung.nii.gz')
    nii_image = nii_image.data.squeeze(0)
    nii_data = np.array(nii_image)
    nii_data_p = np.array(nii_image)

    # 创建一个空白的四维数组
    color_image = np.zeros((384, 236, 216, 3))
    # 将灰度图像复制到新数组的每个颜色通道中
    color_image[..., 0] = nii_data
    color_image[..., 1] = nii_data
    color_image[..., 2] = nii_data

    # 创建一个空白的四维数组
    color_image_p = np.zeros((384, 236, 216, 3))
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
        x1, y1, z1, x2, y2, z2 = map(int, box_p)
        
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
        x1, y1, z1, x2, y2, z2 = map(int, box_p)
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

    model_path = 'D:\Work_file\det\save\\best_model_res101_resize512512256_0718-30.pt'
    file_path = 'D:\\Work_file\\det\\csv_file\\test_names_server.csv'
    names = read_names_from_csv(file_path)
    # embed()
    precisions, recalls, ious, offsets = [], [], [], []
    for name in names:
        # embed()
        # name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.935683764293840351008008793409'
        print(name)
        test_luna16(name, model_path)
        # precision, recall, iou, offset = test_luna16(name, model_path)
        # precisions.append(precision)
        # recalls.append(recall)
        # ious.append(iou)
        # offsets.append(offset)
        # embed()
    # print('the finish result\n')
    # print(f'the average pre is : {np.mean(precisions)}, recall is : {np.mean(recalls)}, the iou is : {np.mean(ious)}, the offset is : {np.mean(offsets)}\n')
    # print(f'the pre is {precisions}\nthe recalls is : {recalls}, \nthe ious is : {ious}, \nthe offsets is :{offsets}')


    # name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.935683764293840351008008793409'
    # ground_truth_boxes = [[286.5794461671875, 214.41151872433034, 100.7641956783854, 290.37465257566964, 218.2067251328125, 105.19193648828123], [88.4696189341518, 203.8544790198661, 112.82729978984378, 91.2676157229911, 206.65247580870536, 116.09162937682294]]
    # predicted_boxes = [[(87.3351), (205.8732), (114.8675), (90.2318), (208.3230), (117.7344)], [(287.7888), (216.3213), (100.9687), (291.9958), (219.9482), (105.6656)]]
    # draw_boxes_on_nii(name, ground_truth_boxes, predicted_boxes)