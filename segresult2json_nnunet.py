import numpy as np
import torchio as tio
from skimage import measure
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json



def get_boxes(number, _type):
    if _type == 'gt':
        file_path = f'/public_bme/data/xiongjl/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset501_LymphNodes/testlabelTr/lymph_{number}.nii.gz'
    elif _type == 'pred':
        file_path = f"/public_bme/data/xiongjl/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset501_LymphNodes/resultTr_3d_fullrs_4/lymph_{number}.nii.gz"
    else:
        print(f'the type name is wrong, now is {_type}')
    label_nii = tio.ScalarImage(file_path).numpy()[0, :, :, :]

    # get the 连通域
    pre_label = measure.label(label_nii)
    pre_region = measure.regionprops(pre_label)

    bboxes = []
    for i in range(len(pre_region)):
        bboxes.append(pre_region[i].bbox)
    return bboxes


def filter_boxes(gt, pred, iou_confi):
    result = []
    for box in gt:
        overlap = False
        for p_box in pred:
            IoU = iou(box, p_box)
            if IoU >= iou_confi:
                overlap = True
                break
        if not overlap:
            result.append(box)

    return result



def get_fp(gt, pred, iou_confi):
    result = []
    for box in pred:
        overlap = False
        for p_box in gt:
            IoU = iou(box, p_box)
            if IoU >= iou_confi:
                overlap = True
                break
        if not overlap:
            result.append(box)

    return result



def iou(boxA, boxB):
    # if boxes dont intersect
    if _boxesIntersect(boxA, boxB) is False:
        return 0
    interArea = _getIntersectionArea(boxA, boxB)
    union = _getUnionAreas(boxA, boxB, interArea=interArea)
    # intersection over union
    iou = interArea / union
    if iou < 0:
        iou = - iou
        print('the iou < 0, and i do the iou = - iou')
    # print(f'the iou is {iou}, the interArea is {interArea}, the union is {union}')
    assert iou >= 0
    return iou


def _boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[3]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[3]:
        return False  # boxA is left of boxB
    if boxA[2] > boxB[5]:
        return False  # boxA is left of boxB
    if boxB[2] > boxA[5]:
        return False  # boxA is left of boxB
    if boxA[4] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[4]:
        return False  # boxA is below boxB
    return True
def _getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    zA = max(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[4], boxB[4])
    zB = min(boxA[5], boxB[5])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1) * (zB - zA + 1)

def _getUnionAreas(boxA, boxB, interArea=None):
    # print(f'the boxa is {boxA}, the boxb is {boxB}')
    area_A = _getArea(boxA)
    area_B = _getArea(boxB)
    # print(f'the areaa is {area_A}, the areab is {area_B}')
    if interArea is None:
        interArea = _getIntersectionArea(boxA, boxB)
        # print(f'the interarea is None, the interarea is {interArea}')
    # print(f'the interarea is None, the interarea is {interArea}')
    # print(f'the iou is {area_A + area_B - interArea}')
    return float(area_A + area_B - interArea)

def _getArea(box):
    return (box[3] - box[0] + 1) * (box[4] - box[1] + 1) * (box[5] - box[2] + 1)





def get_result(iou_confi, json_name):

    names = []
    name_file = os.listdir('/public_bme/data/xiongjl/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset501_LymphNodes/testlabelTr/')
    for name in name_file:
        if name.endswith('.nii.gz'):
            names.append(name.split('_')[1].split(".")[0])

    FP_ls = []
    FN_ls = []
    TP_ls = []

    json_path = f'/public_bme/data/xiongjl/nnUNet/nnUNetFrame/DATASET/json_results/{json_name}'

    # 读取已有数据（如果有的话）
    # with open(json_path, 'r') as json_file:
    #     try:
    #         existing_data = json.load(json_file)
    #     except json.JSONDecodeError:  # 如果文件为空或格式不正确
    #         existing_data = []


    all_data = []
    for number in tqdm(names):
        # print(f'number is {number}')
        ground_truth_boxes = get_boxes(number, _type='gt')  # 这个就是提取gt
        pred_bboxes = get_boxes(number, _type='pred')  # 这个就是根据confi来提取大于这个confi的bbox

        no_predbox_FN = filter_boxes(ground_truth_boxes, pred_bboxes, iou_confi=iou_confi)  # 得到没有被预测出来的gt_box FN #!这个地方应该再加上iou的一些设置
        FN_ls.append(len(no_predbox_FN))

        fp = get_fp(ground_truth_boxes, pred_bboxes, iou_confi=iou_confi) # 多检测出来的结节 FP
        FP_ls.append(len(fp))

        TP_ls.append(len(ground_truth_boxes) - len(no_predbox_FN)) # 测出来的结节 TP
        tp = [box for box in pred_bboxes if box not in fp]
        
        #* 开始计算一些结果
        accuracy_ = ((len(ground_truth_boxes) - len(no_predbox_FN))/((len(ground_truth_boxes) - len(no_predbox_FN)) + (len(fp)) + (len(no_predbox_FN))))
        if ((len(ground_truth_boxes) - len(no_predbox_FN)) + (len(fp))) == 0:
            precision_ = 0.
        else:
            precision_ = ((len(ground_truth_boxes) - len(no_predbox_FN))/((len(ground_truth_boxes) - len(no_predbox_FN)) + (len(fp))))
        recall_ = ((len(ground_truth_boxes) - len(no_predbox_FN))/((len(ground_truth_boxes) - len(no_predbox_FN)) + (len(no_predbox_FN))))
        f1_ = ((2 * (len(ground_truth_boxes) - len(no_predbox_FN))) / (2 * (len(ground_truth_boxes) - len(no_predbox_FN)) + (len(fp)) + (len(no_predbox_FN))))

        data = {
            'name' : number,
            'recall': recall_,
            'accuracy': accuracy_,
            'precision': precision_,
            'f1': f1_,
            'FP': len(fp),
            'GT_boxes_length': len(ground_truth_boxes),
            'pred_boxes_length': len(pred_bboxes),
            'GT_FN_length': len(no_predbox_FN),
            'pred_FP_length': len(fp),
            'pred_TP_length': len(ground_truth_boxes) - len(no_predbox_FN),
            'GT_boxes': ground_truth_boxes,
            'GT_FN': no_predbox_FN,
            'pred_FP': fp,
            'pred_TP': tp ,
        }
        all_data.append(data)

        # 将新数据添加到现有数据中
        # existing_data.append(data)

        # # 更新文件
        # with open(json_path, 'a') as json_file:
        #     json.dump(data, json_file, indent=4)


    fp_point = np.mean(FP_ls)
    fn_point = np.mean(FN_ls)
    tp_point = np.mean(TP_ls)

    accuracy = (tp_point/(tp_point + fp_point + fn_point))
    if (tp_point + fp_point) == 0:
        precision = (0)
    else:
        precision = (tp_point/(tp_point + fp_point))
    recall = (tp_point/(tp_point + fn_point))
    f1 = ((2 * tp_point) / (2 * tp_point + fp_point + fn_point))

    print(f'recall is {recall}')
    print(f'precision is {precision}')
    print(f'accuracy is {accuracy}')
    print(f'f1 is {f1}')
    print(f'fp is {fp_point}')
    summary = {
        'all_recall': recall,
        'all_accuracy': accuracy,
        'all_precision': precision,
        'all_f1': f1,
        'all_FP': fp_point
    }

    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump({'results': all_data, 'summary': summary}, json_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':

    get_result(0.01, json_name='3d_fullres_4.json')
