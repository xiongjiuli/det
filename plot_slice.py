import matplotlib.pyplot as plt
import numpy as np
import torchio as tio
import os


def get_gtboxes(folder_path, filename):
    result = []
    if filename.endswith('.txt'):
        with open(os.path.join(folder_path, filename), 'r') as f:
            for line in f:
                data = line.strip().split()
                x1, y1, z1, x2, y2, z2 = map(float, data[1:7])
                result.append([x1, y1, z1, x2, y2, z2])
    return result


def get_predboxes(folder_path, filename, peak_confi):
    result = []
    if filename.endswith('.txt'):   
        txt_path = os.path.join(folder_path, filename)
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    confi = float(data[1])
                    if confi >= peak_confi:
                        x1, y1, z1, x2, y2, z2 = map(float, data[2:8])
                        result.append([x1, y1, z1, x2, y2, z2])
        else:
            result.append([0., 0., 0., 0., 0., 0.])
            print(f'in the {filename} no pred bbox!')
    return result



def process_nii_image(image_path, bboxes, output_path):
    # 加载nii图像
    img = tio.ScalarImage(image_path)
    data = img.data[0, :, :, :]
    # 遍历每个框
    # bboxes = cxcyczwhd2x1y1z1x2y2z2(bboxes)
    for bbox in bboxes:
        
        x1, y1, z1, x2, y2, z2 = map(int, bbox)
        # print(x1, y1, z1, x2, y2, z2)
        # 将框内的像素值设置为50
        # 将框的边线上的像素值设置为50
        data[x1:x2+1, y1:y1+1, z1:z2+1] = 1.1
        data[x1:x2+1, y2:y2+1, z1:z2+1] = 1.1
        data[x1:x1+1, y1:y2+1, z1:z2+1] = 1.1
        data[x2:x2+1, y1:y2+1, z1:z2+1] = 1.1
        data[x1:x2+1, y1:y2+1, z1:z1+1] = 1.1
        data[x1:x2+1, y1:y2+1, z2:z2+1] = 1.1
        # print(f'data max is {data.max()}')
    
    # 保存结果
    # affine =  np.diag([-0.7, -0.7, 0.7, 1.])
    affine = np.array([[-0.7, 0, 0, 0], [0, -0.7, 0, 0], [0, 0, 0.7, 0], [0, 0, 0, 1]])
    # resample = tio.Resample(0.7)
    # resampled_img = resample(img)
    # print(resampled_img.affine)
    new_img = tio.ScalarImage(tensor=data.unsqueeze(0), affine=affine)
    new_img.save(output_path)
    return data




def plot_planes_with_boxes(array_3d, no_predbox_FN, fp, tp, point):
    x0, y0, z0 = point

    # 计算平面
    plane_xy = array_3d[:, :, z0]
    plane_xz = array_3d[:, y0, :]
    plane_zy = array_3d[x0, :, :]

    # 创建画布
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 绘制 XY 平面
    axes[0].imshow(plane_xy.T, cmap='gray', origin='lower')
    axes[0].invert_yaxis()  # 反转 Y 轴
    axes[0].set_title('XY Plane')

    # 绘制 XZ 平面
    axes[1].imshow(plane_xz.T, cmap='gray', origin='lower')
    axes[1].set_title('XZ Plane')

    # 绘制 ZY 平面
    axes[2].imshow(plane_zy.T, cmap='gray', origin='lower')
    axes[2].set_title('ZY Plane')


    # 识别并绘制框
    for box in no_predbox_FN:
        x1, y1, z1, x2, y2, z2 = box
        # 绘制框于 XY 平面
        if z1 <= z0 <= z2:
            rect_xy = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='cyan')
            axes[0].add_patch(rect_xy)

        # 绘制框于 XZ 平面
        if y1 <= y0 <= y2:
            rect_xz = plt.Rectangle((x1, z1), x2 - x1, z2 - z1, fill=False, edgecolor='cyan')
            axes[1].add_patch(rect_xz)

        # 绘制框于 ZY 平面
        if x1 <= x0 <= x2:
            rect_zy = plt.Rectangle((y1, z1),  y2 - y1, z2 - z1, fill=False, edgecolor='cyan')
            axes[2].add_patch(rect_zy)

    # 识别并绘制框
    for box in fp:
        x1, y1, z1, x2, y2, z2 = box
        # 绘制框于 XY 平面
        if z1 <= z0 <= z2:
            rect_xy = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red')
            axes[0].add_patch(rect_xy)

        # 绘制框于 XZ 平面
        if y1 <= y0 <= y2:
            rect_xz = plt.Rectangle((x1, z1), x2 - x1, z2 - z1, fill=False, edgecolor='red')
            axes[1].add_patch(rect_xz)

        # 绘制框于 ZY 平面
        if x1 <= x0 <= x2:
            rect_zy = plt.Rectangle((y1, z1),  y2 - y1, z2 - z1, fill=False, edgecolor='red')
            axes[2].add_patch(rect_zy)

    # 识别并绘制框
    for box in tp:
        x1, y1, z1, x2, y2, z2 = box
        # 绘制框于 XY 平面
        if z1 <= z0 <= z2:
            rect_xy = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='green')
            axes[0].add_patch(rect_xy)

        # 绘制框于 XZ 平面
        if y1 <= y0 <= y2:
            rect_xz = plt.Rectangle((x1, z1), x2 - x1, z2 - z1, fill=False, edgecolor='green')
            axes[1].add_patch(rect_xz)

        # 绘制框于 ZY 平面
        if x1 <= x0 <= x2:
            rect_zy = plt.Rectangle((y1, z1),  y2 - y1, z2 - z1, fill=False, edgecolor='green')
            axes[2].add_patch(rect_zy)

    plt.show()



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
    
# boxA = (Ax1,Ay1,Ax2,Ay2)
# boxB = (Bx1,By1,Bx2,By2)
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



def class_box(pred_boxes, gt_boxes, iou_confi):

    no_predbox_FN = filter_boxes(gt_boxes, pred_boxes, iou_confi=iou_confi)  # 得到没有被预测出来的gt_box FN #!这个地方应该再加上iou的一些设置
    fp = get_fp(gt_boxes, pred_boxes, iou_confi=iou_confi) # 多检测出来的结节 FP
    tp = [box for box in pred_boxes if box not in fp] # 检测出来的结节 tp

    return no_predbox_FN, fp, tp


# 示例用法
if __name__ == '__main__':

    filename = "/public_bme/data/xiongjl/lymph_nodes/all_whole_testing/202004020032.nii.gz"
    pred_folder_path = '/public_bme/data/xiongjl/lymph_det/plot/bbox_txt/atten_unet/atten_unet_48_1120211850/'
    gt_folder_path = '/public_bme/data/xiongjl/lymph_det/plot/bbox_txt/gt/whole_groundtruths'
    peak_confi = 0.25
    iou_confi = 0.01
    name = '202004020032.txt'
    output_path = '/public_bme/data/xiongjl/lymph_det/nii_temp/202004020032_pred.nii'


    pred_boxes = get_predboxes(pred_folder_path, name, peak_confi)
    gt_boxes = get_gtboxes(gt_folder_path, name)
    # * 开始分框
    no_predbox_FN, fp, tp = class_box(pred_boxes, gt_boxes, iou_confi)
    #* 得到nii图像
    image = tio.ScalarImage(filename)
    array_3d = np.array(image.data[0, :, :, :])
    # array_3d = process_nii_image(filename, pred_boxes, output_path)
    point = (344, 306, 298)  # 示例点

    plot_planes_with_boxes(array_3d, no_predbox_FN, fp, tp, point)