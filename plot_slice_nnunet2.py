# 这个block是从json文件中读取bboxes的信息，然后去进行bbox的一些plot
# 先可以是整个nii文件的bbox的plot
# 然后再是一张slice的plot

#*  这个block是从json文件中读取bboxes的信息，然后去进行bbox的一些plot
import json

import matplotlib.pyplot as plt
import numpy as np
import torchio as tio
import os


def get_classboxes(json_path, number):
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    results = data['results']  # 这将是一个列表，包含所有迭代的数据

    # 遍历 results 列表
    for item in results:
        # 访问每个项目的特定字段
        name = item['name']
        if name == number:
            gt_box = item['GT_boxes']
            FN_box = item['GT_FN']
            fp_box = item['pred_FP']
            tp_box = item['pred_TP']
            pred_box = item['pred_boxes']

    if len(gt_box) == 0:
        print(f'gt box is 0!!!!!!!!!!!!!!! maybe wrong with the number name!!!!!!!!!!!!')
    return gt_box, pred_box, FN_box, fp_box, tp_box




def process_nii_image(number, bboxes, output_path):
    # 加载nii图像
    img = tio.ScalarImage(f'/public_bme/data/xiongjl/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset501_LymphNodes/testingTr/lymph_{number}_0000.nii.gz')
    data = img.data[0, :, :, :]
    data[data > 240] = 240
    data[data < -160] = -160
    # print(bboxes)
    # 遍历每个框
    # bboxes = cxcyczwhd2x1y1z1x2y2z2(bboxes)
    for bbox in bboxes:
        
        x1, y1, z1, x2, y2, z2 = map(int, bbox)

        data[x1:x2+1, y1:y1+1, z1:z2+1] = 250
        data[x1:x2+1, y2:y2+1, z1:z2+1] = 250
        data[x1:x1+1, y1:y2+1, z1:z2+1] = 250
        data[x2:x2+1, y1:y2+1, z1:z2+1] = 250
        data[x1:x2+1, y1:y2+1, z1:z1+1] = 250
        data[x1:x2+1, y1:y2+1, z2:z2+1] = 250

    
    # 保存结果
    # affine =  np.diag([-0.7, -0.7, 0.7, 1.])
    # affine = np.array([[-0.7, 0, 0, 0], [0, -0.7, 0, 0], [0, 0, 0.7, 0], [0, 0, 0, 1]])
    # resample = tio.Resample(0.7)
    # resampled_img = resample(img)
    # print(resampled_img.affine)
    new_img = tio.ScalarImage(tensor=data.unsqueeze(0), affine=img.affine)
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
        # print('cyan')
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
        # print('red')
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
        # print('green')
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
    # 显示图像前保存画布
    fig.savefig(f'/public_bme/data/xiongjl/nnUNet/nnUNetFrame/DATASET/png_results/{number}_{point}.png')  # 指定保存路径和文件名
    # plt.show()


# 示例用法
if __name__ == '__main__':
    #* 这里的iou confi在生成json文件的时候就已经指定了
    #* 这里因为是分割的结果，所以也没有peak confi的指定

    number = '020'
    filename = "/public_bme/data/xiongjl/lymph_nodes/all_whole_testing/202004020032.nii.gz"
    json_path = '/public_bme/data/xiongjl/nnUNet/nnUNetFrame/DATASET/json_results/3d_fullres_4.json'
    output_path = '/public_bme/data/xiongjl/lymph_det/nii_temp/202004020032_020.nii'
    # image = tio.ScalarImage(filename)
    # array_3d = np.array(image.data[0, :, :, :])

    # * 开始分框
    gt_box, pred_box, FN_box, fp_box, tp_box = get_classboxes(json_path, number)
    print(f'the fn is {FN_box}\nthe fp is {fp_box}\nthe tp is {tp_box}')
    array_3d = process_nii_image(number, pred_box, output_path)
    print(f'array_3d shape is {array_3d.shape}')

    point = (252, 261, 232)  # 示例点

    plot_planes_with_boxes(array_3d, FN_box, fp_box, tp_box, point)
