import os
from IPython import embed
import torchio as tio
from tqdm import tqdm
from utils import *




def get_min_shape_with_torchio(files):
    min_shape = float('inf')
    min_name = None
    for file in tqdm(files):
        name = file['name']
        path = file['path']
        image = tio.ScalarImage(path)
        shape = image.shape[1:]
        if min(shape) < 128:
            print(min(shape))
            # print(name)
            min_shape = min(shape)
            min_name = name
    return min_shape, min_name


def get_max_shape_with_torchio(files):
    max_shape = 1.0
    max_name = None
    for file in tqdm(files):
        name = file['name']
        path = file['path']
        image = tio.ScalarImage(path)
        shape = image.shape[1:]
        if max(shape) > max_shape:
            # print(max(shape))
            # print(name)
            max_shape = max(shape)
            max_name = name
    return max_shape, max_name



import matplotlib.pyplot as plt
def get_shape_with_torchio(files):
    # max_shape = 1.0
    shape_x = []    # max_name = None
    shape_y = []
    shape_z = []
    for file in tqdm(files):
        name = file['name']
        path = file['path']
        image = tio.ScalarImage(path)
        shape = image.shape[1:]
        shape_x.append(shape[0])
        shape_y.append(shape[1])
        shape_z.append(shape[2])

    fig, axs = plt.subplots(3)
    axs[0].plot(shape_x)
    axs[0].set_title('List 1')
    axs[1].plot(shape_y)
    axs[1].set_title('List 2')
    axs[2].plot(shape_z)
    axs[2].set_title('List 3')
    plt.show()

    plot_list_counts(shape_z)
    # return max_shape, max_name

import matplotlib.pyplot as plt
from collections import Counter


def plot_list_counts(lst):
    # counts = Counter(lst)
    # plt.bar(counts.keys(), counts.values())
    plt.hist(lst, bins=len(set(lst)))
    plt.show()



def get_spacing_with_torchio(files):
    # max_shape = 1.0
    spacing_x = []    # max_name = None
    spacing_y = []
    spacing_z = []
    for file in tqdm(files):
        name = file['name']
        path = file['path']
        image = tio.ScalarImage(path)
        spacing = image.spacing[:]
        spacing_x.append(spacing[0])
        spacing_y.append(spacing[1])
        spacing_z.append(spacing[2])

    fig, axs = plt.subplots(3)
    axs[0].scatter(range(len(spacing_x)), spacing_x)
    axs[0].set_title('List 1')
    axs[1].scatter(range(len(spacing_y)), spacing_y)
    axs[1].set_title('List 2')
    axs[2].scatter(range(len(spacing_z)), spacing_z)
    axs[2].set_title('List 3')
    plt.show()
    plot_list_counts(spacing_x)
    plot_list_counts(spacing_y)
    plot_list_counts(spacing_z)
    # return max_shape, max_name


def get_slices_with_torchio(files):
    # max_shape = 1.0
    slices = []
    for file in tqdm(files):
        name = file['name']
        path = file['path']
        image = tio.ScalarImage(path)
        shape = image.shape[3]
        spacing = image.spacing[2]
        slice = shape / spacing
        slices.append(slice)

    # fig, axs = plt.subplots(1)
    plt.scatter(range(len(slices)), slices)
    plt.title('List 1')
    plt.show()
    plot_list_counts(slices)

    # return max_shape, max_name



def resample_image(files, new_spacing, plot=None, save=None):

    shape_x = []    # max_name = None
    shape_y = []
    shape_z = []
    for file in tqdm(files[0: 300]):
        # embed()
        name = file['name']
        path = file['path']
        # print(name)
        image = tio.ScalarImage(path)
        # print(image.spacing)
        resample = tio.Resample(new_spacing)
        resampled_image = resample(image)
        embed()
        if save != None:
            npy2nii(name, resampled_image.data, suffix='resampled', resample=True, affine=resampled_image.affine)
        shape = resampled_image.shape[1:]
        shape_x.append(shape[0])
        shape_y.append(shape[1])
        shape_z.append(shape[2])

    if plot != None:
        fig, axs = plt.subplots(3)
        axs[0].scatter(range(len(shape_x)), shape_x)
        axs[0].set_title('resample shape x')
        axs[1].scatter(range(len(shape_y)), shape_y)
        axs[1].set_title('resample shape y')
        axs[2].scatter(range(len(shape_z)), shape_z)
        axs[2].set_title('resample shape z')
        plt.show()
        plot_list_counts(shape_x)
        plot_list_counts(shape_y)
        plot_list_counts(shape_z)
    return print('resample list done')

def resample_image_shape(files, new_spacing, plot=None):

    shape_x = [] 
    shape_y = []
    shape_z = []
    for file in tqdm(files):
        # embed()
        name = file['name']
        path = file['path']
        # print(name)
        image = tio.ScalarImage(path)
        # print(image.spacing)
        new_shape = (image.shape[1:] * np.array(image.spacing)) / np.array(new_spacing)

        # shape = resampled_image.shape[1:]
        shape_x.append(new_shape[0])
        shape_y.append(new_shape[1])
        shape_z.append(new_shape[2])

    if plot != None:
        fig, axs = plt.subplots(3)
        axs[0].scatter(range(len(shape_x)), shape_x)
        axs[0].set_title('resample shape x')
        axs[1].scatter(range(len(shape_y)), shape_y)
        axs[1].set_title('resample shape y')
        axs[2].scatter(range(len(shape_z)), shape_z)
        axs[2].set_title('resample shape z')
        plt.show()
        plot_list_counts(shape_x)
        plot_list_counts(shape_y)
        plot_list_counts(shape_z)
    return print('resample list done')

import matplotlib.pyplot as plt
from collections import Counter

def add_mhd_path_to_csv(mhd_list):

    csv_file_dir = 'D:\\Work_file\\det_LUNA16_data\\annotations.csv'
    new_csv_file = 'D:\\Work_file\\det_LUNA16_data\\annotations_withpath.csv'
    name_to_path = {}
    for item in mhd_list:
        name_to_path[item['name']] = item['path']
    with open(csv_file_dir, 'r') as input_file, open(new_csv_file, 'w') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)
        for row in reader:
            name = row[0]
            if name in name_to_path:
                row.append(name_to_path[name])
            writer.writerow(row)
    return new_csv_file



def convert_world_to_voxel_coordinates():
    csv_file = 'D:\\Work_file\\det_LUNA16_data\\annotations_withpath.csv'
    new_csv_file = 'D:\\Work_file\\det_LUNA16_data\\annotations_pathcoord_test.csv'
    with open(csv_file, 'r') as input_file, open(new_csv_file, 'w') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        # 跳过标题行
        next(reader)

        for row in reader:
            mdh_name = row[0]
            # if mdh_name != '1.3.6.1.4.1.14519.5.2.1.6279.6001.964952370561266624992539111877': # ('R', 'A', 'S') not ok affine is (+,+,+)
            # if mdh_name != '1.3.6.1.4.1.14519.5.2.1.6279.6001.128023902651233986592378348912': #  ('L', 'P', 'S'), ok affine is (-,-,+)
            # if mdh_name != '1.3.6.1.4.1.14519.5.2.1.6279.6001.997611074084993415992563148335': #  ('L', 'P', 'S'), ok affine is (-,-,+)
            # if mdh_name != '1.3.6.1.4.1.14519.5.2.1.6279.6001.187694838527128312070807533473': #  ('L', 'P', 'S'), ok affine is (-,-,+)
            # if mdh_name != '1.3.6.1.4.1.14519.5.2.1.6279.6001.126264578931778258890371755354':  #  ('L', 'P', 'S'), ok affine is (-,-,+)
            if mdh_name != '1.3.6.1.4.1.14519.5.2.1.6279.6001.123697637451437522065941162930': #  ('L', 'P', 'S'), ok affine is (-,-,+)
                continue
            mhd_path = row[5]
            x_world = float(row[1])
            y_world = float(row[2])
            z_world = float(row[3])
            diameter_mm = float(row[4])
            image = tio.ScalarImage(mhd_path)
            world_coordinates = [x_world, y_world, z_world]
            # embed()
            # print(image.orientation)
            if image.orientation == ('R', 'A', 'S'):
                # embed()
                voxel_coordinates = (np.array(world_coordinates) - image.origin) / image.spacing
            else:
                voxel_coordinates = (np.array(world_coordinates) - image.origin * np.array((-1, -1, 1))) / image.spacing

            # image_trans = tio.ScalarImage(tensor=image.data, affine=np.array(image.affine) * np.array([[-1, 1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]]))
            npy2nii(mdh_name, image.data, suffix='another')

            print(world_coordinates)
            print('=================')
            print(voxel_coordinates)

            embed()
            # new_row = [mdh_name] + [mhd_path] + list(voxel_coordinates) + [diameter_mm]
            # writer.writerow(new_row)
    return print('trans done')



def npy2nii_name(name):
    csv_dir = 'D:\\Work_file\\det_LUNA16_data\\annotations_pathcoord.csv'
    df = pd.read_csv(csv_dir)
    df = df[df['seriesuid'] == name]
    mhd_path = str(df[['path']].values[0])[2:-2]
    image = tio.ScalarImage(mhd_path)
    affine = image.affine

    image_npy_dir = 'D:\\Work_file\\det\\npy_data\\{}_image.npy'.format(name)
    image_npy = np.load(image_npy_dir)
    image_npy = torch.from_numpy(image_npy).unsqueeze(0)
    image_nii = tio.ScalarImage(tensor=image_npy, affine=affine)
    image_nii.save('./nii_temp/{}_image.nii'.format(name))

    whd_npy_dir = 'D:\\Work_file\\det\\npy_data\\{}_whd.npy'.format(name)
    whd_npy = np.load(whd_npy_dir)
    whd_npy = torch.from_numpy(whd_npy)
    whd_nii = tio.ScalarImage(tensor=whd_npy, affine=affine)
    whd_nii.save('./nii_temp/{}_whd.nii'.format(name))

    offset_npy_dir = 'D:\\Work_file\\det\\npy_data\\{}_offset.npy'.format(name)
    offset_npy = np.load(offset_npy_dir)
    offset_npy = torch.from_numpy(offset_npy)
    offset_nii = tio.ScalarImage(tensor=offset_npy, affine=affine)
    offset_nii.save('./nii_temp/{}_offset.nii'.format(name))

    mask_npy_dir = 'D:\\Work_file\\det\\npy_data\\{}_mask.npy'.format(name)
    mask_npy = np.load(mask_npy_dir)
    mask_npy = torch.from_numpy(mask_npy).unsqueeze(0)
    mask_nii = tio.ScalarImage(tensor=mask_npy, affine=affine)
    mask_nii.save('./nii_temp/{}_mask.nii'.format(name))

    hmap_npy_dir = 'D:\\Work_file\\det\\npy_data\\{}_hmap.npy'.format(name)
    hmap_npy = np.load(hmap_npy_dir)
    hmap_npy = torch.from_numpy(hmap_npy).unsqueeze(0)
    hmap_nii = tio.ScalarImage(tensor=hmap_npy, affine=affine)
    hmap_nii.save('./nii_temp/{}_hmap.nii'.format(name))

    return print('save done')




def npy2nii_forsee(name, image_crop, hmap_crop, whd_crop, offset_crop, mask_crop):
    csv_dir = 'D:\\Work_file\\det_LUNA16_data\\annotations_pathcoord.csv'
    df = pd.read_csv(csv_dir)
    df = df[df['seriesuid'] == name]
    mhd_path = str(df[['path']].values[0])[2:-2]
    image = tio.ScalarImage(mhd_path)
    affine = image.affine

    image_nii = tio.ScalarImage(tensor=image_crop.unsqueeze(0), affine=affine)
    image_nii.save('./nii_temp/{}_image_crop.nii'.format(name))

    whd_nii = tio.ScalarImage(tensor=whd_crop, affine=affine)
    whd_nii.save('./nii_temp/{}_whd_crop.nii'.format(name))

    offset_nii = tio.ScalarImage(tensor=offset_crop, affine=affine)
    offset_nii.save('./nii_temp/{}_offset_crop.nii'.format(name))

    mask_nii = tio.ScalarImage(tensor=mask_crop.unsqueeze(0), affine=affine)
    mask_nii.save('./nii_temp/{}_mask_crop.nii'.format(name))

    hmap_nii = tio.ScalarImage(tensor=hmap_crop.unsqueeze(0), affine=affine)
    hmap_nii.save('./nii_temp/{}_hmap_crop.nii'.format(name))

    return print('save done')



def process_file(file_path):
    # 读取csv文件
    df = pd.read_csv(file_path)

    # 定义一个函数来检查mhd文件的形状
    def check_shape(file_path):
        # 使用torchio读取mhd文件
        image = tio.ScalarImage(file_path)
        # if image.shape[3] < 128 or image.shape[1] < 128 or image.shape[2] < 128:
        if image.orientation == ('R', 'A', 'S'):
            return False
        else:
            return True

    # 应用函数并删除不符合条件的行
    df = df[df['path'].apply(lambda x: check_shape(x))]

    # 保存新的csv文件
    new_file_path =  'D:\\Work_file\\det_LUNA16_data\\annotations_pathcoord_noras.csv'
    df.to_csv(new_file_path, index=False)
    return new_file_path




if __name__ == '__main__':

    # convert_world_to_voxel_coordinates()









    # mhd_files = get_mhd_files('D:\Work_file\det_LUNA16_data')
    # example_path = mhd_files[0]['path']
    # image_data = read_mhd_file_with_torchio(example_path)
    # max_shape, max_name = get_max_shape_with_torchio(mhd_files)
    # get_spacing_with_torchio(mhd_files)
    # new_spacing = (0.7, 0.7, 1.2)
    # resample_image(mhd_files, new_spacing)
    # data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    # plot_list_counts(data)
    # resample_image(mhd_files, new_spacing)
    # new_csv_file = add_mhd_path_to_csv(mhd_files)
    # name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.997611074084993415992563148335'
    # new_csv_file = convert_world_to_voxel_coordinates()
    # result = find_name_in_csv(name)
    # print(max_shape, max_name)
    # npy2nii(name)
    # embed()
    # print(mhd_files)

    # 调用函数并传入文件路径
    new_file = process_file('D:\\Work_file\\det_LUNA16_data\\annotations_pathcoord_less128.csv')
    print(f'新文件已保存为: {new_file}')


