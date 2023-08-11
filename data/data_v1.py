import torch
import torchio 
import sys
import os
from IPython import embed
import torch.utils.data as data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_v2 import *
# from utils import read_names_from_csv
from preprocess import npy2nii_forsee
from time import time 


def read_names_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过第一行
        names = [row[0] for row in reader]
    return list(set(names))

def get_filenames(path):
    filenames = []
    for filename in os.listdir(path):
        filenames.append(filename.split('_')[0])
        # embed()
    return filenames

class luna16Dataset(data.Dataset):

    def __init__(self, 
                 mode='train',
                 root_dir='D:\Work_file\det'
         
                 ):
  
        super(luna16Dataset, self).__init__()
        self.mode=mode
        self.root_dir=root_dir
        self.setup()

    def __getitem__(self, index):
     
        name = self.names[index]
      
        # name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.129567032250534530765928856531'
        # time_1 = time()
        dict = resize_data(name, self.root_dir, new_shape=(256, 256, 256))
        # print('image_crop : {}'.format(time() - time_1))
      
        return dict
       
    
    

    def __len__(self):

        return len(self.names[0:10])
        
    def setup(self):
        print('set up ')
        self.names = []
        # file_path = 'D:\Work_file\det_LUNA16_data\\AT_afterlungcrop.csv'
        # names = read_names_from_csv(file_path)  
        file_path = 'D:\Work_file\det\\nii_data_resample_seg_crop'
        names = get_filenames(file_path)
        random.seed(0)
        if self.mode == 'train':
            np.random.shuffle(names)    
        train_names = names[ : int(len(names) * 0.7)]
        valid_names = names[int(len(names) * 0.7) : int(len(names) * 0.8)]
        test_names = names[int(len(names) * 0.8) : ]

        if self.mode == 'valid':
            self.names = valid_names
        elif self.mode == 'test':
            self.names = test_names
        else:
            self.names = train_names
        # embed()

    

if __name__ == '__main__':
    
    dataset = luna16Dataset(mode='valid')
    sample = dataset
    # embed()
