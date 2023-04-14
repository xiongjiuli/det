import torch
import torchio 
import sys
import os
from IPython import embed
import torch.utils.data as data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from preprocess import npy2nii_forsee
from time import time 


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
        # time_1 = time()
        dict = resize_data(name, self.root_dir, new_shape=(512, 512, 256))
        # print('image_crop : {}'.format(time() - time_1))

        return dict
    

    def __len__(self):

        return len(self.names)
        
    def setup(self):
        print('set up ')
        self.names = []
        file_path = 'D:\Work_file\det_LUNA16_data\\AT_afterlungcrop.csv'
        names = read_names_from_csv(file_path)  
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
    
    dataset = luna16Dataset(mode='train')
    sample = dataset[0]
    # embed()
