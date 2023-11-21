# _*_ coding: utf-8 _*_
# Author: Jielong
# @Time: 21/08/2019 15:52
import sys
import time
import torch
import torch.nn as nn
# from unet3d_model.building_components import EncoderBlock, DecoderBlock
sys.path.append("..")
# _*_ coding: utf-8 _*_
# Author: Jielong
# @Time: 21/08/2019 15:52
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                                stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

    def forward(self, x):
        x = self.batch_norm(self.conv3d(x))
        # x = self.conv3d(x)
        x = F.elu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, model_depth=4, pool_size=2):
        super(EncoderBlock, self).__init__()
        self.root_feat_maps = 16
        self.num_conv_blocks = 2
        # self.module_list = nn.ModuleList()
        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.root_feat_maps
            for i in range(self.num_conv_blocks):
                # print("depth {}, conv {}".format(depth, i))
                if depth == 0:
                    # print(in_channels, feat_map_channels)
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
                else:
                    # print(in_channels, feat_map_channels)
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
            if depth == model_depth - 1:
                break
            else:
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling

    def forward(self, x):
        down_sampling_features = []
        for k, op in self.module_dict.items():
            if k.startswith("conv"):
                x = op(x)
                # print(k, x.shape)
                if k.endswith("1"):
                    down_sampling_features.append(x)
            elif k.startswith("max_pooling"):
                x = op(x)
                # print(k, x.shape)

        return x, down_sampling_features


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=2, padding=1, output_padding=1):
        super(ConvTranspose, self).__init__()
        self.conv3d_transpose = nn.ConvTranspose3d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=k_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   output_padding=output_padding)

    def forward(self, x):
        return self.conv3d_transpose(x)


class DecoderBlock(nn.Module):
    def __init__(self, out_channels, model_depth=4):
        super(DecoderBlock, self).__init__()
        self.num_conv_blocks = 2
        self.num_feat_maps = 16
        # user nn.ModuleDict() to store ops
        self.module_dict = nn.ModuleDict()

        for depth in range(model_depth - 2, -1, -1):
            # print(depth)
            feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps
            # print(feat_map_channels * 4)
            self.deconv = ConvTranspose(in_channels=feat_map_channels * 4, out_channels=feat_map_channels * 4)
            self.module_dict["deconv_{}".format(depth)] = self.deconv
            for i in range(self.num_conv_blocks):
                if i == 0:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 6, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
                else:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
            if depth == 0:
                self.final_conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=out_channels)
                self.module_dict["final_conv"] = self.final_conv

    def forward(self, x, down_sampling_features):
        """
        :param x: inputs
        :param down_sampling_features: feature maps from encoder path
        :return: output
        """
        for k, op in self.module_dict.items():
            if k.startswith("deconv"):
                x = op(x)
                x = torch.cat((down_sampling_features[int(k[-1])], x), dim=1)
            elif k.startswith("conv"):
                x = op(x)
            else:
                x = op(x)
        return x


# if __name__ == "__main__":
#     # x has shape of (batch_size, channels, depth, height, width)
#     x_test = torch.randn(1, 1, 96, 96, 96)
#     x_test = x_test.cuda()
#     print("The shape of input: ", x_test.shape)

#     encoder = EncoderBlock(in_channels=1)
#     encoder.cuda()
#     print(encoder)
#     x_test, h = encoder(x_test)

#     db = DecoderBlock(out_channels=1)
#     db.cuda()
#     x_test = db(x_test, h)

class UnetModel(nn.Module):

    def __init__(self, in_channels, out_channels, model_depth=4, final_activation="sigmoid"):
        super(UnetModel, self).__init__()
        self.encoder = EncoderBlock(in_channels=in_channels, model_depth=model_depth)
        self.decoder = DecoderBlock(out_channels=out_channels, model_depth=model_depth)
        if final_activation == "sigmoid":
            self.sigmoid = nn.Sigmoid()
        else:
            self.softmax = nn.Softmax(dim=1)
        self.head = Head(num_classes=1)

    def forward(self, x):
        x, downsampling_features = self.encoder(x)
        x = self.decoder(x, downsampling_features)
        x = self.sigmoid(x)
        # print("Final output shape: ", x.shape)
        x = self.head(x)
        return x


# class Trainer(object):

#     def __init__(self, data_dir, net, optimizer, criterion, no_epochs, batch_size=8):
#         """
#         Parameter initialization
#         :param data_dir: folder that stores images for each modality
#         :param net: the created model
#         :param optimizer: the optimizer mode
#         :param criterion: loss function
#         :param no_epochs: number of epochs to train the model
#         :param batch_size: batch size for generating data during training
#         """
#         self.data_dir = data_dir
#         self.modalities = ["PET", "MASK"]
#         self.net = net
#         if torch.cuda.is_available():
#             self.net.cuda()
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.no_epochs = no_epochs
#         self.batch_size = batch_size

    # def train(self, data_paths_loader, dataset_loader, batch_data_loader):
    #     """
    #     Load corresponding data and start training
    #     :param data_paths_loader: get data paths ready for loading
    #     :param dataset_loader: get images and masks data
    #     :param batch_data_loader: generate batch data
    #     :return: None
    #     """
    #     # self.net.train()
    #     pet_paths = data_paths_loader(self.data_dir, self.modalities[0])
    #     print(pet_paths)
    #     mask_paths = data_paths_loader(self.data_dir, self.modalities[1])
    #     pets, masks = dataset_loader(pet_paths, mask_paths)
    #     training_steps = len(pets) // self.batch_size

    #     for epoch in range(self.no_epochs):
    #         start_time = time.time()
    #         train_losses, train_iou = 0, 0
    #         for step in range(training_steps):
    #             print("Training step {}".format(step))

    #             x_batch, y_batch = batch_data_loader(pets, masks, iter_step=step, batch_size=self.batch_size)
    #             x_batch = torch.from_numpy(x_batch).cuda()
    #             y_batch = torch.from_numpy(y_batch).cuda()

    #             self.optimizer.zero_grad()

    #             logits = self.net(x_batch)
    #             y_batch = y_batch.type(torch.int8)
    #             loss = self.criterion(logits, y_batch)
    #             loss.backward()
    #             self.optimizer.step()
    #             # train_iou += mean_iou(y_batch, logits)
    #             train_losses += loss.item()
    #         end_time = time.time()
    #         print("Epoch {}, training loss {:.4f}, time {:.2f}".format(epoch, train_losses / training_steps,
    #                                                                    end_time - start_time))

    # def predict(self):
    #     pass

    # def _save_checkpoint(self):
    #     pass


class Head(nn.Module):
    
    def __init__(
        self,
        num_classes,
        in_channels=64,
        inter_channels=64
    ):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            # nn.Conv3d(inter_channels, num_classes, kernel_size=3, stride=1, padding=1),  
            nn.Conv3d(inter_channels, num_classes, kernel_size=1, stride=1, padding=0)   
        )
        self.wh_head = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            # nn.Conv3d(inter_channels, 3, # !
            #           kernel_size=3, stride=1, padding=1))
            nn.Conv3d(inter_channels, 3, # !
                      kernel_size=1, stride=1, padding=0))

        self.reg_head = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            # nn.Conv3d(inter_channels, 3, # !
            #           kernel_size=3, stride=1, padding=1))
            nn.Conv3d(inter_channels, 3, # !
                      kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        hm = self.cls_head(x).sigmoid_()
        wh = self.wh_head(x)
        offset = self.reg_head(x)
        return hm, wh, offset





if __name__ == "__main__":
    inputs = torch.randn(1, 1, 160, 160, 160)
    # print("The shape of inputs: ", inputs.shape)
    # data_folder = "../processed"
    model = UnetModel(in_channels=1, out_channels=64)
    inputs = inputs.cuda()
    model.cuda()
    x = model(inputs)
    # print(model)
    print(x[0].shape)
    print(x[1].shape)
    print(x[2].shape)