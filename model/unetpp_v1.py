from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

# class VGGBlock(nn.Module):
#     def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
#         super(VGGBlock, self).__init__()
#         self.act_func = act_func
#         self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(middle_channels)
#         self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.act_func(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.act_func(out)
#         return out

class NestedUNet(nn.Module):
    def __init__(self, args,in_channel,out_channel):
        super().__init__()

        self.args = args

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool3d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv0_0 = DoubleConv(in_channel, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv0_1 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2]*2+nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1]*3+nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0]*4+nb_filter[1], nb_filter[0])
        self.sigmoid = nn.Sigmoid()
        self.head = Head(num_classes=1)
        # if self.args.deepsupervision:
        if self.args:
            self.final1 = nn.Conv3d(nb_filter[0], out_channel, kernel_size=1)
            self.final2 = nn.Conv3d(nb_filter[0], out_channel, kernel_size=1)
            self.final3 = nn.Conv3d(nb_filter[0], out_channel, kernel_size=1)
            self.final4 = nn.Conv3d(nb_filter[0], out_channel, kernel_size=1)
        else:
            self.final = nn.Conv3d(nb_filter[0], out_channel, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.args:
            output1 = self.final1(x0_1)
            output1 = self.sigmoid(output1)
            output2 = self.final2(x0_2)
            output2 = self.sigmoid(output2)
            output3 = self.final3(x0_3)
            output3 = self.sigmoid(output3)
            output4 = self.final4(x0_4)
            output4 = self.sigmoid(output4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            output = self.sigmoid(output)
            return self.head(output)

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
    args = False
    model = NestedUNet(False, in_channel=1, out_channel=64)
    inputs = inputs.cuda()
    model.cuda()
    x = model(inputs)
    print(len(x))
    # print(x.shape)
    print(model)
    print(x[0].shape)
    print(x[1].shape)
    print(x[2].shape)