import numpy as np
import torch
import torch.nn as nn


#-------------------------#
#   卷积+标准化+激活函数
#-------------------------#
class conv2d(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(conv2d, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

#-------------------------#
#   残差结构
#-------------------------#
class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)

def make_layer(k, inp_dim, out_dim, modules, **kwargs):
    layers = [residual(k, inp_dim, out_dim, **kwargs)]
    for _ in range(modules - 1):
        layers.append(residual(k, out_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

def make_hg_layer(k, inp_dim, out_dim, modules, **kwargs):
    layers  = [residual(k, inp_dim, out_dim, stride=2)]
    for _ in range(modules - 1):
        layers += [residual(k, out_dim, out_dim)]
    return nn.Sequential(*layers)

def make_layer_revr(k, inp_dim, out_dim, modules, **kwargs):
    layers = []
    for _ in range(modules - 1):
        layers.append(residual(k, inp_dim, inp_dim, **kwargs))
    layers.append(residual(k, inp_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)


class kp_module(nn.Module):
    def __init__(self, n, dims, modules, **kwargs):
        super(kp_module, self).__init__()
        self.n   = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        # 将输入进来的特征层进行两次残差卷积，便于和后面的层进行融合
        self.up1  = make_layer(
            3, curr_dim, curr_dim, curr_mod, **kwargs
        )  

        # 进行下采样
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod, **kwargs
        )

        # 构建U形结构的下一层
        if self.n > 1 :
            self.low2 = kp_module(
                n - 1, dims[1:], modules[1:], **kwargs
            ) 
        else:
            self.low2 = make_layer(
                3, next_dim, next_dim, next_mod, **kwargs
            )

        # 将U形结构下一层反馈上来的层进行残差卷积
        self.low3 = make_layer_revr(
            3, next_dim, curr_dim, curr_mod, **kwargs
        )
        # 将U形结构下一层反馈上来的层进行上采样
        self.up2  = nn.Upsample(scale_factor=2)

    def forward(self, x):
        print('***********************************************')
        print(f'in the kp_module the input shape is {x.shape}')
        up1  = self.up1(x)
        print(f'in the kp_module the self.up1(x) shape is {up1.shape}')
        low1 = self.low1(x)
        print(f'in the kp_module the self.low1(x) shape is {low1.shape}')
        low2 = self.low2(low1)
        print(f'in the kp_module the self.low2(low1) shape is {low2.shape}')
        low3 = self.low3(low2)
        print(f'in the kp_module the self.low3(low2) shape is {low3.shape}')
        up2  = self.up2(low3)
        print(f'in the kp_module the self.up2(low3) shape is {up2.shape}')
        outputs = up1 + up2
        print(f'in the kp_module the up1 + up2 shape is {outputs.shape}')
        print('========================================================')
        return outputs

if __name__ == '__main__':

    hourglass = kp_module(n=5, dims=[128, 128, 256, 512, 512, 1024], modules = [2, 2, 2, 2, 2, 4] )
    x1 = torch.randn( 1, 1 , 128, 128)

    x2 = torch.randn(1, 1, 512, 512)

    y = hourglass(x1)
    print(f'y shape is {y.shape}')
    # print(f'the hmap shape is {hmap.shape}')
    # print(f'the whd shape is {whd.shape}')
    # print(f'the offset shape is {offset.shape}')
    # embed() hmap, whd, offset