from functools import reduce
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from .layers import SelfAttention
from torchsummary import summary

class Generator(nn.Module):
  def __init__(self, image_size = 64, z_dim = 100, conv_dim = 32):
    super().__init__()
    layer1, layer2, layer3 = [], [], []
    repeat_num = int(np.log2(image_size)) - 3
    mult = 2 ** repeat_num
    layer1.append(spectral_norm(nn.ConvTranspose2d(in_channels = z_dim, out_channels = conv_dim * mult, kernel_size = 4)))
    layer1.append(nn.BatchNorm2d(conv_dim * mult))
    layer1.append(nn.ReLU())
    curr_dim = conv_dim * mult
    layer2.append(spectral_norm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)))
    layer2.append(nn.BatchNorm2d(curr_dim // 2))
    layer2.append(nn.ReLU())
    curr_dim = curr_dim // 2
    layer3.append(spectral_norm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)))
    layer3.append(nn.BatchNorm2d(curr_dim // 2))
    layer3.append(nn.ReLU())
    layer4 = []
    curr_dim = curr_dim // 2
    layer4.append(spectral_norm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)))
    layer4.append(nn.BatchNorm2d(curr_dim // 2))
    layer4.append(nn.ReLU())
    self.l4 = nn.Sequential(*layer4)
    curr_dim = curr_dim // 2
    self.l1 = nn.Sequential(*layer1)
    self.l2 = nn.Sequential(*layer2)
    self.l3 = nn.Sequential(*layer3)
    self.last = nn.Sequential(
        nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1),
        nn.Tanh()
        )
    self.attn1 = SelfAttention(64)
    self.attn2 = SelfAttention(32)
  def forward(self, input):
    input = input.view(input.size(0), input.size(1), 1, 1)
    out = self.l1(input) #256
    out = self.l2(out) # 128
    out = self.l3(out) # 64
    out = self.attn1(out) # 64
    out = self.l4(out) # 32 
    out = self.attn2(out)
    out = self.last(out)
    return out
# class Generator(nn.Module):
#     def __init__(self, out_channels = 3, image_size = 64, latent_dim = 100, ngf = 32):
#         super().__init__()
#         def convt_2d(in_channels, out_channels, kernel_size, stride = 1, padding = 0):
#             return nn.Sequential(
#                 spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU()
#             )
#         repeat_num = int(np.log2(image_size)) - 3
#         mult = 1 << repeat_num
#         self.block_1 = convt_2d(latent_dim, ngf * mult, 4)
#         current_dim = ngf * mult
#         self.block_2 = convt_2d(current_dim, current_dim // 2, 4, 2, 1)
#         current_dim //= 2
#         self.block_3 = convt_2d(current_dim, current_dim // 2, 4, 2, 1)
#         current_dim //= 2
#         self.attn_layer_1 = SelfAttention(current_dim)
#         self.block_4 = convt_2d(current_dim, current_dim // 2, 4, 2, 1)
#         current_dim //= 2
#         self.attn_layer_2 = SelfAttention(current_dim)
#         self.last_layer = nn.Sequential(
#             nn.ConvTranspose2d(current_dim, out_channels, 4, 2, 1),
#             nn.Tanh()
#         )
#     def forward(self, input):
#         all_layers = [self.block_1, self.block_2, self.block_3, 
#                         self.attn_layer_1, self.block_4, self.attn_layer_2, self.last_layer]
#         return reduce(lambda x, layer: layer(x), all_layers, input)

class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, image_size = 64, ndf = 32):
        super().__init__()
        def conv_2d(in_channels, out_channels, kernel_size, stride = 1, padding = 0):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)),
                nn.LeakyReLU(.1)
            )
        self.block_1 = conv_2d(in_channels, ndf, 4, 2, 1)
        current_dim = ndf
        self.block_2 = conv_2d(current_dim, current_dim * 2, 4, 2, 1)
        current_dim *= 2
        self.block_3 = conv_2d(current_dim, current_dim * 2, 4, 2, 1)
        current_dim *= 2
        self.attn_layer_1 = SelfAttention(current_dim)
        self.block_4 = conv_2d(current_dim, current_dim * 2, 4, 2, 1)
        current_dim *= 2
        self.attn_layer_2 = SelfAttention(current_dim)
        self.last_layer = nn.Conv2d(current_dim, 1, 4)
    def forward(self, input):
        all_layers = [self.block_1, self.block_2, self.block_3, self.attn_layer_1,
                        self.block_4, self.attn_layer_2, self.last_layer]
        return reduce(lambda x, layer: layer(x), all_layers, input)
