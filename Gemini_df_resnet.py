# Model file for Gemini DF-ResNet in 'Golden Gemini is All You Need: 
# Finding the Sweet Spots for Speaker Verification' 
# https://arxiv.org/abs/2312.03620

# Author: Tianchi Liu
# Special thanks to the Author of DF-ResNet: Dr. Bei Liu

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchaudio.transforms as trans


class Inverted_Bottleneck(nn.Module):
    def __init__(self, dim):
        super(Inverted_Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4 * dim)
        self.conv2 = nn.Conv2d(4 * dim, 4 * dim, kernel_size=3, padding=1, groups=4 * dim, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * dim)
        self.conv3 = nn.Conv2d(4 * dim, dim, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(dim)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x
        out = F.relu(out)
        return out


class Gemini_DF_ResNet(nn.Module):
    # DF_ResNet with Golden Gemini T14c stride strategy of Golden Gemini
    def __init__(self, depths=[3, 3, 9, 3], dims=[32, 64, 128, 256], feat_dim=40, emb_dim=128, feat_type='fbank', sr=16000):
        super(Gemini_DF_ResNet, self).__init__()
        self.feat_dim = feat_dim
        self.emb_dim = emb_dim
        self.feat_type = feat_type

        win_len = int(sr * 0.025)
        hop_len = int(sr * 0.01)

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(1, dims[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU()
        )
        self.downsample_layers.append(stem)

        # Golden Gemini T14c stride strategy 
        stride_f = [2,2,2,2]
        stride_t = [1,2,1,1]

        for i in range(4):
            downsample_layer = nn.Sequential(
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=(stride_f[i], stride_t[i]), padding=1, bias=False),
                nn.BatchNorm2d(dims[i + 1])
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[Inverted_Bottleneck(dim=dims[i+1]) for j in range(depths[i])]
            )
            self.stages.append(stage)

        self.embedding = nn.Linear(math.ceil(feat_dim / 8) * dims[-1], emb_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)

        x = x.unsqueeze_(1)
        x = self.downsample_layers[0](x)
        x = self.downsample_layers[1](x)
        x = self.stages[0](x)
        x = self.downsample_layers[2](x)
        x = self.stages[1](x)
        x = self.downsample_layers[3](x)
        x = self.stages[2](x)        
        x = self.downsample_layers[4](x)
        x = self.stages[3](x)

        pooling_mean = torch.mean(x, dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-10)
        out = torch.cat((torch.flatten(pooling_mean, start_dim=1),
                         torch.flatten(pooling_std, start_dim=1)), 1)


        embedding = self.embedding(out)
        return embedding


# following models do NOT include separate downsmapling layers into layer counting
def Gemini_df_resnet56(feat_dim, embed_dim, feat_type='fbank', sr=16000): # this is actually Gemini_df_resnet60 in paper
    return Gemini_DF_ResNet(depths=[3, 3, 9, 3], dims=[32, 32, 64, 128, 256], feat_dim=feat_dim, emb_dim=embed_dim, feat_type=feat_type, sr=sr)

def Gemini_df_resnet110(feat_dim, embed_dim, feat_type='fbank', sr=16000): # this is actually Gemini_df_resnet114 in paper
    return Gemini_DF_ResNet(depths=[3, 3, 27, 3], dims=[32, 32, 64, 128, 256], feat_dim=feat_dim, emb_dim=embed_dim, feat_type=feat_type, sr=sr)


def Gemini_df_resnet179(feat_dim, embed_dim, feat_type='fbank', sr=16000): # this is actually Gemini_df_resnet183 in paper
    return Gemini_DF_ResNet(depths=[3, 8, 45, 3], dims=[32, 32, 64, 128, 256], feat_dim=feat_dim, emb_dim=embed_dim, feat_type=feat_type, sr=sr)

# following models do include separate downsmapling layers into layer counting
def Gemini_df_resnet60(feat_dim, embed_dim, feat_type='fbank', sr=16000):
    return Gemini_DF_ResNet(depths=[3, 3, 9, 3], dims=[32, 32, 64, 128, 256], feat_dim=feat_dim, emb_dim=embed_dim, feat_type=feat_type, sr=sr)

def Gemini_df_resnet114(feat_dim, embed_dim, feat_type='fbank', sr=16000):
    return Gemini_DF_ResNet(depths=[3, 3, 27, 3], dims=[32, 32, 64, 128, 256], feat_dim=feat_dim, emb_dim=embed_dim, feat_type=feat_type, sr=sr)


def Gemini_df_resnet183(feat_dim, embed_dim, feat_type='fbank', sr=16000):
    return Gemini_DF_ResNet(depths=[3, 8, 45, 3], dims=[32, 32, 64, 128, 256], feat_dim=feat_dim, emb_dim=embed_dim, feat_type=feat_type, sr=sr)

if __name__ == '__main__':
    net = Gemini_DF_ResNet56(80, 256)
    x = torch.randn(2, 32000)
    out = net(x)
    print(out.shape)
