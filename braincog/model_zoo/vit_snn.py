# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2022/7/26 18:56
# User      : Floyed
# Product   : PyCharm
# Project   : BrainCog
# File      : vgg_snn.py
# explain   :

from functools import partial
from torch.nn import functional as F
import torchvision
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule
from braincog.datasets import is_dvs_data

import math
import torch
import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, node=LIFNode, **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = partial(node, **kwargs)()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, node=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, node_type=node)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, node=LIFNode):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.node = node

        self.proj = BaseConvModule(in_chans, embed_dim, kernel_size=patch_size,padding=(0, 0),stride=patch_size,node=self.node)

    def forward(self, x):
        x = self.proj(x)
        B,C,H,W = x.shape
        x = x.reshape(B,C,H*W).permute(0,2,1)
        return x

@register_model
class VisionTransformer(BaseModule):
    """ Vision Transformer """
    def __init__(self, 
                 num_classes=100,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 img_size=[32], patch_size=8, in_chans=3, embed_dim=128, depth=6,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm,
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, node=node_type)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, node=node_type)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def prepare_tokens(self, x):

        B = x.shape[0]
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.pos_embed

        return x

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()

        if self.layer_by_layer:
            x = self.prepare_tokens(inputs)
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)[:, 0, :]
            x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            return x
        
        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                x = self.prepare_tokens(x)
                print(x.shape)
                for blk in self.blocks:
                    x = blk(x)
                x = self.norm(x)
                outputs.append(x[:, 0, :])

            return self.head(sum(outputs) / len(outputs))
     

if __name__ == '__main__':
    from thop import profile
    size = 320
    model = VisionTransformer(img_size=[size])
    img = torch.randn([1,3,size,size])
    # y = model(img)

    flops, params = profile(model, inputs=(img,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')