import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F



class _bn_relu_conv(nn.Module):
    def __init__(self, in_filters, nb_filters, fw, fh, subsample=1):
        super(_bn_relu_conv, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_filters, eps=1e-3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_filters, nb_filters, (fw, fh), stride=subsample, padding=(fw//2, fh//2), padding_mode='zeros')
        )

    def forward(self, x):
        return self.model(x)

        # the following are for debugs
        print("****", np.max(x.cpu().numpy()), np.min(x.cpu().numpy()), np.mean(x.cpu().numpy()), np.std(x.cpu().numpy()), x.shape)
        for i,layer in enumerate(self.model):
            if i != 2:
                x = layer(x)
            else:
                x = layer(x)
                #x = nn.functional.pad(x, (1, 1, 1, 1), mode='constant', value=0)
            print("____", np.max(x.cpu().numpy()), np.min(x.cpu().numpy()), np.mean(x.cpu().numpy()), np.std(x.cpu().numpy()), x.shape)
            print(x[0])
        return x


class _u_bn_relu_conv(nn.Module):
    def __init__(self, in_filters, nb_filters, fw, fh, subsample=1):
        super(_u_bn_relu_conv, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_filters, eps=1e-3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_filters, nb_filters, (fw, fh), stride=subsample, padding=(fw//2, fh//2)),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        return self.model(x)



class _shortcut(nn.Module):
    def __init__(self, in_filters, nb_filters, subsample=1):
        super(_shortcut, self).__init__()
        self.process = False
        self.model = None
        if in_filters != nb_filters or subsample != 1:
            self.process = True
            self.model = nn.Sequential(
                    nn.Conv2d(in_filters, nb_filters, (1, 1), stride=subsample)
                )

    def forward(self, x, y):
        #print(x.size(), y.size(), self.process)
        if self.process:
            y0 = self.model(x)
            #print("merge+", torch.max(y0+y), torch.min(y0+y),torch.mean(y0+y), torch.std(y0+y), y0.shape)
            return y0 + y
        else:
            #print("merge", torch.max(x+y), torch.min(x+y),torch.mean(x+y), torch.std(x+y), y.shape)
            return x + y

class _u_shortcut(nn.Module):
    def __init__(self, in_filters, nb_filters, subsample):
        super(_u_shortcut, self).__init__()
        self.process = False
        self.model = None
        if in_filters != nb_filters:
            self.process = True
            self.model = nn.Sequential(
                nn.Conv2d(in_filters, nb_filters, (1, 1), stride=subsample, padding_mode='zeros'),
                nn.Upsample(scale_factor=2, mode='nearest')
            )

    def forward(self, x, y):
        if self.process:
            return self.model(x) + y
        else:
            return x + y


class basic_block(nn.Module):
    def __init__(self, in_filters, nb_filters, init_subsample=1):
        super(basic_block, self).__init__()
        self.conv1 = _bn_relu_conv(in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _shortcut(in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.residual(x1)
        return self.shortcut(x, x2)

class _u_basic_block(nn.Module):
    def __init__(self, in_filters, nb_filters, init_subsample=1):
        super(_u_basic_block, self).__init__()
        self.conv1 = _u_bn_relu_conv(in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _u_shortcut(in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x):
        y = self.residual(self.conv1(x))
        return self.shortcut(x, y)


class _residual_block(nn.Module):
    def __init__(self, in_filters, nb_filters, repetitions, is_first_layer=False):
        super(_residual_block, self).__init__()
        layers = []
        for i in range(repetitions):
            init_subsample = 1
            if i == repetitions - 1 and not is_first_layer:
                init_subsample = 2
            if i == 0:
                l = basic_block(in_filters=in_filters, nb_filters=nb_filters, init_subsample=init_subsample)
            else:
                l = basic_block(in_filters=nb_filters, nb_filters=nb_filters, init_subsample=init_subsample)
            layers.append(l)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class _upsampling_residual_block(nn.Module):
    def __init__(self, in_filters, nb_filters, repetitions):
        super(_upsampling_residual_block, self).__init__()
        layers = []
        for i in range(repetitions):
            l = None
            if i == 0: 
                l = _u_basic_block(in_filters=in_filters, nb_filters=nb_filters)#(input)
            else:
                l = basic_block(in_filters=nb_filters, nb_filters=nb_filters)#(input)
            layers.append(l)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class res_skip(nn.Module):

    def __init__(self):
        super(res_skip, self).__init__()
        self.block0 = _residual_block(in_filters=1, nb_filters=24, repetitions=2, is_first_layer=True)#(input)
        self.block1 = _residual_block(in_filters=24, nb_filters=48, repetitions=3)#(block0)
        self.block2 = _residual_block(in_filters=48, nb_filters=96, repetitions=5)#(block1)
        self.block3 = _residual_block(in_filters=96, nb_filters=192, repetitions=7)#(block2)
        self.block4 = _residual_block(in_filters=192, nb_filters=384, repetitions=12)#(block3)
        
        self.block5 = _upsampling_residual_block(in_filters=384, nb_filters=192, repetitions=7)#(block4)
        self.res1 = _shortcut(in_filters=192, nb_filters=192)#(block3, block5, subsample=(1,1))

        self.block6 = _upsampling_residual_block(in_filters=192, nb_filters=96, repetitions=5)#(res1)
        self.res2 = _shortcut(in_filters=96, nb_filters=96)#(block2, block6, subsample=(1,1))

        self.block7 = _upsampling_residual_block(in_filters=96, nb_filters=48, repetitions=3)#(res2)
        self.res3 = _shortcut(in_filters=48, nb_filters=48)#(block1, block7, subsample=(1,1))

        self.block8 = _upsampling_residual_block(in_filters=48, nb_filters=24, repetitions=2)#(res3)
        self.res4 = _shortcut(in_filters=24, nb_filters=24)#(block0,block8, subsample=(1,1))

        self.block9 = _residual_block(in_filters=24, nb_filters=16, repetitions=2, is_first_layer=True)#(res4)
        self.conv15 = _bn_relu_conv(in_filters=16, nb_filters=1, fh=1, fw=1, subsample=1)#(block7)

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x5 = self.block5(x4)
        res1 = self.res1(x3, x5)

        x6 = self.block6(res1)
        res2 = self.res2(x2, x6)

        x7 = self.block7(res2)
        res3 = self.res3(x1, x7)

        x8 = self.block8(res3)
        res4 = self.res4(x0, x8)

        x9 = self.block9(res4)
        y = self.conv15(x9)

        return y

class MyDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def get_class_label(self, image_name):
        # your method here
        head, tail = os.path.split(image_name)
        #print(tail)
        return tail
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        y = self.get_class_label(image_path.split('/')[-1])
        if self.transform is not None:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.image_paths)

def loadImages(folder):
    imgs = []
    matches = []
    
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            matches.append(file_path)
    
    return matches


def crop_center_square(image):

    width, height = image.size
    

    side_length = min(width, height)

    left = (width - side_length) // 2
    top = (height - side_length) // 2
    right = left + side_length
    bottom = top + side_length
    
    cropped_image = image.crop((left, top, right, bottom))
    
    return cropped_image

def crop_image(image, crop_size, stride):

    width, height = image.size
    crop_width, crop_height = crop_size
    cropped_images = []

    for j in range(0, height - crop_height + 1, stride):
        for i in range(0, width - crop_width + 1, stride):
            crop_box = (i, j, i + crop_width, j + crop_height)
            cropped_image = image.crop(crop_box)
            cropped_images.append(cropped_image)

    return cropped_images

def process_image_ref(image):

    resized_image_512 = image.resize((512, 512))


    image_list = [resized_image_512]


    crop_size_384 = (384, 384)
    stride_384 = 128
    image_list.extend(crop_image(resized_image_512, crop_size_384, stride_384))


    return image_list


def process_image_Q(image):

    resized_image_512 = image.resize((512, 512)).convert("RGB").convert("RGB")

    image_list = []

    crop_size_384 = (384, 384)
    stride_384 = 128
    image_list.extend(crop_image(resized_image_512, crop_size_384, stride_384))

    return image_list

def process_image(image, target_width=512, target_height = 512):
    img_width, img_height = image.size
    img_ratio = img_width / img_height
    
    target_ratio = target_width / target_height
    
    ratio_error = abs(img_ratio - target_ratio) / target_ratio
    
    if ratio_error < 0.15:
        resized_image = image.resize((target_width, target_height), Image.BICUBIC)
    else:
        if img_ratio > target_ratio:
            new_width = int(img_height * target_ratio)
            left = int((0 + img_width - new_width)/2)
            top = 0
            right = left + new_width
            bottom = img_height
        else:
            new_height = int(img_width / target_ratio)
            left = 0
            top = int((0 + img_height - new_height)/2)
            right = img_width
            bottom = top + new_height
        
        cropped_image = image.crop((left, top, right, bottom))
        resized_image = cropped_image.resize((target_width, target_height), Image.BICUBIC)
    
    return resized_image.convert('RGB')

def crop_image_varres(image, crop_size, h_stride, w_stride):
        width, height = image.size
        crop_width, crop_height = crop_size
        cropped_images = []

        for j in range(0, height - crop_height + 1, h_stride):
            for i in range(0, width - crop_width + 1, w_stride):
                crop_box = (i, j, i + crop_width, j + crop_height)
                cropped_image = image.crop(crop_box)
                cropped_images.append(cropped_image)

        return cropped_images

def process_image_ref_varres(image, target_width=512, target_height = 512):
    resized_image_512 = image.resize((target_width, target_height))

    image_list = [resized_image_512]

    crop_size_384 = (target_width//4*3, target_height//4*3)
    w_stride_384 = target_width//4
    h_stride_384 = target_height//4
    image_list.extend(crop_image_varres(resized_image_512, crop_size_384, h_stride = h_stride_384, w_stride = w_stride_384))

    return image_list


def process_image_Q_varres(image, target_width=512, target_height = 512):

    resized_image_512 = image.resize((target_width, target_height)).convert("RGB").convert("RGB")

    image_list = []

    crop_size_384 = (target_width//4*3, target_height//4*3)
    w_stride_384 = target_width//4
    h_stride_384 = target_height//4
    image_list.extend(crop_image_varres(resized_image_512, crop_size_384, h_stride = h_stride_384, w_stride = w_stride_384))


    return image_list



import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 直接相加
        out = F.relu(out)
        return out

class TwoLayerResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TwoLayerResNet, self).__init__()
        self.block1 = ResNetBlock(in_channels, out_channels)
        self.block2 = ResNetBlock(out_channels, out_channels)
        self.block3 = ResNetBlock(out_channels, out_channels)
        self.block4 = ResNetBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x
    

class MultiHiddenResNetModel(nn.Module):
    def __init__(self, channels_list, num_tensors):
        super(MultiHiddenResNetModel, self).__init__()
        self.two_layer_resnets = nn.ModuleList([TwoLayerResNet(channels_list[idx]*2, channels_list[min(len(channels_list)-1,idx+2)]) for idx in range(num_tensors)])

    def forward(self, tensor_list):
        processed_list = []
        for i, tensor in enumerate(tensor_list):
            tensor = self.two_layer_resnets[i](tensor)
            processed_list.append(tensor)
        
        return processed_list
    

def calculate_target_size(h, w):
    if random.random()>0.5:
        target_h = (h // 8) * 8
        target_w = (w // 8) * 8
    elif random.random()>0.5:
        target_h = (h // 8) * 8
        target_w = (w // 8) * 8
    else:
        target_h = (h // 8) * 8
        target_w = (w // 8) * 8
    
    if target_h == 0:
        target_h = 8
    if target_w == 0:
        target_w = 8
    
    return target_h, target_w


def downsample_tensor(tensor):
    b, c, h, w = tensor.shape
    
    target_h, target_w = calculate_target_size(h, w)
    
    downsampled_tensor = F.interpolate(tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
    
    return downsampled_tensor



def get_pixart_config():
    pixart_config = {
            "_class_name": "Transformer2DModel",
            "_diffusers_version": "0.22.0.dev0",
            "activation_fn": "gelu-approximate",
            "attention_bias": True,
            "attention_head_dim": 72,
            "attention_type": "default",
            "caption_channels": 4096,
            "cross_attention_dim": 1152,
            "double_self_attention": False,
            "dropout": 0.0,
            "in_channels": 4,
            # "interpolation_scale": 2,
            "norm_elementwise_affine": False,
            "norm_eps": 1e-06,
            "norm_num_groups": 32,
            "norm_type": "ada_norm_single",
            "num_attention_heads": 16,
            "num_embeds_ada_norm": 1000,
            "num_layers": 28,
            "num_vector_embeds": None,
            "only_cross_attention": False,
            "out_channels": 8,
            "patch_size": 2,
            "sample_size": 128,
            "upcast_attention": False,
            # "use_additional_conditions": False,
            "use_linear_projection": False
            }
    return pixart_config



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # left
        self.left_conv_1 = DoubleConv(6, 64)
        self.down_1 = nn.MaxPool2d(2, 2)

        self.left_conv_2 = DoubleConv(64, 128)
        self.down_2 = nn.MaxPool2d(2, 2)

        self.left_conv_3 = DoubleConv(128, 256)
        self.down_3 = nn.MaxPool2d(2, 2)

        self.left_conv_4 = DoubleConv(256, 512)
        self.down_4 = nn.MaxPool2d(2, 2)

        # center
        self.center_conv = DoubleConv(512, 1024)

        # right
        self.up_1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.right_conv_1 = DoubleConv(1024, 512)

        self.up_2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.right_conv_2 = DoubleConv(512, 256)

        self.up_3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.right_conv_3 = DoubleConv(256, 128)

        self.up_4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.right_conv_4 = DoubleConv(128, 64)

        # output
        self.output = nn.Conv2d(64, 3, 1, 1, 0)

    def forward(self, x):
        # left
        x1 = self.left_conv_1(x)
        x1_down = self.down_1(x1)

        x2 = self.left_conv_2(x1_down)
        x2_down = self.down_2(x2)

        x3 = self.left_conv_3(x2_down)
        x3_down = self.down_3(x3)

        x4 = self.left_conv_4(x3_down)
        x4_down = self.down_4(x4)

        # center
        x5 = self.center_conv(x4_down)

        # right
        x6_up = self.up_1(x5)
        temp = torch.cat((x6_up, x4), dim=1)
        x6 = self.right_conv_1(temp)

        x7_up = self.up_2(x6)
        temp = torch.cat((x7_up, x3), dim=1)
        x7 = self.right_conv_2(temp)

        x8_up = self.up_3(x7)
        temp = torch.cat((x8_up, x2), dim=1)
        x8 = self.right_conv_3(temp)

        x9_up = self.up_4(x8)
        temp = torch.cat((x9_up, x1), dim=1)
        x9 = self.right_conv_4(temp)

        # output
        output = self.output(x9)

        return output



from copy import deepcopy

def init_causal_dit(model, base_model):
    temp_ckpt = deepcopy(base_model)
    checkpoint = temp_ckpt.state_dict()
    # checkpoint['pos_embed_1d.weight'] = torch.zeros(3, model.config.num_attention_heads * model.config.attention_head_dim, device=model.pos_embed_1d.weight.device, dtype = model.pos_embed_1d.weight.dtype)
    model.load_state_dict(checkpoint, strict=True)
    del temp_ckpt
    return model

def init_controlnet(model, base_model):
    temp_ckpt = deepcopy(base_model)
    checkpoint = temp_ckpt.state_dict()
    checkpoint_weight = checkpoint['pos_embed.proj.weight']
    new_weight = torch.zeros(model.pos_embed.proj.weight.shape, device=model.pos_embed.proj.weight.device, dtype = model.pos_embed.proj.weight.dtype)
    print('model.pos_embed.proj.weight.shape',model.pos_embed.proj.weight.shape)
    new_weight[:, :4] = checkpoint_weight
    checkpoint['pos_embed.proj.weight'] = new_weight
    print('new_weight', new_weight.dtype)
    model.load_state_dict(checkpoint, strict=False)
    del temp_ckpt
    return model