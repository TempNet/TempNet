from torchvision import transforms
from cv2 import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ToRGBTensor(object):
    def __call__(self, img):
        img = np.moveaxis(np.array(img), [0,1,2],[1,2,0])
        img = torch.from_numpy(img).to(torch.float32)
        return img

    def __repr__(self):
        return 'netvlad.ToRGBTensor'

    base_normlize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    base_train_transform_list =  [transforms.ToPILImage(),
                                  transforms.Resize(256),
                                  transforms.RandomCrop(224),
                                  transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
                                  transforms.ToTensor()]
    base_test_transform_list = [transforms.ToPILImage(),
                                  transforms.Resize(256),
                                  transforms.RandomCrop(224),
                                  transforms.ToTensor()]


def netvlad_transform(img_size=None, augmentations=[]):
    """
        Composes a data transformation for embedding
        Note: this will generate an RGB range image (not [0,1]) to optimize netvlad performance
        :param img_size: (int or None) if integer is given the image is resized to (img_size, img_size)
        :param: augmentations: (list<transformations>) a list of augmentation transformations to apply
        :return: the transformation to apply
    """
    if img_size is None:
        resize_transform = []
    else:
        resize_transform = [transforms.Resize((img_size, img_size))]
    transforms_list = [transforms.ToPILImage(), *resize_transform, *augmentations, ToRGBTensor(), transforms.Normalize(
        mean=[123.68, 116.779, 103.939], std=[1.0, 1.0, 1.0])]

    transform = transforms.Compose(transforms_list)

    return transform


def load_state_dict(net, net_state_dict):
    """
    Loads a state dictionary
    :param net: (nn.module) the network to update
    :param net_state_dict: (dict) the state dictionary to load
    """
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(net_state_dict))
    else: # when loading a model trained on gpu, on a machine without gpu, we need to to map it to the CPU
        net.load_state_dict(torch.load(net_state_dict, map_location='cpu'))


def embed_keyframes(transformer, device, net, input):
    """
    Embeds the dataset using the given network
    :param transformer: composes a data transformation for embedding
    :param device: (torch.device) device context to use
    :param net: (nn.Module) embedding network
    :param input: data dict
    :return: Keypoints' embedding
    """

    # Calculate embedding of the database
    db_global_descriptors = []
    embedder = net.to(device).eval()

    with torch.no_grad():  # This is essential to avoid gradients accumulating on the GPU, which will cause memory blow-up
        for i, im in enumerate(input):
            raw_img = imread(im)
            raw_img = transformer(raw_img).to(device)
            img = raw_img.reshape(1, *raw_img.shape)
            db_embedding = embedder(img)
            global_desc = db_embedding.get('global_desc')
            db_global_descriptors.append(global_desc.tolist())

    db_global_descriptors = np.asarray(db_global_descriptors).reshape(len(db_global_descriptors), 4096)
    db_global_descriptors_T = torch.cuda.FloatTensor(db_global_descriptors)

    return db_global_descriptors, db_global_descriptors_T


def find_pair_indexes(nv_scores, precision):
    """
    Find image pairs indexes by using NetVLAD scores
    :param nv_scores: (nd array) a scores matrix
    :param precision: (float) a threshold for pairing images
    """
    pairs = np.argwhere(nv_scores > precision)
    non_dup_pairs = []
    for i in range(pairs.shape[0]):
        if pairs[i][0] < pairs[i][1]:
            non_dup_pairs.append(pairs[i])
    np.asarray(non_dup_pairs)
    return non_dup_pairs


class NetVLAD(nn.Module):
    """
    A class to represent a NetVLAD architecture, as described in:
    'NetVLAD: CNN architecture of weakly supervised recognition' Arandjelovix et al
    For a quick review - see Fig. 2 in the paper

    A NetVLAD implementation based on the following sources:
    https://github.com/Nanne/pytorch-NetVlad - Pytorch NetVLAD without whitening and PCA and with different normalization
    https://github.com/Relja/netvlad/blob/master/relja_simplenn_tidy.m - Original implementation in Matlab
    https://github.com/uzh-rpg/netvlad_tf_open - Tensorflow NetVLAD
    """

    def __init__(self):
        """
        NetVLAD constructor
        """
        super(NetVLAD, self).__init__()

        # vgg-16 encoder
        self._conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._relu1_1 = nn.ReLU()
        self._conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self._relu1_2 = nn.ReLU()
        self._conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._relu2_1 = nn.ReLU()
        self._conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self._relu2_2 = nn.ReLU()
        self._conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._relu3_1 = nn.ReLU()
        self._conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._relu3_2 = nn.ReLU()
        self._conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self._relu3_3 = nn.ReLU()
        self._conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._relu4_1 = nn.ReLU()
        self._conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._relu4_2 = nn.ReLU()
        self._conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self._relu4_3 = nn.ReLU()
        self._conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._relu5_1 = nn.ReLU()
        self._conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._relu5_2 = nn.ReLU()
        self._conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))

        # VLAD layer
        self._num_clusters = 64
        self._dim = 512
        self._centroids = nn.Parameter(torch.rand(self._num_clusters, self._dim))
        self._vlad_conv = nn.Conv2d(self._dim, self._num_clusters, kernel_size=(1, 1), bias=False)
        self._centroids = nn.Parameter(torch.rand(self._num_clusters, self._dim))

        # Whitening and pca
        self._wpca = nn.Conv2d(32768, 4096, kernel_size=[1, 1])

    def vgg16_descriptor(self, x0, latent_return_layers=None):
        """
        Embed an input image with the VGG16 encoder
        :param x0: (torch.tensor) input image batch of size N
        :param latent_return_layers: list<str> list of names of latent layers, for which the latent representations
            should be returned
        :return: Nx 512-dimensional feature map
        """
        latent_return_repr = []
        x1 = self._conv1_1(x0)
        x2 = self._relu1_1(x1)
        x3 = self._conv1_2(x2)
        x4 = self._pool1(x3)
        x5 = self._relu1_2(x4)
        x6 = self._conv2_1(x5)
        x7 = self._relu2_1(x6)
        x8 = self._conv2_2(x7)
        x9 = self._pool2(x8)
        x10 = self._relu2_2(x9)
        x11 = self._conv3_1(x10)
        x12 = self._relu3_1(x11)
        x13 = self._conv3_2(x12)
        x14 = self._relu3_2(x13)
        x15 = self._conv3_3(x14)
        x16 = self._pool3(x15)
        x17 = self._relu3_3(x16)
        x18 = self._conv4_1(x17)
        x19 = self._relu4_1(x18)
        x20 = self._conv4_2(x19)
        x21 = self._relu4_2(x20)
        x22 = self._conv4_3(x21)
        x23 = self._pool4(x22)
        x24 = self._relu4_3(x23)
        x25 = self._conv5_1(x24)
        x26 = self._relu5_1(x25)
        x27 = self._conv5_2(x26)
        x28 = self._relu5_2(x27)
        x29 = self._conv5_3(x28)

        if latent_return_layers is not None:
            # fine layer conv3, coarse layer conv5
            latent_map = {"conv3":x15, "conv4":x22, "conv5":x29}
            for layer_name in latent_return_layers:
                latent_return_repr.append((latent_map.get(layer_name)))
        return x29, latent_return_repr

    def vlad_descriptor(self, x):
        """
        Apply a VLAD layer on an embedded tensor, followed by dimensionality reduction, whitening and normalization
        :param x: embedded tensor
        :return: 32-K tenstor (VLAD descriptor before dimensionality reduction)
        """
        batch_size, orig_num_of_channels = x.shape[:2]

        # Normalize across the descriptors dimension
        x = F.normalize(x, p=2, dim=1)

        # Soft-assignment of descriptors to cluster centers
        # NxDxWxH map interpreted as NxDxK descriptors
        soft_assign = self._vlad_conv(x).view(batch_size, self._num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        # VLAD core
        x_flatten = x.view(batch_size, orig_num_of_channels, -1)

        # Centroids are originally saved with minus sign, so the following operation translates to: x - C
        vlad = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) + self._centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)

        vlad *= soft_assign.unsqueeze(2)
        vlad = vlad.sum(dim=-1)

        # Intra-normalization (implemented as in matchconvnet)
        vlad = self.matconvnet_normalize(vlad, dim=2)
        vlad = vlad.permute(0,2,1)
        vlad = vlad.flatten(1)

        # L2 post normalization
        vlad = self.matconvnet_normalize(vlad, dim=1)

        return vlad # Shape: NX32k

    def reduced_vlad_descriptor(self, vlad_desc):
        reduced_vlad_desc = self._wpca(vlad_desc.reshape((*vlad_desc.shape, 1, 1)))
        reduced_vlad_desc = reduced_vlad_desc.view(reduced_vlad_desc.shape[0], reduced_vlad_desc.shape[1])
        reduced_vlad_desc = F.normalize(reduced_vlad_desc, p=2, dim=1)
        return reduced_vlad_desc

    def matconvnet_normalize(self, x, dim, eps=1e-12):
        denom = torch.sqrt(torch.sum(x**2, dim=dim, keepdim=True) + eps)
        return x/denom

    def forward(self, img, latent_return_layers=None):
        """
       Forward pass of the network
       :param img: (torch.tensor) input image batch of size N
       :param latent_return_layers: list<str> list of names of latent layers, for which the latent representations
            should be returned
       :return: a dictionary containing the following:
                    (1) Nx4096 VLAD descriptors
                    (2) Nx32K VLAD descriptors
                    (3) dictionary of latent representations whose keys are the strings given latent_return_layers
                        and its values are the corresponding latent representations
        """
        vgg_desc, latent_reprs = self.vgg16_descriptor(img, latent_return_layers=latent_return_layers)
        vlad_desc = self.vlad_descriptor(vgg_desc)
        reduced_vlad_desc = self.reduced_vlad_descriptor(vlad_desc)

        res = {'global_desc': reduced_vlad_desc,
               'raw_global_desc': vlad_desc,
               'latent_reprs': latent_reprs,
               'input_size': img.shape[2:]}
        return res