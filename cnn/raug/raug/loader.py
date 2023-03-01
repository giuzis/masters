#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements the methods and functions to load data as a PyTorch dataset

If you find any bug or have some suggestion, please, email me.
"""

from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np


# class MyDataset (data.Dataset):
#     """
#     This is the standard way to implement a dataset pipeline in PyTorch. We need to extend the torch.utils.data.Dataset
#     class and implement the following methods: __len__, __getitem__ and the constructor __init__
#     """

#     def __init__(self, imgs_path, labels, meta_data=None, transform=None):
#         """
#         The constructor gets the images path and their respectively labels and meta-data (if applicable).
#         In addition, you can specify some transform operation to be carry out on the images.

#         It's important to note the images must match with the labels (and meta-data if applicable). For example, the
#         imgs_path[x]'s label must take place on labels[x].

#         Parameters:
#         :param imgs_path (list): a list of string containing the image paths
#         :param labels (list) a list of labels for each image
#         :param meta_data (list): a list of meta-data regarding each image. If None, there is no information.
#         Defaul is None.
#         :param transform (torchvision.transforms.Compose): transform operations to be carry out on the images
#         """

#         super().__init__()
#         self.imgs_path = imgs_path
#         self.labels = labels
#         self.meta_data = meta_data

#         # if transform is None, we need to ensure that the PIL image will be transformed to tensor, otherwise we'll get
#         # an exception
#         if transform is not None:
#             self.transform = transform
#         else:
#             self.transform = transforms.ToTensor()


#     def __len__(self):
#         """ This method just returns the dataset size """
#         return len(self.imgs_path)


#     def __getitem__(self, item):
#         """
#         It gets the image, labels and meta-data (if applicable) according to the index informed in `item`.
#         It also performs the transform on the image.

#         :param item (int): an index in the interval [0, ..., len(img_paths)-1]
#         :return (tuple): a tuple containing the image, its label and meta-data (if applicable)
#         """
#         # try:
#         image = Image.open(self.imgs_path[item]).convert("RGB")

#         # print(self.imgs_path[item])
#         img_id = self.imgs_path[item].split('\\')[-1].split('.')[0]

#         # Applying the transformations
#         image = self.transform(image)

#         if self.meta_data is None:
#             meta_data = []
#         else:
#             meta_data = self.meta_data[item]

#         if self.labels is None:
#             labels = []
#         else:
#             labels = self.labels[item]

#         # except Exception as err:
#         #     print(self.imgs_path[item])
#         #     print(f"Unexpected in image {img_id}: {err}, {type(err)}")
#         #     raise

#         return image, labels, meta_data, img_id

class MyDatasetSeg (data.Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    PArameters:
        :param imgs_path (list): a list of string containing the image paths
        :param imgs_path_mask (list): a list of string containing the image mask paths
        :param class_values (list): values of classes to extract from segmentation mask
        :param augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        :param preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['lesion']
    
    def __init__(
            self, 
            imgs_path, 
            imgs_path_mask, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        
        self.imgs_path = imgs_path
        self.imgs_path_mask = imgs_path_mask
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, item):
        
        # read data
        
        image = Image.open(self.imgs_path[item]).convert("RGB")
        mask = Image.open(self.imgs_path_mask[item]).convert("L")
        # image = cv2.imread(self.images_fps[i])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        """ This method just returns the dataset size """
        return len(self.imgs_path)


def get_data_loader (imgs_path, labels, meta_data=None, transform=None, batch_size=30, shuf=True, num_workers=4,
                     pin_memory=True, segmentation = False):
    """
    This function gets a list og images path, their labels and meta-data (if applicable) and returns a DataLoader
    for these files. You also can set some transformations using torchvision.transforms in order to perform data
    augmentation. Lastly, params is a dictionary that you can set the following parameters:
    batch_size (int): the batch size for the dataset. If it's not informed the default is 30
    shuf (bool): set it true if wanna shuffe the dataset. If it's not informed the default is True
    num_workers (int): the number thread in CPU to load the dataset. If it's not informed the default is 0 (which


    :param imgs_path (list): a list of string containing the images path
    :param labels (list): a list of labels for each image
    :param meta_data (list, optional): a list of meta-data regarding each image. If it's None, it means there's
    no meta-data. Default is None
    :param transform (torchvision.transforms, optional): use the torchvision.transforms.compose to perform the data
    augmentation for the dataset. Alternatively, you can use the jedy.pytorch.utils.augmentation to perform the
    augmentation. If it's None, none augmentation will be perform. Default is None
    :param batch_size (int): the batch size. If the key is not informed or params = None, the default value will be 30
    :param shuf (bool): if you'd like to shuffle the dataset. If the key is not informed or params = None, the default
    value will be True
    :param num_workers (int): the number of threads to be used in CPU. If the key is not informed or params = None, the
    default value will be  4
    :param pin_memory (bool): set it to True to Pytorch preload the images on GPU. If the key is not informed or
    params = None, the default value will be True
    :return (torch.utils.data.DataLoader): a dataloader with the dataset and the chose params
    """

    dt = MyDataset(imgs_path, labels, meta_data, transform)
    dl = data.DataLoader (dataset=dt, batch_size=batch_size, shuffle=shuf, num_workers=num_workers,
                          pin_memory=pin_memory)
    return dl
