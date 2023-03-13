
from imgaug import augmenters as iaa
import numpy as np
import torchvision
import torchvision.transforms as transforms
from auto_augment import AutoAugment, Cutout
import albumentations
import numpy

class ImgTrainTransform2:

    def __init__(self, size=(224,224), normalization=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):

        self.normalization = normalization
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.25, iaa.Affine(scale={"x": (1.0, 2.0), "y": (1.0, 2.0)})),
            iaa.Scale(size),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            iaa.Sometimes(0.25, iaa.Affine(rotate=(-120, 120), mode='symmetric')),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 1.5))),

            # noise
            iaa.Sometimes(0.1,
                          iaa.OneOf([
                              iaa.Dropout(p=(0, 0.05)),
                              iaa.CoarseDropout(0.02, size_percent=0.25)
                          ])),

            iaa.Sometimes(0.25,
                          iaa.OneOf([
                              iaa.Add((-15, 15), per_channel=0.5), # brightness
                              iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
                          ])),

        ])

    def __call__(self, img):
        img = self.aug.augment_image(np.array(img)).copy()
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.normalization[0], self.normalization[1]),
        ])
        return transforms(img)


class ImgTrainTransform3:

    def __init__(self, size=(224,224), normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):

        self.normalization = normalization
        self.size = size

    def __call__(self, img):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(degrees=30),      # random rotation up to 30 degrees
            torchvision.transforms.RandomResizedCrop(size=self.size),     # random zoom and cropping to size 224x224
            torchvision.transforms.RandomAffine(degrees=0,          # random width and height shift up to 0.1
                                    translate=(0.1, 0.1),
                                    scale=(0.9, 1.1)),
            torchvision.transforms.RandomAffine(degrees=0,          # random shearing up to 10 degrees
                                    shear=10),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),     # horizontal flipping with probability 0.5
            torchvision.transforms.RandomVerticalFlip(p=0.5),       # vertical flipping with probability 0.5
            # torchvision.transforms.Resize(self.size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.normalization[0], self.normalization[1]),
        ])
        return transforms(img)

class ImgEvalTransform:

    def __init__(self, size=(224,224), normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):

        self.normalization = normalization
        self.size = size

    def __call__(self, img):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.normalization[0], self.normalization[1]),
        ])
        return transforms(img)

class ImgTrainTransform:

    def __init__(self, size=(224,224), normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), type=1, crop_mode = None):

        self.size = size
        self.normalization = normalization
        self.type = type
        self.crop_mode = crop_mode

    def __call__(self, img):
        if self.type == 1:
            transform = self.getTransform(img.size[1])
            return transform(img)
        else:
            transform = self.getTransform2()
            return transform(image=numpy.array(img))['image']
    
    def getTransform(self, img_size):
        #See: https://github.com/4uiiurz1/pytorch-auto-augment
        t = [transforms.RandomCrop(img_size)] if self.crop_mode == "random" else []
        t = [transforms.CenterCrop(img_size)] if self.crop_mode == "center" else []
        t.extend([
            transforms.Resize(self.size),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            Cutout(),
            transforms.ToTensor(),
            transforms.Normalize(self.normalization[0], self.normalization[1]),
        ])

        return transforms.Compose(t)

    def getTransform2(self):
        return albumentations.Compose([
            albumentations.Transpose(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightness(limit=0.2, p=0.75),
            albumentations.RandomContrast(limit=0.2, p=0.75),
            albumentations.OneOf([
                albumentations.MotionBlur(blur_limit=5),
                albumentations.MedianBlur(blur_limit=5),
                albumentations.GaussianBlur(blur_limit=5),
                albumentations.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),

            albumentations.OneOf([
                albumentations.OpticalDistortion(distort_limit=1.0),
                albumentations.GridDistortion(num_steps=5, distort_limit=1.),
                albumentations.ElasticTransform(alpha=3),
            ], p=0.7),

            albumentations.CLAHE(clip_limit=4.0, p=0.7),
            albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            albumentations.Resize(self.size[0], self.size[0]),
            albumentations.Cutout(max_h_size=int(self.size[0] * 0.375), max_w_size=int(self.size[0] * 0.375), num_holes=1, p=0.7),
            albumentations.Normalize()
        ])


