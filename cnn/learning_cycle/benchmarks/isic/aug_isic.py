
from imgaug import augmenters as iaa
import numpy as np
import torchvision

class ImgTrainTransform:

    def __init__(self, size=(224,224), normalization=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):

        self.normalization = normalization
        self.aug = torchvision.transforms.Compose([
            torchvision.transforms.Transpose(p=0.5),
            torchvision.transforms.VerticalFlip(p=0.5),
            torchvision.transforms.HorizontalFlip(p=0.5),
            torchvision.transforms.RandomBrightness(limit=0.2, p=0.75),
            torchvision.transforms.RandomContrast(limit=0.2, p=0.75),
            torchvision.transforms.OneOf([
                torchvision.transforms.MotionBlur(blur_limit=5),
                torchvision.transforms.MedianBlur(blur_limit=5),
                torchvision.transforms.GaussianBlur(blur_limit=5),
                torchvision.transforms.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),

            torchvision.transforms.OneOf([
                torchvision.transforms.OpticalDistortion(distort_limit=1.0),
                torchvision.transforms.GridDistortion(num_steps=5, distort_limit=1.),
                torchvision.transforms.ElasticTransform(alpha=3),
            ], p=0.7),

            torchvision.transforms.CLAHE(clip_limit=4.0, p=0.7),
            torchvision.transforms.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            torchvision.transforms.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            torchvision.transforms.Resize(size, size),
            torchvision.transforms.Cutout(max_h_size=int(size * 0.375), max_w_size=int(size * 0.375), num_holes=1, p=0.7),    
            torchvision.transforms.Normalize()
        ])

    def __call__(self, img):
        img = self.aug.augment_image(np.array(img)).copy()
        transforms = torchvision.transforms.Compose([
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


