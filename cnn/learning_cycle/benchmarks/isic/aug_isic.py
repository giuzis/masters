
from imgaug import augmenters as iaa
import numpy as np
import torchvision
import torchvision.transforms as transforms
from auto_augment import AutoAugment, Cutout
from preprocessing import PreProcessing
import albumentations
from albumentations.pytorch import ToTensorV2


class ImgTrainTransform2_wrong:

    def __init__(self, size=(224,224), normalization=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                 pp_enhancement = None, pp_hair_removal = None, pp_color_constancy = None, pp_denoising = None):

        self.size = size
        self.normalization = normalization
        self.pp_enhancement = pp_enhancement
        self.pp_hair_removal = pp_hair_removal
        self.pp_color_constancy = pp_color_constancy
        self.pp_denoising = pp_denoising
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.25, iaa.Affine(scale={"x": (1.0, 2.0), "y": (1.0, 2.0)})), # zoom in
            iaa.Scale(size), # resize to 224x224
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            iaa.Sometimes(0.25, iaa.Affine(rotate=(-120, 120), mode='symmetric')), # rotate up to 120 degrees
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 1.5))), # blur images with a sigma of 0 to 1.5

            # prevent overfitting and encourage generalization
            iaa.Sometimes(0.1, 
                          iaa.OneOf([
                              iaa.Dropout(p=(0, 0.05)), # remove rectangular regions of up to 5% of the image
                              iaa.CoarseDropout(0.02, size_percent=0.25) # randomly remove up to 5% of the pixels
                          ])),

            iaa.Sometimes(0.25,
                          iaa.OneOf([
                              iaa.Add((-15, 15), per_channel=0.5), # change brightness of images (by -15 to 15 of original value)
                              iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True) # change hue and saturation
                          ])),

        ])

    def __call__(self, img):
        img = self.aug.augment_image(np.array(img)).copy()
        transforms = []
        if self.pp_enhancement is not None or self.pp_hair_removal is not None or self.pp_color_constancy is not None or self.pp_denoising is not None:
            transforms.append(PreProcessing(self.pp_enhancement, self.pp_hair_removal, self.pp_color_constancy, self.pp_denoising))
        transforms.extend([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.normalization[0], self.normalization[1]),
        ])
        transforms = torchvision.transforms.Compose(transforms)
        return transforms(img)
    
# right
class ImgTrainTransform2:

    def __init__(self, size=(224,224), normalization=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                 pp_enhancement = None, pp_hair_removal = None, pp_color_constancy = None, pp_denoising = None):

        self.size = size
        self.normalization = normalization
        self.pp_enhancement = pp_enhancement
        self.pp_hair_removal = pp_hair_removal
        self.pp_color_constancy = pp_color_constancy
        self.pp_denoising = pp_denoising
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.25, iaa.Affine(scale={"x": (1.0, 2.0), "y": (1.0, 2.0)})), # zoom in
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            iaa.Sometimes(0.25, iaa.Affine(rotate=(-120, 120), mode='symmetric')), # rotate up to 120 degrees
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 1.5))), # blur images with a sigma of 0 to 1.5

            # prevent overfitting and encourage generalization
            iaa.Sometimes(0.1, 
                          iaa.OneOf([
                              iaa.Dropout(p=(0, 0.05)), # remove rectangular regions of up to 5% of the image
                              iaa.CoarseDropout(0.02, size_percent=0.25) # randomly remove up to 5% of the pixels
                          ])),

            iaa.Sometimes(0.25,
                          iaa.OneOf([
                              iaa.Add((-15, 15), per_channel=0.5), # change brightness of images (by -15 to 15 of original value)
                              iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True) # change hue and saturation
                          ])),

        ])

    def __call__(self, img):
        transforms_pp = []
        transforms_pp.append(torchvision.transforms.Resize(self.size))

        if self.pp_enhancement is not None or self.pp_hair_removal is not None or self.pp_color_constancy is not None or self.pp_denoising is not None:
            transforms_pp.append(PreProcessing(self.pp_enhancement, self.pp_hair_removal, self.pp_color_constancy, self.pp_denoising))
        
        transforms_pp = torchvision.transforms.Compose(transforms_pp)

        img = transforms_pp(img)

        img = self.aug.augment_image(np.array(img)).copy()

        transform = []
        transform.extend([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.normalization[0], self.normalization[1]),
        ])
        transform = torchvision.transforms.Compose(transform)

        return transform(img)

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

class ImgTrainTransform1:

    def __init__(self, size=(224,224), normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):

        self.size = size
        self.normalization = normalization

    def __call__(self, img):
            transform = self.getTransform(img.size[1])
            return transform(img)
    
    def getTransform(self, img_size):
        #See: https://github.com/4uiiurz1/pytorch-auto-augment
        t = []
        # t = [transforms.RandomCrop(img_size)] if self.crop_mode == "random" else []
        # t = [transforms.CenterCrop(img_size)] if self.crop_mode == "center" else []
        t.extend([
            transforms.Resize(self.size),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            Cutout(),
            transforms.ToTensor(),
            transforms.Normalize(self.normalization[0], self.normalization[1]),
        ])

        return transforms.Compose(t)

    


class ImgTrainTransform0:

    def __init__(self, size=(224,224), normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
                 pp_enhancement = None, pp_hair_removal = None, pp_color_constancy = None, pp_denoising = None):

        self.size = size
        self.normalization = normalization
        self.pp_enhancement = pp_enhancement
        self.pp_hair_removal = pp_hair_removal
        self.pp_color_constancy = pp_color_constancy
        self.pp_denoising = pp_denoising

    def __call__(self, img):
        transforms = []
        transforms.append(torchvision.transforms.Resize(self.size))
        transforms.extend([
            torchvision.transforms.RandomResizedCrop(size=self.size, scale=(0.5,1)),     # random zoom and cropping to size 224x224
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            Cutout(),
            torchvision.transforms.RandomAffine(degrees=0,          # random width and height shift up to 0.1
                                    translate=(0.1, 0.1),
                                    scale=(0.9, 1.1)),
            torchvision.transforms.RandomRotation(degrees=30),      # random rotation up to 30 degrees
            torchvision.transforms.RandomHorizontalFlip(p=0.5),     # horizontal flipping with probability 0.5
            torchvision.transforms.RandomVerticalFlip(p=0.5),       # vertical flipping with probability 0.5
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.normalization[0], self.normalization[1]),
        ])
        if self.pp_enhancement is not None or self.pp_hair_removal is not None or self.pp_color_constancy is not None or self.pp_denoising is not None:
            transforms.append(PreProcessing(self.pp_enhancement, self.pp_hair_removal, self.pp_color_constancy, self.pp_denoising))
        
        transforms = torchvision.transforms.Compose(transforms)
        return transforms(img)
    
class ImgTrainTransform4:

    def __init__(self, size=(224,224), normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):

        self.size = size
        self.normalization = normalization

    def __call__(self, img):
            transform = self.getTransform(self.size[0])
            return transform(image = np.array(img))['image']
    
    def getTransform(self, img_size):
        transforms_train = albumentations.Compose([
            albumentations.Resize(img_size, img_size),
            albumentations.Transpose(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightnessContrast(p=0.75),
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
            albumentations.CoarseDropout(max_height=int(img_size * 0.375), max_width=int(img_size * 0.375), min_holes=1, p=0.7),
            # albumentations.Cutout(max_h_size=int(img_size * 0.375), max_w_size=int(img_size * 0.375), num_holes=1, p=0.7),
            albumentations.Normalize(),
            ToTensorV2()
        ])

        return transforms_train


    
class ImgTrainTransformWithPP:

    def __init__(self, size=(224,224), normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 pp_enhancement = None, pp_hair_removal = None, pp_color_constancy = None, pp_denoising = None):

        self.normalization = normalization
        self.size = size
        
        self.pp_enhancement = pp_enhancement
        self.pp_hair_removal = pp_hair_removal
        self.pp_color_constancy = pp_color_constancy
        self.pp_denoising = pp_denoising

    def __call__(self, img):
        transforms = []
        transforms.append(torchvision.transforms.Resize(self.size))
        if self.pp_enhancement is not None or self.pp_hair_removal is not None or self.pp_color_constancy is not None or self.pp_denoising is not None:
            transforms.append(PreProcessing(self.pp_enhancement, self.pp_hair_removal, self.pp_color_constancy, self.pp_denoising))
        transforms.extend([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.normalization[0], self.normalization[1]),
        ])
        transforms = torchvision.transforms.Compose(transforms)
        return transforms(img)

