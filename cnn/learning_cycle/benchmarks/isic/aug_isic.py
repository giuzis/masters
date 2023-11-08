
from imgaug import augmenters as iaa
import numpy as np
import torchvision
from preprocessing import PreProcessing


class ImgTrainTransformWithDA:

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

