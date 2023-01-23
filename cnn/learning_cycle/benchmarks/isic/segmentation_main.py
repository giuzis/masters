# -*- coding: utf-8 -*-
'''
Requeriments:
- Python (We use 3.5).
- Packages: matplotlib, skimage, numpy, mahotas.
'''
import faulthandler

faulthandler.enable()
import numpy as np
import pandas as pd
from skimage.color import rgb2gray, gray2rgb
from matplotlib.pyplot import imshow, figure, title, imsave, savefig, subplot
from skimage.metrics import mean_squared_error
import sys
import cv2

#Segmentation
from segmentation.segmentation import segment
from segmentation.utils import get_mask, compare_jaccard

from math import floor

def save_image(name, img, path):
    cv2.imwrite(path + '{}.jpg'.format(name), img)

def crop_image(img, coords):
    if len(coords) != 4:
        return img
    
    [b,t,r,l] = coords
    crop = img[t:b, l:r, :]

    return crop

def get_coordinates(img, gap, square = True):
    contours, _ = cv2.findContours(img.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return []

    y, x = img.shape
    coords = []
    
    b,t, r, l = [-1, 9999999, -1, 9999999]
    
    for points in contours:
        for p in points:
            if t > p[0][1]:
                t = p[0][1]
            if b < p[0][1]:
                b = p[0][1]
            if l > p[0][0]:
                l = p[0][0]
            if r < p[0][0]:
                r = p[0][0]

    if square:
        if b-t > r-l:
            l -= floor(((b-t) - (r-l))/2)
            r += floor(((b-t) - (r-l))/2) - l if l < 0 else floor(((b-t) - (r-l))/2)
        elif r-l > b-t:
            t -= floor(((r-l) - (b-t))/2)
            b += floor(((r-l) - (b-t))/2) - t if t < 0 else floor(((r-l) - (b-t))/2)

    b = b + gap if b + gap < y else y
    t = t - gap if t - gap > -1 else 0
    r = r + gap if r + gap < x else x
    l = l - gap if l - gap > -1 else 0

    return [b,t,r,l]

def remove_minimal_blobs(img):

    kernel = np.ones((10,10),np.uint8)
    img_clean = cv2.erode(img,kernel,iterations = 2)

    contours, _ = cv2.findContours(img_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_clean = np.zeros_like(img)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 8000:
            cv2.fillPoly(img_clean, [contour], 255)


    return img_clean

def segment_image(img, results_path, image_name, method='color', k=400):

    # segmentation_path = os.path.join(results_path, 'our_segmentation')
    # figures_path = os.path.join(results_path, 'figures')
    all_mse = []
    all_jaccard = []
    all_acc = []


    original_image = img
    image = image_name

    # Gets the mask to avoid dark areas in segmentation
    mask = get_mask(original_image.shape[0:2])
    I = gray2rgb(mask) * original_image
    # GT = (rgb2gray(gtimg.astype(float)*255) * mask) > 0 

    #Segment the each mole
    Isegmented, LMerged, Islic2, IOtsu, Superpixels = segment(I, mask, method=method, k=k)

    # auxmse = mean_squared_error(GT, Isegmented)
    # all_mse.append(auxmse)
    # aux_jaccard = compare_jaccard(GT, Isegmented)
    # aux_acc = 1.0 - np.sum(np.logical_xor(GT, Isegmented)) / float(GT.size)
    # all_jaccard.append(aux_jaccard)
    # all_acc.append(aux_acc)

    # print("Image name, MSE, JACCARD_IDX, ACC")
    # print("{:10} {:0.25f} {:0.25f} {:0.25f}".format(image, auxmse, aux_jaccard, aux_acc))

    # if not os.path.exists(segmentation_path):
    #     os.makedirs(segmentation_path)
    # if not os.path.exists(figures_path):
    #     os.makedirs(figures_path)

    # subplot(2, 3, 1)
    # title('Original + Superpixels')
    # imshow(Superpixels)
    # subplot(2, 3, 2)
    # title('Ground Truth')
    # # imshow(GT, cmap='gray')
    # subplot(2, 3, 3)
    # title('Our Segmentation')
    # imshow(Isegmented, cmap='gray')
    # subplot(2, 3, 4)
    # title('Labels')
    # imshow(LMerged)
    # subplot(2, 3, 5)
    # title('Merged Superpixels')
    # imshow(Islic2)
    # subplot(2, 3, 6)
    # title('Otsu')
    # imshow(IOtsu, cmap='gray')
    # savefig(figures_path + '/' + image + '_all.png')

    # imsave(segmentation_path + '/' + image + '_our.png', 255*Isegmented.astype(int), cmap='gray')

    # C = np.zeros_like(Isegmented).astype(int)
    # a = np.where(np.logical_and(GT, Isegmented)) # TP
    # b = np.where(np.logical_and(GT, np.logical_not(Isegmented))) #FN
    # d = np.where(np.logical_and(Isegmented, np.logical_not(GT))) #FP
    # C[a] = 1
    # C[b] = 2
    # C[d] = 3

    # figure()
    # title('Seg. comparison')
    # imshow(C)
    # savefig(figures_path + '/' + image + '_k_{}_seg_comp.png'.format(k))

    # figure()
    # title('SLIC Segmentation, k = {}'.format(k))
    # imshow(Superpixels)
    # savefig(figures_path + '/' + image + '_k_{}_seg.png'.format(k))

    # figure()
    # title('Merged superpixels')
    # imshow(Islic2)
    # savefig(figures_path + '/' + image + '_k_{}_merged.png'.format(k))

    # figure()
    # title('Otsu')
    # imshow(IOtsu, cmap='gray')
    # savefig(figures_path + '/' + image + '_k_{}_otsu.png'.format(k))

    coords = get_coordinates(Isegmented, 100)
        
    img_final = crop_image(img, coords)

    save_image(image_name, img_final, results_path)

    # print('jaccard overall: {}'.format(np.mean(np.array(all_jaccard))))
    # print('acc. overall: {}'.format(np.mean(np.array(all_acc))))


'''
Clinical Diagnosis:
    1 - Common Nevus;
    2 - Atypical Nevus;
    3 - Melanoma.
'''

path = 'imgs'
path_to_segmentation = '/home/a52550/Desktop/datasets/ISIC2017/train/ISIC-2017_Training_Data_cropped/'
# ph2_dataset_path = './PH2Dataset'
# isic_dataset = './ISIC-2017'
csv_path_train = '/home/a52550/Desktop/datasets/ISIC2017/train/ISIC-2017_Training_Part3_GroundTruth.csv'
imgs_path_train = '/home/a52550/Desktop/datasets/ISIC2017/train/ISIC-2017_Training_Data/'
gtimgs_path_train = '/home/a52550/Desktop/datasets/ISIC2017/train/ISIC-2017_Training_Part1_GroundTruth'
train_csv_folder = pd.read_csv(csv_path_train)
train_imgs_id = train_csv_folder['image_id'].values
train_imgs_path = ["{}{}.jpg".format(imgs_path_train, img_id) for img_id in train_imgs_id]
train_meta_data = None
train_labels = train_csv_folder['category'].values

#Set a class to manage the whole dataset
#dataset = PH2Dataset(ph2_dataset_path)
#dataset = ISICDataset(isic_dataset)

min_ = 0
max_ = 200

# Uncomment the following line to use just images: 'IMD242', 'IMD368', 'IMD306', instead of the whole dataset
#dataset.set_sample(image_names=['IMD242', 'IMD368', 'IMD306'])
init = int(sys.argv[1])
end = int(sys.argv[2])
image_ids = pd.read_csv(csv_path_train)
for _, img in image_ids.loc[init:end,:].iterrows():
    print("Segmenting image: ", img.image_id)
    # img = cv2.imdecode(np.fromfile(imgs_path_train + '/ISIC_0000004.jpg', dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.imread(imgs_path_train + img.image_id +'.jpg', cv2.IMREAD_COLOR)

    img_rgb_2 = morphological_operations(img_rgb)
    img_black_corners = highlight_black_corners(cv.cvtColor(img_rgb_2, cv.COLOR_BGR2GRAY))
    img_black_corners_no_noise = remove_minimal_blobs(img_black_corners)
    coords_1 = get_coordinates(img_black_corners_no_noise, 0, False)
    crop = crop_image(img_rgb_2, coords_1)
    img_rgb = crop_image(img_rgb, coords_1)

    img_gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    img_blue = crop[:,:,0]

    # gtimg = cv2.imdecode(np.fromfile(gtimgs_path_train + '/ISIC_0000004_segmentation.png', dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    segment_image(img=image, results_path=path_to_segmentation, image_name=img.image_id, method='color', k=400)

