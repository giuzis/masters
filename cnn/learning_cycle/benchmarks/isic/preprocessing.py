import cv2
import numpy as np

class PreProcessing(object):
    def __init__(self, pp_enhancement = None, pp_hair_removal = None, pp_color_constancy = None, pp_denoising = None):
        self.pp_enhancement = pp_enhancement
        self.pp_hair_removal = pp_hair_removal
        self.pp_color_constancy = pp_color_constancy
        self.pp_denoising = pp_denoising

    def __call__(self, img):
        img = image_preprocessing(np.array(img), pp_enhancement=self.pp_enhancement, pp_hair_removal=self.pp_hair_removal, pp_color_constancy=self.pp_color_constancy, pp_denoising=self.pp_denoising) 
        return img


def dullrazor(img):
    #Gray scale
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
    #Black hat filter
    kernel = cv2.getStructuringElement(1,(9,9)) 
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    #Gaussian filter
    bhg= cv2.GaussianBlur(blackhat,(3,3),cv2.BORDER_DEFAULT)
    #Binary thresholding (MASK)
    ret,mask = cv2.threshold(bhg,10,255,cv2.THRESH_BINARY)
    #Replace pixels of the mask
    dst = cv2.inpaint(img,mask,6,cv2.INPAINT_TELEA)   
    return dst

def shade_of_gray_cc(img, power=6, gamma=None):
    """
    img (numpy array): the original image with format of (h, w, c)
    power (int): the degree of norm, 6 is used in reference paper
    gamma (float): the value of gamma correction, 2.2 is used in reference paper
    """
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256,1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i/255, 1/gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0,1)), 1/power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    img = np.clip(img, a_min=0, a_max=255)
    
    return img.astype(img_dtype)

def contrast_enhancement(img):
    # Converte a imagem para o espaço de cor YCrCb
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    # Separa os canais Y, Cr, e Cb
    y, cr, cb = cv2.split(img_ycrcb)

    # Calcula o histograma de cada canal
    hist_y = cv2.calcHist([y], [0], None, [256], [0, 256])
    hist_cr = cv2.calcHist([cr], [0], None, [256], [0, 256])
    hist_cb = cv2.calcHist([cb], [0], None, [256], [0, 256])

    # Normaliza cada histograma
    # hist_y_norm = cv2.normalize(hist_y, hist_y, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # hist_cr_norm = cv2.normalize(hist_cr, hist_cr, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # hist_cb_norm = cv2.normalize(hist_cb, hist_cb, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Aplica a equalização de histograma em cada canal
    y_eq = cv2.equalizeHist(y)
    cr_eq = cv2.equalizeHist(cr)
    cb_eq = cv2.equalizeHist(cb)

    # Junta os canais de volta
    img_ycrcb_eq = cv2.merge((y_eq, cr_eq, cb_eq))

    # Converte a imagem de volta para o espaço de cor BGR
    img_eq = cv2.cvtColor(img_ycrcb_eq, cv2.COLOR_YCrCb2RGB)

    return img_eq

def image_preprocessing(img, pp_enhancement = None, pp_hair_removal = None, pp_color_constancy = None, pp_denoising = None):
    if pp_enhancement == "CLAHE":
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(9,9))
        img[:, :, 0] = clahe.apply(img[:, :, 0])
        img[:, :, 1] = clahe.apply(img[:, :, 1])
        img[:, :, 2] = clahe.apply(img[:, :, 2])
    if pp_color_constancy == "shades_of_gray":
        img = shade_of_gray_cc(img)
    if pp_denoising == "non_local_means":
        img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    if pp_denoising == "gaussian_filter":
        img = cv2.GaussianBlur(img,(5,5),0)
    if pp_denoising == "mean_filter":
        img = cv2.blur(img,(5,5))
    if pp_hair_removal == "dull_razor":
        img = dullrazor(img)
    return img

