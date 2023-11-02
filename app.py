import cv2
import numpy as np
from PIL import Image


# Implementing Metrics


def psnr(img1, img2):
    # Load images
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)

    # Calculate PSNR
    if mse == 0:
        return float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

def ssim(x, y):
    μ_x = np.mean(x)
    μ_y = np.mean(y)
    s_x = np.std(x)
    s_y = np.std(y)
    s_xy = np.cov(x, y)[0, 1]

    L = (2 ** 8) - 1

    k1 = 0.01
    k2 = 0.03

    C_1 = (k1 * L) ** 2
    C_2 = (k2 * L) ** 2

    SSIM = ((2 * μ_x * μ_y + C_1) * (2 * s_xy + C_2)) / ((μ_x ** 2 + μ_y ** 2 + C_1) * (s_x ** 2 + s_y ** 2 + C_2))

    return SSIM


def ms_ssim(x, y, levels=5):
    result = 0
    for _ in range(levels):
        result *= c(x, y) ** s(x, y)
    
    return result * l(x, y)

def l(x, y):
    u_x = np.mean(x)
    u_y = np.mean(y)

    L = (2 ** 8) - 1

    k1 = 0.01

    return (2 * u_x * u_y + (k1 * L) ** 2) / (u_x ** 2 + u_y ** 2 + (k1 * L) ** 2)  
    
def c(x, y):
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.cov(x, y)[0, 1]

    L = (2 ** 8) - 1

    k2 = 0.03

    return (2 * sigma_xy + (k2 * L) ** 2) / (sigma_x ** 2 + sigma_y ** 2 + (k2 * L) ** 2)

def s(x, y): 
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.cov(x, y)[0, 1]

    L = (2 ** 8) - 1

    k2 = 0.03

    return (2 * sigma_xy + (k2 * L) ** 2) / (2 * sigma_x * sigma_y + (k2 * L) ** 2)



# comparaison

def rmse(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    return np.sqrt(np.mean((img1 - img2) ** 2) / 982)

def pcc(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.cov(img1, img2)[0, 1]

    return sigma_xy / (sigma_x * sigma_y)

def rho(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

   
with open('/home/yassg4mer/Downloads/Py/jp2k/info.txt') as f:
    lines = f.readlines()
    for line in lines:
        image_origin = line.split(' ')[0]
        image_degraded = line.split(' ')[1]

        image_origin_path = '/home/yassg4mer/Downloads/Py/refimgs/' + image_origin
        image_degraded_path = '/home/yassg4mer/Downloads/Py/jp2k/' + image_degraded

        x = np.array(Image.open(image_origin_path).convert("L"), dtype=np.uint8)
        y = np.array(Image.open(image_degraded_path).convert("L"), dtype=np.uint8)

        # Calculate the SSIM between the two images.
        ssim_result = ssim(x, y)
        print(image_origin, image_degraded, ssim_result)


    print(image_origin)

