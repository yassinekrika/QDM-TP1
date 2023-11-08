import cv2
import numpy as np
from PIL import Image
import openpyxl
import pandas as pd
import math


# Implementing Metrics

def psnr(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")

    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / float(np.sqrt(mse)))
        return psnr

def Psnr(ImageOrig,ImageD):

    print("PSNR Globale")
    ResultatPSNR = 0
    Somme = 0
    
    L,H=ImageOrig.shape
    MatOrig=np.array(ImageOrig)
    MatD=np.array(ImageD)
    
    for i in range(H):        
        for j in range(L):
            
            Somme = Somme +((int(MatOrig[i,j]) - int(MatD[i,j]))**2)

  
    Max2 = 255**2       
    MSE = (Somme/float(L*H)) 
    M = Max2/float(MSE)     
    ResultatPSNR= 20 * math.log10(M)
    return ResultatPSNR

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

# def ssim(img1, img2): 
#     # Constants for SSIM calculation
#     C1 = (0.01 * 255) ** 2
#     C2 = (0.03 * 255) ** 2

#     # Mean of the images
#     mu1 = np.mean(img1)
#     mu2 = np.mean(img2)

#     # Variance of the images
#     sigma1_sq = np.var(img1)
#     sigma2_sq = np.var(img2)

#     # Covariance between the images
#     sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]

#     # SSIM calculation
#     numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
#     denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

#     ssim = numerator / denominator

#     return ssim

def ms_ssim(x, y, levels=5):
    result = 0
    for _ in range(levels):
        result *= c(x, y) * s(x, y)
    
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

# def mean(img):
#     mean = 0
#     w, h = img.shape
#     n = 0

#     for i in range(w):
#         for j in range(h):
#             mean += img[i][j]
#             n += 1

#     return mean / float(n)

# def sigma(img):
#     sigma = 0
#     n = 0
#     w, h = img.shape
#     u = mean(img)

#     for i in range(w):
#         for j in range(h):

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

    image_origin_array = []
    image_degraded_array = []
    ssim_result_array = []
    ms_ssim_result_array = []
    psnr_result_array = []

    for line in lines:
        image_origin = line.split(' ')[0]
        image_degraded = line.split(' ')[1]

        image_origin_array.append(image_origin)
        image_degraded_array.append(image_degraded)

        image_origin_path = '/home/yassg4mer/Downloads/Py/refimgs/' + image_origin
        image_degraded_path = '/home/yassg4mer/Downloads/Py/jp2k/' + image_degraded

        # x = np.array(Image.open(image_origin_path).convert("L"))
        # y = np.array(Image.open(image_degraded_path).convert("L"))

        x = cv2.imread(image_origin_path)
        y = cv2.imread(image_degraded_path)

        # convert into gray scale
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)

        # Calculate the SSIM between the two images.
        ssim_result = ssim(x, y)
        ms_ssim_result = ms_ssim(x, y)
        psnr_result = psnr(x, y)

        ms_ssim_result_array.append(ms_ssim_result)
        ssim_result_array.append(ssim_result)
        psnr_result_array.append(psnr_result)

        print(image_origin, image_degraded, ssim_result, ms_ssim_result, psnr_result)
        
    # add multiple columns to dataframe at once 
    df = pd.DataFrame({'image_origin': image_origin_array, 
                        'image_degraded': image_degraded_array, 
                        'ssim_result': ssim_result_array, 
                        'ms_ssim_result': ms_ssim_result_array, 
                        'psnr_result': psnr_result_array
                        })
    df.to_excel('ssim.xlsx', index=False)




    print(image_origin)

