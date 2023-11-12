import cv2
import numpy as np
import pandas as pd


# Implementing Metrics
    
def psnr(img1, img2, max_pixel_value=255):    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100

    psnr_value = 10 * np.log10((max_pixel_value ** 2) / mse)

    return psnr_value

def ssim(img1, img2):
    μ_x = np.mean(img1)
    μ_y = np.mean(img2)

    s_x = sigma(img1, μ_x)
    s_y = sigma(img2, μ_y)
    s_xy = sigmaXY(img1, img2, μ_x, μ_y)

    L = (2 ** 8) - 1

    k1 = 0.01
    k2 = 0.03

    C_1 = (k1 * L) ** 2
    C_2 = (k2 * L) ** 2

    SSIM = ((2 * μ_x * μ_y + C_1) * (2 * s_xy + C_2)) / ((μ_x ** 2 + μ_y ** 2 + C_1) * (s_x ** 2 + s_y ** 2 + C_2))

    return SSIM

def ms_ssim(img1, img2, max_value=255, num_scales=5):
    scale_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # Adjust these weights as needed

    ssim_values = []
    for scale in range(1, num_scales + 1):
        img1_scaled = _downsample(img1, scale)
        img2_scaled = _downsample(img2, scale)

        ssim_values.append(ssim(img1_scaled, img2_scaled))

    ms_ssim_result = np.prod(np.power(ssim_values, scale_weights))
    return ms_ssim_result

def _downsample(img, scale):
    """Downsample the image by a specified scale."""
    return img[::scale, ::scale]

def sigma(img, mu):
    return np.sqrt(np.mean((img - mu)**2))

def sigmaXY(img1, img2, mu1, mu2):
    return np.mean((img1 - mu1) * (img2 - mu2))

# comparaison

def rmse(img1, img2):
    return np.sqrt(np.mean((img1 - img2) ** 2) / 982)

def pcc(img1, img2):

    mean_X = np.mean(img1)
    mean_Y = np.mean(img2)

    # Calculate covariance
    covariance = np.sum((img1 - mean_X) * (img2 - mean_Y))

    # Calculate standard deviations
    std_X = np.sqrt(np.sum((img1 - mean_X)**2))
    std_Y = np.sqrt(np.sum((img2 - mean_Y)**2))

    # Calculate Pearson Correlation Coefficient
    pcc = covariance / (std_X * std_Y)

    return pcc

def rho(img1, img2):
    n = len(img1)
    
    # Create pairs of ranks
    ranked_data1 = sorted(range(n), key=lambda i: img1[i])
    ranked_data2 = sorted(range(n), key=lambda i: img2[i])

    # Calculate differences between ranks
    d = [rank1 - rank2 for rank1, rank2 in zip(ranked_data1, ranked_data2)]

    # Calculate Spearman's Rank Correlation Coefficient
    rho = 1 - (6 * sum(x**2 for x in d)) / (n * (n**2 - 1))

    return rho

df = pd.read_excel('/home/yassg4mer/Downloads/Py/objective.xlsx', usecols=['ssim_result', 'ms_ssim_result', 'psnr_result', 'subjective_result'] )
print(df['ssim_result'])

rmse_psnr_result_array = []
pcc_psnr_result_array = []
rho_psnr_result_array = []

rmse_ssim_result_array = []
pcc_ssim_result_array = []
rho_ssim_result_array = []

rmse_ms_ssim_result_array = []
pcc_ms_ssim_result_array = []
rho_ms_ssim_result_array = []


for i in range(len(df['psnr_result'])):

    # psnr result
    rmse_psnr_result = rmse(df['psnr_result'], df['subjective_result'])
    pcc_psnr_result = pcc(df['psnr_result'], df['subjective_result'])
    rho_psnr_result = rho(df['psnr_result'], df['subjective_result'])

    # ssim result
    rmse_ssim_result = rmse(df['ssim_result'], df['subjective_result'])
    pcc_ssim_result = pcc(df['ssim_result'], df['subjective_result'])
    rho_ssim_result = rho(df['ssim_result'], df['subjective_result'])

    # ms_ssim result
    rmse_ms_ssim_result = rmse(df['ms_ssim_result'], df['subjective_result'])
    pcc_ms_ssim_result = pcc(df['ms_ssim_result'], df['subjective_result'])
    rho_ms_ssim_result = rho(df['ms_ssim_result'], df['subjective_result'])

    rmse_psnr_result_array.append(rmse_psnr_result)
    pcc_psnr_result_array.append(pcc_psnr_result)
    rho_psnr_result_array.append(rho_psnr_result)

    rmse_ssim_result_array.append(rmse_ssim_result)
    pcc_ssim_result_array.append(pcc_ssim_result)
    rho_ssim_result_array.append(rho_ssim_result)

    rmse_ms_ssim_result_array.append(rmse_ms_ssim_result)
    pcc_ms_ssim_result_array.append(pcc_ms_ssim_result)
    rho_ms_ssim_result_array.append(rho_ms_ssim_result)


dff = pd.DataFrame({ # psnr
                    'rmse_psnr_result': rmse_psnr_result_array, 
                    'pcc_psnr_result': pcc_psnr_result_array,
                    'rho_psnr_result': rho_psnr_result_array, 
                    # ssim
                    'rmse_ssim_result': rmse_ssim_result_array,
                    'pcc_ssim_result': pcc_ssim_result_array,
                    'rho_ssim_result': rho_ssim_result_array,
                    # ms_ssim
                    'rmse_ms_ssim_result': rmse_ms_ssim_result_array,
                    'pcc_ms_ssim_result': pcc_ms_ssim_result_array,
                    'rho_ms_ssim_result': rho_ms_ssim_result_array,})

dff.to_excel('comparaison.xlsx', index=False)

# with open('/home/yassg4mer/Downloads/Py/jp2k/info.txt') as f:
#     lines = f.readlines()

#     image_origin_array = []
#     image_degraded_array = []

#     ssim_result_array = []
#     ms_ssim_result_array = []
#     psnr_result_array = []

#     for line in lines:
#         image_origin = line.split(' ')[0]
#         image_degraded = line.split(' ')[1]

#         image_origin_array.append(image_origin)
#         image_degraded_array.append(image_degraded)

#         image_origin_path = '/home/yassg4mer/Downloads/Py/refimgs/' + image_origin
#         image_degraded_path = '/home/yassg4mer/Downloads/Py/jp2k/' + image_degraded

#         x = cv2.imread(image_origin_path, cv2.IMREAD_GRAYSCALE)
#         y = cv2.imread(image_degraded_path, cv2.IMREAD_GRAYSCALE)

#         x = x.astype(np.float64)
#         y = y.astype(np.float64)

#         # Calculate the SSIM between the two images.
#         ssim_result = ssim(x, y)
#         ms_ssim_result = ms_ssim(x, y)
#         psnr_result = psnr(x, y)

#         ms_ssim_result_array.append(ms_ssim_result)
#         ssim_result_array.append(ssim_result)
#         psnr_result_array.append(psnr_result)

#         print(image_origin, image_degraded, ssim_result, ms_ssim_result, psnr_result)
        
#     # add multiple columns to dataframe at once 
#     df = pd.DataFrame({'image_origin': image_origin_array, 
#                         'image_degraded': image_degraded_array, 
#                         'ssim_result': ssim_result_array, 
#                         'ms_ssim_result': ms_ssim_result_array, 
#                         'psnr_result': psnr_result_array
#                         })
#     df.to_excel('objective.xlsx', index=False)




#     print(image_origin)

