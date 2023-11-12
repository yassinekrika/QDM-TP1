import tkinter
from tkinter import *
from tkinter import filedialog
import numpy as np
import cv2
import pandas as pd
import os


class App(tkinter.Tk):

    image1 = None
    image2 = None

    targer_folder = None

    def __init__(self):
        super().__init__()
        self.title("TP1")
        self.geometry("500x500")
        self.maxsize(500, 500)
        self.minsize(500, 500)

        self.frame = Frame(self, bg='#41B77F')

        self.image1_label = Label(self, text="Image 1:")
        self.image1_label.grid(row=0, column=1, padx=10, pady=10)
        
        self.image1_button = Button(self, text="Upload Image 1", command=self.upload_image1)
        self.image1_button.grid(row=0, column=0, padx=10, pady=10)

        self.image2_label = Label(self, text="Image 2:")
        self.image2_label.grid(row=1, column=1, padx=10, pady=10)

        self.image2_button = Button(self, text="Upload Image 2", command=self.upload_image2)
        self.image2_button.grid(row=1, column=0, padx=10, pady=10)

        self.clear_button = Button(self, text="Clear Images", command=self.clear_images)
        self.clear_button.grid(row=2, column=0, pady=10)

        self.psnr_button = Button(self, text="PSNR", command=self.calcualte_psnr)
        self.psnr_button.grid(row=4, column=0, pady=10)

        self.ssim_button = Button(self, text="SSIM", command=self.calulate_ssim)
        self.ssim_button.grid(row=5, column=0, pady=10)

        self.ms_ssim_button = Button(self, text="MS-SSIM", command=self.caculate_ms_ssim)
        self.ms_ssim_button.grid(row=6, column=0, pady=10)

        # add two buttons 
        self.objective_result_button = Button(self, text="Objective Result", command=self.calculate_objective_result)
        self.objective_result_button.grid(row=7, column=0, pady=10)

        self.comparaison_button = Button(self, text="Make Comparaison", command=self.calculate_comparaison)
        self.comparaison_button.grid(row=8, column=0, pady=10)



    def upload_image1(self):
        self.image1 = filedialog.askopenfilename(initialdir = "/home/yassg4mer/Downloads/Py/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        self.image1_label = Label(self, text=self.image1)
        self.image1_label.grid(row=0, column=1, padx=10, pady=10)

        self.image1 = cv2.imread(self.image1, cv2.IMREAD_GRAYSCALE)
        self.image1 = self.image1.astype(np.float64)
        

    def upload_image2(self):
        self.image2 = filedialog.askopenfilename(initialdir = "/home/yassg4mer/Downloads/Py/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        self.image2_label = Label(self, text=self.image2)
        self.image2_label.grid(row=1, column=1, padx=10, pady=10)

        self.image2 = cv2.imread(self.image2, cv2.IMREAD_GRAYSCALE)
        self.image2 = self.image2.astype(np.float64)

    def clear_images(self):
        self.image1_label.destroy()
        self.image2_label.destroy()

    def psnr(self, img1, img2, max_pixel_value=255): 
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100

        psnr_value = 10 * np.log10((max_pixel_value ** 2) / mse)

        return psnr_value

    def calcualte_psnr(self):
        psnr = self.psnr(self.image1, self.image2)

        self.psnr_label = Label(self, text=str(psnr),) 
        self.psnr_label.grid(row=4, column=1, pady=10)

    def ssim(self, img1, img2):
        mu_x = np.mean(img1)
        mu_y = np.mean(img2)

        s_x = self.sigma(img1, mu_x)
        s_y = self.sigma(img2, mu_y)
        s_xy = self.sigmaXY(img1, img2, mu_x, mu_y)

        L = (2 ** 8) - 1

        k1 = 0.01
        k2 = 0.03

        C_1 = (k1 * L) ** 2
        C_2 = (k2 * L) ** 2

        SSIM = ((2 * mu_x * mu_y + C_1) * (2 * s_xy + C_2)) / ((mu_x ** 2 + mu_y ** 2 + C_1) * (s_x ** 2 + s_y ** 2 + C_2))

        return SSIM

    def sigma(self, img, mu):
        return np.sqrt(np.mean((img - mu)**2))

    def sigmaXY(self, img1, img2, mu1, mu2):
        return np.mean((img1 - mu1) * (img2 - mu2))
    
    def calulate_ssim(self):
        ssim = self.ssim(self.image1, self.image2)

        self.ssim_label = Label(self, text=str(ssim)) 
        self.ssim_label.grid(row=5, column=1, pady=10)

    def ms_ssim(self, img1, img2, max_value=255, num_scales=5):
        scale_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # Adjust these weights as needed

        ssim_values = []
        for scale in range(1, num_scales + 1):
            img1_scaled = self._downsample(img1, scale)
            img2_scaled = self._downsample(img2, scale)

            ssim_values.append(self.ssim(img1_scaled, img2_scaled))

        ms_ssim_result = np.prod(np.power(ssim_values, scale_weights))
        return ms_ssim_result

    def _downsample(self, img, scale):
        """Downsample the image by a specified scale."""
        return img[::scale, ::scale]

    def caculate_ms_ssim(self):
        ms_ssim = self.ms_ssim(self.image1, self.image2)

        self.ms_ssim_label = Label(self, text=str(ms_ssim),) 
        self.ms_ssim_label.grid(row=6, column=1, pady=10)

    def calculate_objective_result(self):
        self.targer_folder = filedialog.askdirectory(initialdir = "/home/yassg4mer/Downloads/Py/",title = "Select folder")
        # print(self.targer_folder)
        self.open_folder()

    def open_folder(self):

        info_file = self.targer_folder + '/info.txt'
        with open(info_file) as f:
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
                image_degraded_path = self.targer_folder + '/' + image_degraded
                print(image_degraded_path)

                x = cv2.imread(image_origin_path, cv2.IMREAD_GRAYSCALE)
                y = cv2.imread(image_degraded_path, cv2.IMREAD_GRAYSCALE)



                x = x.astype(np.float64)
                y = y.astype(np.float64)

                # Calculate the SSIM between the two images.
                ssim_result = self.ssim(x, y)
                ms_ssim_result = self.ms_ssim(x, y)
                psnr_result = self.psnr(x, y)

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
            df.to_excel('objective.xlsx', index=False)

            # open objective.xlsx file
            os.system('libreoffice --calc objective.xlsx')

    def rmse(self, img1, img2):
        return np.sqrt(np.mean((img1 - img2) ** 2) / 982)

    def pcc(self, img1, img2):

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

    def rho(self, img1, img2):
        n = len(img1)
        
        # Create pairs of ranks
        ranked_data1 = sorted(range(n), key=lambda i: img1[i])
        ranked_data2 = sorted(range(n), key=lambda i: img2[i])

        # Calculate differences between ranks
        d = [rank1 - rank2 for rank1, rank2 in zip(ranked_data1, ranked_data2)]

        # Calculate Spearman's Rank Correlation Coefficient
        rho = 1 - (6 * sum(x**2 for x in d)) / (n * (n**2 - 1))

        return rho

    def calculate_comparaison(self):
        self.open_comparaison_folder()
    
    def open_comparaison_folder(self):
        df = pd.read_excel('/home/yassg4mer/Downloads/Py/objective.xlsx', usecols=['ssim_result', 'ms_ssim_result', 'psnr_result'] )
        dff = pd.read_excel('/home/yassg4mer/Downloads/Py/subjective.xlsx', usecols=['subjective_result'] )


        rmse_psnr_result_array = []
        pcc_psnr_result_array = []
        rho_psnr_result_array = []

        rmse_ssim_result_array = []
        pcc_ssim_result_array = []
        rho_ssim_result_array = []

        rmse_ms_ssim_result_array = []
        pcc_ms_ssim_result_array = []
        rho_ms_ssim_result_array = []


        # psnr result
        rmse_psnr_result = self.rmse(df['psnr_result'], dff['subjective_result'])
        pcc_psnr_result = self.pcc(df['psnr_result'], dff['subjective_result'])
        rho_psnr_result = self.rho(df['psnr_result'], dff['subjective_result'])

        # ssim result
        rmse_ssim_result = self.rmse(df['ssim_result'], dff['subjective_result'])
        pcc_ssim_result = self.pcc(df['ssim_result'], dff['subjective_result'])
        rho_ssim_result = self.rho(df['ssim_result'], dff['subjective_result'])

        # ms_ssim result
        rmse_ms_ssim_result = self.rmse(df['ms_ssim_result'], dff['subjective_result'])
        pcc_ms_ssim_result = self.pcc(df['ms_ssim_result'], dff['subjective_result'])
        rho_ms_ssim_result = self.rho(df['ms_ssim_result'], dff['subjective_result'])

        rmse_psnr_result_array.append(rmse_psnr_result)
        pcc_psnr_result_array.append(pcc_psnr_result)
        rho_psnr_result_array.append(rho_psnr_result)

        rmse_ssim_result_array.append(rmse_ssim_result)
        pcc_ssim_result_array.append(pcc_ssim_result)
        rho_ssim_result_array.append(rho_ssim_result)

        rmse_ms_ssim_result_array.append(rmse_ms_ssim_result)
        pcc_ms_ssim_result_array.append(pcc_ms_ssim_result)
        rho_ms_ssim_result_array.append(rho_ms_ssim_result)

        dfff = pd.DataFrame({ # psnr
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

        dfff.to_excel('comparaison.xlsx', index=False)


        os.system('libreoffice --calc comparaison.xlsx')

    def quit(self):
        self.destroy()

if __name__=="__main__":
    app = App()
    app.mainloop()
