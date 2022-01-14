import torch
import numpy as np
import math
import os 
import sys
sys.path.append(os.getcwd())
from matplotlib import pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from physics_model.Myfftshift import *

import torch
import math
# import physics_model.Myfftshift


def AS(inpt,lamb,L,Z):
    device = torch.device('cuda')
    M = int(inpt.shape[1]) 
    
    # image = torch.tensor(inpt,dtype = torch.complex128)
    image = 1j*inpt            
    U_in = torch.exp(image)             

    U_out = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(U_in)))
    
    fx=1/L
    
    x = np.linspace(-M/2,M/2-1,M) 
    fx = fx*x                           
    [Fx,Fy]=np.meshgrid(fx,fx)
    
    k = 2*math.pi/lamb 
    H = np.sqrt(1-lamb*lamb*(Fx*Fx+Fy*Fy))
    temp = k*Z*H
    temp = torch.tensor(temp,dtype = torch.complex64)
    
    H = torch.exp(1j*temp)
    H = H.to(device)
    U_out = U_out*H
        
    U_out = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(U_out)))    
    I1 = torch.abs(U_out) * torch.abs(U_out)   
    I1 = I1/torch.max(torch.max(I1))
    
    return I1


if __name__ == '__main__':
    dim = 440
    img_W = dim
    img_H = dim
    batch_size = 1

    lamb = 632.8e-6                  # wavelength
    pixelsize = 8e-3                 # pixel size
    N = dim                          # num of pixels
    L = N*pixelsize                  # length of the object and image plane

    Steps = 5000                     # iteration steps
    LR = 0.01                        # learning rate
    Z = 22.3                         # diffraction distance mm
    noise_level = 1.0/30
    
    diffraction_name = './physics_model/raw.bmp'
    # measure_temp = np.array(cv2.imread(diffraction_name))
    measure_temp = np.array(plt.imread(diffraction_name))

    
    # measure_temp = cv2.cvtColor(measure_temp, cv2.COLOR_BGR2GRAY)
    # measure_temp = measure_temp[280:280+dim,225:225+dim]  # crop the ROI
    measure_temp = measure_temp/np.max(measure_temp)
    measure_temp = measure_temp.reshape(1, measure_temp.shape[0], measure_temp.shape[1])
    measure_temp = torch.from_numpy(measure_temp)
    img = measure_temp.unsqueeze(0)
    img = img.to(device=torch.device("cuda"))
    
    

    out = AS(img[0,0], lamb,L,Z)
    out = out.cpu().numpy()
    plt.imshow(out)
    plt.pause(60)

