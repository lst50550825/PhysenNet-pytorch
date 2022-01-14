import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.fft as fft

def fftshift(x, axs=None):
    shift = (int(x.shape[-2]) // 2, int(x.shape[-1]) // 2)
    result = torch.roll(x, shift, dims=(-2, -1))  # must be tuple
    result = result.type(torch.complex128)  #torch.cdouble
    
    return result

def ifftshift(x):
    shift = (-(int(x.shape[-2]) // 2), -(int(x.shape[-1]) // 2))
    result = torch.roll(x, shift, dims=(-2, -1))
    result = result.type(torch.complex128)
    
    return result


if __name__ == '__main__':
    diffraction_name = 'diff_1.tif'
    measure_temp = cv2.imread(diffraction_name)
    measure_temp = cv2.cvtColor(measure_temp, cv2.COLOR_BGR2GRAY)
    # measure_temp = measure_temp[280:280+dim,225:225+dim]  # crop the ROI
    measure_temp = measure_temp/np.max(measure_temp)
    measure_temp = measure_temp.reshape(1, measure_temp.shape[0], measure_temp.shape[1])
    measure_temp = torch.from_numpy(measure_temp)
    img = measure_temp.unsqueeze(0)

    U_in = torch.exp(img)
    U_out = ifftshift(torch.fft.fft2(fftshift(U_in)))

    print(U_in)
    print(U_out)
