import torch as t
import numpy as np
from PIL import Image  
import matplotlib.pyplot as plt
import os
from torch import nn
import cv2
from torch.autograd import Variable

from models import unet_models
from physics_model import Measure
import sys 
sys.path.append('..')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dim = 440
device = t.device('cuda')

# Load Image
diffraction_name = 'diff_1.tif'
measure_temp = cv2.imread(diffraction_name)
measure_temp = cv2.cvtColor(measure_temp, cv2.COLOR_BGR2GRAY)
measure_temp = measure_temp[280:280+dim,225:225+dim]  # crop the ROI
measure_temp = measure_temp/np.max(measure_temp)
measure_temp = measure_temp.reshape(1, measure_temp.shape[0], measure_temp.shape[1])
measure_temp = t.from_numpy(measure_temp)
measure_temp = measure_temp.unsqueeze(0)
measure_temp.to(device=device)


# network = unet_model.UNet(n_channels=1, n_classes=1)
network = unet_models.UNet(n_channels=1, n_classes=1)
network.eval()
network.to(device=device)
#network.load_state_dict(t.load('unet_best_model.pth', map_location=device))
network.load_state_dict(t.load('raw_best_model.pth', map_location=device))
pred = network(measure_temp.to(device, dtype=t.float32))
pred = pred.cpu().detach().numpy()

plt.imshow(pred[0,0], cmap='gray')
# plt.pause(60)
plt.savefig("origin.jpg")
