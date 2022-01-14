import torch as t
import numpy as np
from PIL import Image  
import matplotlib.pyplot as plt
import os
from torch import nn
import cv2
from torch.autograd import Variable
import sys 

from models import unet_models
from physics_model import Measure

sys.path.append('..')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Parameters
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

# Create Directory
def mkdir(path): 
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
resut_save_path = '.\\results\\'
model_save_path = '.\\models\\'
mkdir(resut_save_path)
mkdir(model_save_path)
model_save_path =  model_save_path + 'exp_step_%d_lr_%f.ckpt'%(Steps,LR) 

# Load Raw Data
diffraction_name = 'diff_1.tif'
measure_temp = cv2.imread(diffraction_name)
measure_temp = cv2.cvtColor(measure_temp, cv2.COLOR_BGR2GRAY)
measure_temp = measure_temp[280:280+dim,225:225+dim]  # crop the ROI
measure_temp = measure_temp/np.max(measure_temp)
measure_temp = measure_temp.reshape(1, measure_temp.shape[0], measure_temp.shape[1])
measure_temp = t.from_numpy(measure_temp)
measure_temp = measure_temp.unsqueeze(0)

# Train
def train(model, data, device, epochs=500):
    print("start train")
    net = model
    net = net.cuda()
    criterion = nn.MSELoss()
    best_loss = float('inf')
    optimizer = t.optim.Adam(net.parameters(), lr=0.0005, betas=[0.5, 0.999])
    
    # Training Iteration
    for epoch in range(epochs):
        # Training Mode
        net.train()
        optimizer.zero_grad()
        data = data.to(device=device, dtype=t.float32)  # copy data to device
        # data = Variable(data, requires_grad=True)
        pred = net(data)    # use nerwork to predict
        # print(pred[0,0].size())
        simulation = Measure.AS(pred[0,0], lamb, L, Z)  # physical forward model
        
        # Calculate Loss
        # print(simulation.size())
        # print(data.size())
        loss = criterion(simulation, data[0,0])
        
        # Save Network With The Minimum Loss 
        if loss < best_loss:
            best_loss = loss
            # save model parameter Unet/GAN/...
            # t.save(net.state_dict(), 'gan_best_model.pth')
            t.save(net.state_dict(), 'unet_best_model.pth')
        
        # Update Parameter
        loss.backward()
        optimizer.step()
        
        # Show Loss
        if epoch % 100 == 0:
            print('train:epoch=>%d, loss = %f' % (epoch, loss.item()))   
        
        # TODO Visualization

        
        
if __name__ == '__main__':
    import time
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print("data loaded")
    # network = unet_model.UNet(n_channels=1, n_classes=1)
    network = unet_models.UNet(n_channels=1, n_classes=1)
    
    print("model loaded")
    t.cuda.synchronize()    # time
    t0 = time.time()
    train(network, measure_temp, device, Steps)
    t.cuda.synchronize()
    t1 = time.time()
    print(t1-t0)

    