import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

train_dir = # input images directory 

for img in os.listdir( train_dir):
   
    img_array = cv2.imread(train_dir + img)
    if img_array is not None: 
        img_array = cv2.resize(img_array, (128,128))
        lr_img_array = cv2.resize(img_array,(32,32))

        cv2.imwrite("hr_images/" + img, img_array)
        cv2.imwrite("lr_images/"+ img, lr_img_array)