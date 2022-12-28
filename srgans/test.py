import matplotlib.pyplot as plt
import cv2 
import numpy as np
from random import randint

def test(img_no, count): 
    print(f"lr_images/im{img_no}.jpg")
    img_arr_l = cv2.imread(f"lr_images/im{img_no}.jpg")
    img_arr_h = cv2.imread(f"hr_images/im{img_no}.jpg")
    img_arr_l = cv2.cvtColor(img_arr_l, cv2.COLOR_BGR2RGB)
    img_arr_h = cv2.cvtColor(img_arr_h, cv2.COLOR_BGR2RGB)
    img_arr_l = img_arr_l / 255.
    img_arr_h = img_arr_h / 255.
    img_arr_l = np.expand_dims(img_arr_l, axis=0)
    img_arr_h = np.expand_dims(img_arr_h, axis=0)
    model_prediction_img = model.predict(img_arr_l)
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    axes[0].imshow(img_arr_l.reshape(32, 32, 3))
    axes[1].imshow(img_arr_h.reshape(128, 128, 3))
    axes[2].imshow(model_prediction_img.reshape(128, 128, 3))
    plt.savefig(f"images/test_image_srgan_{img_no}.png")
    
for _ in range(10): 
    test(randint(1, 7000), _)