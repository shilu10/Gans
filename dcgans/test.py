import s3fs
from tensorflow.keras.models import load_model 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
import os
import pandas as pd 


if not os.path.exists("images/"): 
    os.mkdir("images/")


fs = s3fs.S3FileSystem(key=os.environ["aws_access_key"], secret=os.environ["aws_secret_key"])
fs.get("/name-of-my-bucket/generator_model.h5", "models/")

generator = load_model("models/generator_model.h5")

def save_images(_):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)

    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(f"images/test_images_{_}.png")
    plt.close()
    
for _ in range(10): 
    save_images(_)