from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.models import load_model
import numpy as np


def generate_latent_points(latent_dim, n_samples, n_classes=10):
    x_input = randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


def show_plot(examples, n, count):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, :])
    plt.savefig(f"conditional_images/cond_gan_test_image_{count}.png")
    
    
start = 0
end = start + 10
model = load_model('cifar100_conditional_gan_generator_v3_100_epoch_model.h5')

def create_test_results(start, end, model): 
    for _ in range(10):
        labels = asarray([x for _ in range(start, end) for x in range(10)])
        latent_points, labels = generate_latent_points(100, 100)
        X  = model.predict([latent_points, labels])   
        X = (X + 1) / 2.0
        X = (X*255).astype(np.uint8)
        show_plot(X, 10, _)
        start = end 
        end = start + 10
        
create_test_results(start, end, model)