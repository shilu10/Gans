import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.datasets import cifar100
import random 
import numpy as np


BATCH_SIZE = 64
NUM_CHANNELS = 3
NUM_CLASSES = 100
IMAGE_SIZE = 32
LATENT_DIM = 100
EMBEDDING_LAYER_DIM = 50 
ALPHA_LEAKY_RELU = 0.2
OPTIMIZER = "adam"
DIS_LEARNING_RATE = 0.0002
DIS_BETA_ONE = 0.5
GAN_LEARNING_RATE = 0.0001
GAN_BETA_ONE = 0.4
DROPOUT_RATE = 0.4 
EPOCHS = 500


def load_real_data(): 
    try: 
        (train_X, train_y), (test_X, test_y) = cifar100.load_data()
        return train_X, train_y 
    
    except Exception as error:
        return error

train_X, train_y = load_real_data()
print("Training Data Shape: ", train_X.shape)


def normalize_input(X: np.ndarray, is_use_tanh: bool=False) -> np.ndarray: 
    try: 
        if is_use_tanh: 
            X = X.reshape(-1, 32, 32, 1).astype('float32')
            X = (X - 127.5) / 127.5
        else: 
            X = X / 255.0
        return X
    
    except Exception as error: 
        return error

train_X = normalize_input(train_X, is_use_tanh=True)

  
def generate_real_samples(dataset, n_samples):
    try: 
        images, labels = dataset  
        ix = randint(0, images.shape[0], n_samples)
        X, labels = images[ix], labels[ix]
        y = ones((n_samples, 1)) 
        return [X, labels], y
    
    except Exception as error:
        return error


def generate_latent_points(latent_dim, n_samples, n_classes=100): 
    try: 
        x_input = randn(latent_dim * n_samples)
        z_input = x_input.reshape(n_samples, latent_dim)
        labels = randint(0, n_classes, n_samples)
        return [z_input, labels]
    
    except Exception as error:
        return error


def generate_fake_samples(generator, latent_dim, n_samples):
    try: 
        z_input, labels_input = generate_latent_points(latent_dim, n_samples)
        images = generator.predict([z_input, labels_input])
        y = zeros((n_samples, 1))  
        return [images, labels_input], y

    except Exception as error:
        return error

    
class ANNBlock(tf.keras.layers.Layer): 
    def __init__(self, alpha: int, units: int, dropout_rate: float): 
        super(ANNBlock, self).__init__()
        self.Dense = tf.keras.layers.Dense(units=units)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.alpha = alpha

    def call(self, input_tensor, training=False): 
        x = self.Dense(input_tensor)
        x = tf.nn.leaky_relu(x, self.alpha)
        #x = self.dropout(x)
        return x

    
class CNNTransposeBlock(tf.keras.layers.Layer): 
    def __init__(self, kernel_shape: tuple, channels: int, strides: tuple, padding: str, use_bias: bool, alpha: float): 
        super(CNNTransposeBlock, self).__init__()
        self.conv2d_trans = tf.keras.layers.Conv2DTranspose(filters=channels, kernel_size=kernel_shape, strides=strides, padding=padding, use_bias=use_bias)
        self.alpha = alpha

    def call(self, input_tensor, training=False):
        x = self.conv2d_trans(input_tensor)
        x = tf.nn.leaky_relu(x, self.alpha)
        return x 


class CNNBlock(tf.keras.layers.Layer): 
    def __init__(self, kernel_shape: tuple, channels: int, strides: tuple, padding: str, use_bias: bool, alpha: float): 
        super(CNNBlock, self).__init__()
        self.conv2d = tf.keras.layers.Conv2D(filters=channels, kernel_size=kernel_shape, strides=strides, padding=padding, use_bias=use_bias)
        self.alpha = alpha

    def call(self, input_tensor, training=False):
        x = self.conv2d(input_tensor)
        x = tf.nn.leaky_relu(x, self.alpha)
        return x 
    
    
def create_discriminator(img_shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), num_classes=NUM_CLASSES): 
    
    input_label = Input(shape=(1, ))
    li = Embedding(num_classes, EMBEDDING_LAYER_DIM)(input_label)
    n_nodes = img_shape[0] * img_shape[1]
    li = Dense(n_nodes)(li)
    li = Reshape((img_shape[0], img_shape[1], 1))(li)
    
    input_image = Input(shape=img_shape)
    merge = Concatenate()([input_image, li])
    
    conv_lay = CNNBlock((3, 3), 128, (2, 2), "same", True, ALPHA_LEAKY_RELU)(merge)
    conv_lay = CNNBlock((3, 3), 128, (2, 2), "same", True, ALPHA_LEAKY_RELU)(conv_lay)
    
    flatten = Flatten()(conv_lay)
    dn_lay = ANNBlock(ALPHA_LEAKY_RELU, 256, 0.4)(flatten)
    output_layer = Dense(1, activation="sigmoid")(dn_lay)
    
    model = Model([input_image, input_label], output_layer)
    opt = Adam(learning_rate=DIS_LEARNING_RATE, beta_1=DIS_BETA_ONE)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
    

def create_generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES): 
    input_label = Input(shape=(1,))
    li = Embedding(num_classes, EMBEDDING_LAYER_DIM)(input_label)
    
    n_nodes = 8 * 8 
    li = Dense(n_nodes)(li)
    li = Reshape((8, 8, 1))(li)
    
    input_lat = Input(shape=(latent_dim,))
    n_nodes = 128 * 8 * 8
    gen = ANNBlock(ALPHA_LEAKY_RELU, n_nodes, 0.4)(input_lat)
    gen = Reshape((8, 8, 128))(gen)
    
    merge = Concatenate()([gen, li])
    upsam_lay = CNNTransposeBlock((6, 6), 256, (2, 2), "same", True, ALPHA_LEAKY_RELU)(merge)
    upsam_lay = CNNTransposeBlock((6, 6), 256, (2, 2), "same", True, ALPHA_LEAKY_RELU)(upsam_lay)
    
    conv_lay = CNNBlock((6, 6), 3, (1, 1), "same", True, 0.2)(upsam_lay)
    output_lay = Conv2D(3, (8,8), activation='tanh', padding='same')(conv_lay)
    model = Model([input_lat, input_label], output_lay)
    return model
    

def create_gan(generator, discriminator): 
    discriminator.trainable = False
    
    gen_noise, gen_label = generator.input
    gen_output = generator.output
    
    gan_output = discriminator([gen_output, gen_label])
    model = Model([gen_noise, gen_label], gan_output)
    
    opt = Adam(learning_rate=GAN_LEARNING_RATE, beta_1=GAN_BETA_ONE)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    
    return model
    

def train(gen_model, dis_model, gan_model, dataset, epochs, batch_size, filename="cifar100_conditional_gan_v2.h5"): 
    try: 
        bat_per_epo = int(dataset[0].shape[0] / batch_size)
        half_batch = int(batch_size / 2)

        for epoch in range(epochs):
            d_loss = 0
            ge_loss = 0
            print(f"Epochs: {epoch}/{epochs}")

            for batch in range(bat_per_epo): 

                [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
                dis_loss_real, _ = dis_model.train_on_batch([X_real, labels_real], y_real)

                [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
                dis_loss_fake, _ = dis_model.train_on_batch([X_fake, labels], y_fake)

                [z_input, labels_input] = generate_latent_points(latent_dim, batch_size)
                y_gan = ones((batch_size, 1))
                g_loss, _ = gan_model.train_on_batch([z_input, labels_input], y_gan)

              

        gen_model.save(filename)
        
    except Exception as error:
        return error
            
        
dataset = load_real_data()
gen_model = create_generator()
dis_model = create_discriminator()
gan_model = create_gan(gen_model, dis_model)
train(gen_model, dis_model, gan_model, dataset, EPOCHS, BATCH_SIZE)