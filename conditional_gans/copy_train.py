from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.cifar100 import load_data
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

from matplotlib import pyplot as plt

def define_discriminator(in_shape=(32, 32, 3), n_classes=100):
    in_label = Input(shape=(1,)) 
    li = Embedding(n_classes, 50)(in_label) 
    n_nodes = in_shape[0] * in_shape[1]   
    li = Dense(n_nodes)(li)  
    li = Reshape((in_shape[0], in_shape[1], 1))(li) 

    in_image = Input(shape=in_shape) 
    merge = Concatenate()([in_image, li]) 
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe) 
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Flatten()(fe) 
    fe = Dropout(0.4)(fe)
    out_layer = Dense(1, activation='sigmoid')(fe)  
    model = Model([in_image, in_label], out_layer)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def define_generator(latent_dim, n_classes=100):

    in_label = Input(shape=(1,))

    li = Embedding(n_classes, 50)(in_label)
    n_nodes = 8 * 8 
    li = Dense(n_nodes)(li) 
    li = Reshape((8, 8, 1))(li)
    in_lat = Input(shape=(latent_dim,))
    
    n_nodes = 128 * 8 * 8
    gen = Dense(n_nodes)(in_lat) 
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((8, 8, 128))(gen) 
    merge = Concatenate()([gen, li])
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge) 
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen) #32x32x128
    gen = LeakyReLU(alpha=0.2)(gen)

    out_layer = Conv2D(3, (8,8), activation='tanh', padding='same')(gen) #32x32x3
    model = Model([in_lat, in_label], out_layer)
    return model 


def define_gan(g_model, d_model):
    d_model.trainable = False 
    gen_noise, gen_label = g_model.input 
    gen_output = g_model.output 
    gan_output = d_model([gen_output, gen_label])
    model = Model([gen_noise, gen_label], gan_output)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def load_real_samples():
    (trainX, trainy), (_, _) = load_data()   #cifar
    X = trainX.astype('float32')
    X = (X - 127.5) / 127.5  
    return [X, trainy]


def generate_real_samples(dataset, n_samples):
    images, labels = dataset  
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = ones((n_samples, 1))  
    return [X, labels], y


def generate_latent_points(latent_dim, n_samples, n_classes=10):
    x_input = randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


def generate_fake_samples(generator, latent_dim, n_samples):
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    images = generator.predict([z_input, labels_input])
    y = zeros((n_samples, 1))  
    return [images, labels_input], y


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=128):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)  
    for i in range(n_epochs):
        print(f"Epoch: {i}/{n_epochs}")
        for j in range(bat_per_epo):
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            d_loss_real, _ = d_model.train_on_batch([X_real, labels_real], y_real)
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss_fake, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)            
            y_gan = ones((n_batch, 1))
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            if i%50 == 0:
                g_model.save(f'cifar100_conditional_gan_generator_v3_{i}_epoch_model.h5')

    g_model.save('cifar100_conditional_gan_generator_v3_full_model.h5')

    
dataset = load_real_samples()
latent_dim = 100
d_model = define_discriminator()
g_model = define_generator(latent_dim)
gan_model = define_gan(g_model, d_model)
dataset = load_real_samples()
train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=300)


