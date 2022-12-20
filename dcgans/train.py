import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt 
import os
from IPython import display
import boto3 
import time
import s3fs 
import random

IMG_WIDTH = 28
IMG_HEIGHT = 28 
CHANNELS = 1
LATENT_SHAPE = 100
BUFFER_SIZE = 60000
BATCH_SIZE = 128
EPOCHS = 1000


def load_real_data(): 
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    return train_X, train_y 

train_X, train_y = load_real_data()
print("Training Data Shape: ", train_X.shape)


def normalize_input(X: np.ndarray, is_use_tanh: bool=False) -> np.ndarray: 
    if is_use_tanh: 
        X = X.reshape(X.shape[0], 28, 28, 1).astype('float32')
        X = (X - 127.5) / 127.5
    else: 
        X = X / 255.0
    return X

train_X = normalize_input(train_X, is_use_tanh=True)


def create_tf_dataset(train_X, buffer_size: int, batch_size: int) -> tf.data.Dataset: 
    dataset = tf.data.Dataset.from_tensor_slices(train_X).shuffle(buffer_size).batch(batch_size)
    return dataset


def generate_real_class_labels(label_size: int): 
    labels = np.ones((label_size)).reshape(label_size, 1)
    return labels


def generate_fake_class_labels(label_size: int): 
    labels = np.zeros((label_size)).reshape(label_size, 1)
    return labels


def generate_latent_points(dim: int, batch_size: int):
    latent_vector = tf.random.normal([batch_size, dim])
    return latent_vector


def load_batch_X(dataset, batch_size: int): 
    idx = [random.randrange(0, (dataset.shape[0])) for i in range(batch_size)]
    data = dataset[idx]
    return data


class ANNBlock(tf.keras.layers.Layer): 
    def __init__(self, alpha: int, units: int, momentum: float, dropout_rate: float): 
        super(ANNBlock, self).__init__()
        self.Dense = tf.keras.layers.Dense(units=units)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, input_tensor, training=False): 
        x = self.Dense(input_tensor)
        x = tf.nn.leaky_relu(x)
        #x = self.dropout(x)
        return x

class CNNTransposeBlock(tf.keras.layers.Layer): 
    def __init__(self, kernel_shape: tuple, channels: int, strides: tuple, padding: str, momentum: float, use_bias: bool, alpha: float): 
        super(CNNTransposeBlock, self).__init__()
        self.conv2d_trans = tf.keras.layers.Conv2DTranspose(filters=channels, kernel_size=kernel_shape, strides=strides, padding=padding, use_bias=use_bias)
        self.alpha = alpha

    def call(self, input_tensor, training=False):
        x = self.conv2d_trans(input_tensor)
        x = tf.nn.leaky_relu(x, self.alpha)
        return x 


class CNNBlock(tf.keras.layers.Layer): 
    def __init__(self, kernel_shape: tuple, channels: int, strides: tuple, padding: str, momentum: float, use_bias: bool, pool_size: tuple, mp_strides: tuple, mp_padding: str, dropout_rate: float, alpha: float): 
        super(CNNBlock, self).__init__()
        self.conv2d_trans = tf.keras.layers.Conv2DTranspose(filters=channels, kernel_size=kernel_shape, strides=strides, padding=padding, use_bias=use_bias)
        self.alpha = alpha

    def call(self, input_tensor, training=False):
        x = self.conv2d_trans(input_tensor)
        x = tf.nn.leaky_relu(x, self.alpha)
        return x 


def create_generator(): 
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(100, )))
    model.add(ANNBlock(0.2, 7*7*256, 0.6, 0.0))
    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)
    model.add(CNNTransposeBlock((4, 4), 128, (2, 2), "same", 0.6, False, 0.2))
    assert model.output_shape == (None, 14, 14, 128)
    model.add(CNNTransposeBlock((4, 4), 128, (2, 2), "same", 0.6, False, 0.2))
    assert model.output_shape == (None, 28, 28, 128)
    model.add(tf.keras.layers.Conv2D(1, (7,7), activation='tanh', padding='same'))
    return model


def create_discriminator(): 
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(28, 28, 1)))
    model.add(CNNBlock((3, 3), 128, strides=(2, 2), padding="same", momentum=0.8, use_bias=False, \
            pool_size=(2, 2), mp_strides=(1, 1), mp_padding="same", dropout_rate=0.3, alpha=0.2))

    model.add(CNNBlock((3, 3), 128, strides=(2, 2), padding="same", momentum=0.8, use_bias=False, \
            pool_size=(2, 2), mp_strides=(1, 1), mp_padding="same", dropout_rate=0.3, alpha=0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def create_gan_model(generator, discriminator): 
    discriminator.trainable = False
    generator.trainable = True
    gan_model = tf.keras.models.Sequential()
    gan_model.add(generator)
    gan_model.add(discriminator)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    gan_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    return gan_model


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, LATENT_SHAPE])

def train(batch_size: int, epochs: int, generator, discriminator, dataset, gan_model, checkpoint, checkpoint_prefix, checkpoint_epoch):
    
    batch_per_epoch = int((dataset.shape[0]) / batch_size)
    half_batch = int(batch_size / 2)
    
    for epoch in range(epochs):
        print(f"Epochs {epoch}/{epochs}")
        for _batch in range(batch_per_epoch): 
            latent_points = generate_latent_points(LATENT_SHAPE, batch_size)    
            # Discriminator
            real_X = load_batch_X(dataset, batch_size)
            real_y = generate_real_class_labels(batch_size)
            d_loss_on_real, _ = discriminator.train_on_batch(real_X, real_y)
            
            # for fake sample
            fake_X = generator.predict(latent_points)
            fake_y = generate_fake_class_labels(batch_size)
            d_loss_on_fake, _ = discriminator.train_on_batch(fake_X, fake_y)
            
            # Generator
            g_loss, _ = gan_model.train_on_batch(latent_points, fake_y)
            
        if epoch % checkpoint_epoch == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            
    gan_model.save("generator.h5")


def save(): 
    fs = s3fs.S3FileSystem(key=os.environ["aws_access_key"], secret=os.environ["aws_secret_key"])
    fs.upload("generator_model.h5", "/name-of-my-bucket/generator_model.h5")


def run(): 
    discriminator_model = create_discriminator()
    generator_model = create_generator()
    gan_model = create_gan_model(generator_model, discriminator_model)
    checkpoint_dir = 'training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator=generator_model,
                                    discriminator=discriminator_model,
                                    gan=gan_model)
    train_X, train_y = load_real_data()
    train_X = normalize_input(train_X, is_use_tanh=True)
    train(BATCH_SIZE, EPOCHS, generator_model, discriminator_model, train_X,  gan_model, checkpoint, checkpoint_prefix, 10)
    save()

if __name__ == "__main__": 
    run()