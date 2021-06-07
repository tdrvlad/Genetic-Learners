import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

'''
agent_initialisation passes the hyperparameters of the agent's definition.

initialisation = {
    embedding_size: 32,
    start_exp_no_filters = 0,
    final_exp_no_filters = 8, 
    start_kernel_size = 1,
    final_kernel_size = 9,
    pooling_frequency = 2
}
'''

INPUT_SHAPE = (32, 32, 1)

initialisation = {
    'exp_embedding_size': 1,
    'no_conv_layers' : 1,
    'start_exp_no_filters' : 8,
    'final_exp_no_filters' : 6, 
    'start_kernel_size' : 3,
    'final_kernel_size' : 3,
    'pooling_frequency' : 1
}


def random_float(a,b):
    if a != b:
        return np.random.uniform(min(a,b), max(a,b))
    else:
        return a

def random_int(a,b):   
    if a != b:
        return np.random.randint(min(a,b), max(a,b))
    else:
        return int(a)


def plot_latent_space(vae, n=20, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 32
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig('latent_space.png')


def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig('label_clusters.png')


class SamplingLayer(keras.layers.Layer):
    '''Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.'''

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class LearnerAgent:
    
    def __init__(self, initialisation):
        
        self.embedding_size = int(2 ** initialisation.get('exp_embedding_size'))
        self.no_conv_layers = initialisation.get('no_conv_layers')
        self.start_exp_no_filters = initialisation.get('start_exp_no_filters')
        self.final_exp_no_filters = initialisation.get('final_exp_no_filters')
        self.start_kernel_size = initialisation.get('start_kernel_size')
        self.final_kernel_size = initialisation.get('final_kernel_size')
        self.pooling_frequency = initialisation.get('pooling_frequency')

        self.__build_autoencoder()
    

    def __build_convolutional_layers(self):

        conv_encoder_layers = []
        conv_decoder_layers = []
        
        no_layers = 0
        no_halvings = 0
        
        current_exp_no_filters = self.start_exp_no_filters
        current_kernel_size = self.start_kernel_size


        while True:

            no_layers += 1

            current_exp_no_filters = random_int(current_exp_no_filters, self.final_exp_no_filters)
            no_filters = int(2 ** current_exp_no_filters)

            current_kernel_size = random_int(current_kernel_size, self.final_kernel_size)

            if no_layers % self.pooling_frequency == 0:
                strides = 2
                no_halvings += 1
            else:
                strides = 1

            conv_encoder_layers.append(
                keras.layers.Conv2D(no_filters, current_kernel_size, strides = strides, activation = "relu", padding = "same")
            )

            conv_decoder_layers.append(
                keras.layers.Conv2DTranspose(no_filters, current_kernel_size, strides = strides, activation = "relu", padding = "same")
            )

            if no_layers > self.no_conv_layers or 2 ** (no_halvings + 1) > INPUT_SHAPE[0]:
                break
        
        conv_decoder_layers.reverse()

        return conv_encoder_layers, conv_decoder_layers
    

    def __build_encoder(self, conv_encoder_layers):

        encoder_input = keras.layers.Input(shape=INPUT_SHAPE)
        x = encoder_input

        for layer in conv_encoder_layers:
            x = layer(x)

        conv_embedding_shape = x.shape[1:]
        x = keras.layers.Flatten()(x)
        encoder_output_dimension = x.shape[1]

        z_mean = keras.layers.Dense(self.embedding_size, name="z_mean")(x)
        z_log_var = keras.layers.Dense(self.embedding_size, name="z_log_var")(x)
        z = SamplingLayer()([z_mean, z_log_var])

        encoder_output = [z_mean, z_log_var, z]
        
        encoder = keras.Model(encoder_input, encoder_output, name = "encoder")

        return encoder, conv_embedding_shape, encoder_output_dimension
    

    def __buil_decoder(self, conv_decoder_layers, conv_embedding_shape, encoder_output_dimension):


        decoder_input = keras.Input(shape = (self.embedding_size,))

        x = keras.layers.Dense(encoder_output_dimension)(decoder_input)
        x = keras.layers.Reshape(conv_embedding_shape)(x)

        for layer in conv_decoder_layers:
            x = layer(x)

        decoder_output = keras.layers.Conv2DTranspose(INPUT_SHAPE[2], 3, activation="sigmoid", padding="same")(x)

        decoder = keras.Model(decoder_input, decoder_output, name="decoder")

        return decoder


    def __build_autoencoder(self):

        conv_encoder_layers, conv_decoder_layers = self.__build_convolutional_layers()
        encoder, conv_embedding_shape, encoder_output_dimension = self.__build_encoder(conv_encoder_layers)
        decoder = self.__buil_decoder(conv_decoder_layers, conv_embedding_shape, encoder_output_dimension)

        self.encoder = encoder
        self.decoder = decoder

        self.autoencoder = VAE(self.encoder, self.decoder)

        self.no_params = self.encoder.count_params() + self.decoder.count_params()

    
    def train(self, data, batch = 128, epochs = 2):
        (x_train, y_train), (x_test, y_test) = data

        y_train = y_train
        x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant')
        x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant')

        mnist_digits = np.concatenate([x_train, x_test], axis=0)
        mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

        self.autoencoder.compile(optimizer=keras.optimizers.Adam())
        history = self.autoencoder.fit(mnist_digits, epochs=epochs, batch_size=batch, verbose=0)

        #plot_latent_space(self.autoencoder)
        #plot_label_clusters(self.autoencoder, np.expand_dims(x_train, -1).astype("float32") / 255, y_train)

        return history.history.get('loss')
    

if __name__ == '__main__':
    data = keras.datasets.mnist.load_data()

    agent = LearnerAgent(initialisation)
    agent.train_mnist(data)
