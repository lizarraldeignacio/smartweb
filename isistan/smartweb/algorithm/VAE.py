
import numpy as np
import os
import tensorflow as tf
import random as rn
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import optimizers
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle

__author__ = 'ignacio'


class VAE(object):
    #Variational autoencoder with Keras.

    #Reference

    # Auto-Encoding Variational Bayes
    # https://arxiv.org/abs/1312.6114

    def __init__(self, latent_dim = 256, intermediate_dim = 512, epsilon_std = 1.0,
                 batch_size = 100, epochs = 50, learning_rate = 0.001, reproducible= False):
        self._batch_size = batch_size
        self._latent_dim = latent_dim
        self._intermediate_dim = intermediate_dim
        self._epochs = epochs
        self._epsilon_std = epsilon_std
        self._learning_rate = learning_rate

        ###################################
        # TensorFlow wizardry
        config = tf.ConfigProto()
        
        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True
        
        # Only allow a total of half the GPU memory to be allocated
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        
        # Create a session with the above options specified.
        K.tensorflow_backend.set_session(tf.Session(config=config))
        #K.set_session(tf.Session(config=config))
        self._graph = tf.get_default_graph()

        if reproducible:
            # The below is necessary in Python 3.2.3 onwards to
            # have reproducible behavior for certain hash-based operations.
            # See these references for further details:
            # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
            # https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

            os.environ['PYTHONHASHSEED'] = '0'

            # The below is necessary for starting Numpy generated random numbers
            # in a well-defined initial state.

            np.random.seed(23)

            # The below is necessary for starting core Python generated random numbers
            # in a well-defined state.

            rn.seed(23)

            # Force TensorFlow to use single thread.
            # Multiple threads are a potential source of
            # non-reproducible results.
            # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

            session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

            # The below tf.set_random_seed() will make random number generation
            # in the TensorFlow backend have a well-defined initial state.
            # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

            tf.set_random_seed(23)

            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
            K.set_session(sess)

    def train(self, x_train, x_test, distance):
         with self._graph.as_default():
            original_dim = x_train.shape[1] #Mirar

            x = Input(shape=(original_dim,))
            h = Dense(self._intermediate_dim, activation='relu')(x)
            z_mean = Dense(self._latent_dim)(h)
            z_log_var = Dense(self._latent_dim)(h)


            def sampling(args):
                z_mean, z_log_var = args
                epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self._latent_dim), mean=0.,
                                        stddev=self._epsilon_std)
                return z_mean + K.exp(z_log_var / 2) * epsilon

            # note that "output_shape" isn't necessary with the TensorFlow backend
            z = Lambda(sampling, output_shape=(self._latent_dim,))([z_mean, z_log_var])

            # we instantiate these layers separately so as to reuse them later
            decoder_h = Dense(self._intermediate_dim, activation='relu')
            decoder_mean = Dense(original_dim, activation='sigmoid')
            h_decoded = decoder_h(z)
            x_decoded_mean = decoder_mean(h_decoded)

            # instantiate VAE model
            self._vae = Model(x, x_decoded_mean)

            # Compute VAE loss
            xent_loss = original_dim * distance(x, x_decoded_mean)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            vae_loss = K.mean(xent_loss + kl_loss)

            self._vae.add_loss(vae_loss)
            self._vae.compile(optimizer=optimizers.Adam(lr=self._learning_rate))
            self._vae.summary()

            self._vae.fit(x_train,
                    shuffle=True,
                    epochs=self._epochs,
                    batch_size=self._batch_size,
                    validation_data=(x_test, None))

            # build a model to project inputs on the latent space
            self._encoder = Model(x, z_mean)

    def transform(self, X):
        with self._graph.as_default():
            return self._encoder.predict(X)

    def save(self, path):
        self._encoder.save(path)

    def load(self, path):        
        self._encoder = load_model(path)
        self._graph = tf.get_default_graph()
