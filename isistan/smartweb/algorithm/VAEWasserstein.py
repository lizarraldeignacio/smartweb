


import numpy as np
import os
import tensorflow as tf
import random as rn
from scipy.stats import norm

from isistan.smartweb.algorithm.WMDDistance import WMDDistance

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


class VAEWasserstein(object):
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

    def train(self, x_train, x_test, embedding_distance):
        with self._graph.as_default():
            L = 50

            original_dim = x_train.shape[1] #Mirar

            x = Input(shape=(original_dim,))

            #emb_distance = K.constant(embedding_distance)
            
            h = Dense(self._intermediate_dim, activation='relu')(x)
            
            encoded_x = Dense(self._latent_dim)(h)

            # build a model to project inputs on the latent space
            self._encoder = Model(x, encoded_x)
            self._encoder.summary()


            # we instantiate these layers separately so as to reuse them later
            embedd = Input(shape=(self._latent_dim,))
            decoder_x = Dense(self._intermediate_dim, activation='relu')
            decoder_mean = Dense(original_dim, activation='sigmoid')
            x_decoded = decoder_x(embedd)
            x_decoded_mean = decoder_mean(x_decoded)

            self._decoder = Model(embedd, x_decoded_mean)
            self._decoder.summary()

            def generateTheta(L, endim):
                theta_ = np.random.normal(size = (L,endim))
                for l in range(L):
                    theta_[l,:] = theta_[l,:] / np.sqrt(np.sum(theta_[l,:] ** 2))
                return theta_

            def generateZ(batchsize,endim):
                z_= 2 * (np.random.uniform(size = (batchsize,endim))-0.5)
                return z_

            #Define a Keras Variable for \theta_ls
            theta = K.variable(generateTheta(L, self._latent_dim))
            
            #Define a Keras Variable for samples of z
            z = K.variable(generateZ(self._batch_size, self._latent_dim))

            # note that "output_shape" isn't necessary with the TensorFlow backend
            #z = Lambda(sampling, output_shape=(self._latent_dim,))([z_mean, z_log_var])

            aencoded = self._encoder(x)
            ae = self._decoder(aencoded)
            self._autoencoder = Model(x, ae)
            self._autoencoder.summary()

            # Let projae be the projection of the encoded samples
            projae = K.dot(aencoded, K.transpose(theta))
            
            # Let projz be the projection of the $q_Z$ samples
            projz = K.dot(z, K.transpose(theta))
            
            # Calculate the Sliced Wasserstein distance by sorting
            # the projections and calculating the L2 distance between
            #cross_entropy_loss = (1.0) * K.mean(K.binary_crossentropy(x, ae))
            #emb_loss = Lambda(WMDDistance.distance)([x, ae, emb_distance])
            w2 = (tf.nn.top_k(tf.transpose(projae), k = tf.shape(x)[0]).values - tf.nn.top_k(tf.transpose(projz), k = tf.shape(x)[0]).values) ** 2

            #emb_loss = K.mean(K.abs(emb_loss))
            l1_loss = (1.0) * K.mean(K.abs(x-ae))
            w2_loss = (10.0) * K.mean(w2)

            # I have a combination of L1 and Cross-Entropy loss
            # for the first term and then and W2 for the second term
            #vae_loss = emb_loss + l1_loss + w2_loss
            vae_loss = l1_loss + w2_loss

            self._autoencoder.add_loss(vae_loss)

            self._autoencoder.compile(optimizer = optimizers.Adam(lr = self._learning_rate))
            
            self._autoencoder.fit(x_train,
                    shuffle=True,
                    epochs=self._epochs,
                    batch_size=self._batch_size,
                    validation_data=(x_test, None))

    def transform(self, X):
        with self._graph.as_default():
            return self._encoder.predict(X)

    def save(self, path):
        self._encoder.save(path)

    def load(self, path):
        self._encoder = load_model(path)
