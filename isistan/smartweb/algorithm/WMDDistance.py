


import numpy as np
import tensorflow as tf
from keras import backend as K

__author__ = 'ignacio'


class WMDDistance(object):
    #Implementation of Word Mover's Distance

    #Reference

    # From Word Embeddings To Document Distances
    # http://www.jmlr.org/proceedings/papers/v37/kusnerb15.pdf

    def __init__(self, dictionary, embeddings_model = None, distance_matrix = None):
        self._vocab_len = len(dictionary)
        self._embeddings = embeddings_model
        self._dictionary = dictionary
        self._distance_matrix = None
        vocab_len = len(dictionary)
        if distance_matrix is None:
            # Compute distance matrix.
            self._distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
            for i, t1 in list(dictionary.items()):
                for j, t2 in list(dictionary.items()):
                    # Compute Euclidean distance between word vectors.
                    self._distance_matrix[i, j] = np.sqrt(np.sum((embeddings_model[t1] - embeddings_model[t2])**2))
        else:
            self._distance_matrix = distance_matrix


    def save(self, path):
        np.save(path, self._distance_matrix)

    @staticmethod
    def load(path, dictionary):
        distance_matrix = np.load(path)
        return WMDDistance(dictionary, distance_matrix = distance_matrix)

    def get_distances(self):
        return self._distance_matrix

    @staticmethod
    def distance(params):
        batch_x = params[0]
        batch_y = params[1]
        distances = params[2]

        i0 = tf.constant(0)
        batch_size = K.shape(batch_x)[0]
        result = tf.zeros(shape=(1, batch_size))

        c = lambda i, similarity_mat, distances, result: i < tf.shape(similarity_mat)[0]

        similarity_mat = tf.equal(tf.not_equal(batch_x, 0), tf.not_equal(batch_y, 0))


        def body(i, similarity_mat, distances, result):
            #
            # Iteration over batch examples, for each example calculates the 
            # mean distance between the shared elements in similarity mat
        
            similarity_row = similarity_mat[i, :]
            
            non_zero_ind = tf.reshape(tf.where(similarity_row), [-1])
            
            '''Performs cartesian product to get distances of shared words'''
            tile_a = tf.tile(tf.expand_dims(non_zero_ind, 1), [1, tf.shape(non_zero_ind)[0]])  
            tile_a = tf.expand_dims(tile_a, 2) 
            tile_b = tf.tile(tf.expand_dims(non_zero_ind, 0), [tf.shape(non_zero_ind)[0], 1]) 
            tile_b = tf.expand_dims(tile_b, 2) 
            cartesian_product = tf.concat([tile_a, tile_b], axis=2)
            
            '''Creates a mask to add to the original tensor since is not posible to add new elements'''
            mean = tf.reduce_mean(tf.gather_nd(distances, cartesian_product))
            mask = tf.reshape(tf.one_hot(i, tf.shape(result)[1], on_value=mean), (1, -1))
            
            return i + 1, similarity_mat, distances, tf.add(result, mask)

        _, _, _, res = tf.while_loop(
            c, body, loop_vars=[i0, similarity_mat, distances, result],
            shape_invariants=[i0.shape, similarity_mat.shape, distances.shape, tf.TensorShape((None))])

        return res
