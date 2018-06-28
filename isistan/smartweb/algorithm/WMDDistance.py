
from __future__ import print_function

import numpy as np
from keras import backend as K

# If pyemd C extension is available, import it.
# If pyemd is attempted to be used, but isn't installed, ImportError will be raised in wmdistance
try:
    from pyemd import emd
    PYEMD_EXT = True
except ImportError:
    PYEMD_EXT = False

__author__ = 'ignacio'


class WMDDistance(object):
    #Implementation of Word Mover's Distance

    #Reference

    # From Word Embeddings To Document Distances
    # http://www.jmlr.org/proceedings/papers/v37/kusnerb15.pdf

    def __init__(self, dictionary, embeddings_model, distance_matrix = None):
        
        if not PYEMD_EXT:
            raise ImportError("Please install pyemd Python package to compute WMD.")
        vocab_len = len(dictionary)
        if distance_matrix is None:
            # Compute distance matrix.
            self._distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
            for i, t1 in dictionary.items():
                for j, t2 in dictionary.items():
                    # Compute Euclidean distance between word vectors.
                    self._distance_matrix[i, j] = np.sqrt(np.sum((embeddings_model[t1] - embeddings_model[t2])**2))
        else:
            self._distance_matrix = distance_matrix

    def save(self, path):
        np.save(path, self._distance_matrix)

    @staticmethod
    def load(path, dictionary, embeddings_model):
        distance_matrix = np.load(path)
        return WMDDistance(dictionary, embeddings_model, distance_matrix)

    def distance(self, nbow_document1, nbow_document2):
        nbow_document1 = K.eval(nbow_document1)
        nbow_document2 = K.eval(nbow_document2)
        if np.sum(self._distance_matrix) == 0.0:
            # `emd` gets stuck if the distance matrix contains only zeros.
            return float('inf')
        return emd(nbow_document1, nbow_document2, self._distance_matrix)
