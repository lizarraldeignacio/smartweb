import configparser
import numpy as np

import pickle

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras import backend as K


from isistan.smartweb.algorithm.VAE import VAE
#from isistan.smartweb.algorithm.VAEWasserstein import VAEWasserstein
from isistan.smartweb.core.SearchEngine import SmartSearchEngine
from isistan.smartweb.preprocess.StringPreprocessor import StringPreprocessor
from isistan.smartweb.preprocess.StringPreprocessorAdapter import StringPreprocessorAdapter
from isistan.smartweb.preprocess.StringTransformer import StringTransformer

__author__ = 'ignacio'


class MLearningSearchEngine(SmartSearchEngine):
    #
    # Uses a Keras model as the base to compute document similarity

    def __init__(self):
        super(MLearningSearchEngine, self).__init__()
        self._service_array = []
        self._index = None
        self._corpus = None
        self._train_model = False

    def load_configuration(self, configuration_file):
        super(MLearningSearchEngine, self).load_configuration(configuration_file)
        config = configparser.ConfigParser()
        config.read(configuration_file)
        latent_dim = config.getint('RegistryConfigurations', 'latent_dim')
        intermediate_dim = config.getint('RegistryConfigurations', 'intermediate_dim')
        batch_size = config.getint('RegistryConfigurations', 'batch_size')
        epochs = config.getint('RegistryConfigurations', 'epochs')
        learning_rate = config.getfloat('RegistryConfigurations', 'learning_rate')
        epsilon_std = config.getfloat('RegistryConfigurations', 'epsilon_std')
        if config.get('RegistryConfigurations', 'train_model').lower() == 'true':
            self._train_model = True
            if config.get('RegistryConfigurations', 'reproducible').lower() == 'true':
                self._model = VAE(latent_dim, intermediate_dim, epsilon_std,
                            batch_size, epochs, learning_rate, reproducible = True)
            else:
                self._model = VAE(latent_dim, intermediate_dim, epsilon_std,
                            batch_size, epochs, learning_rate)
            self._vectorizer = TfidfVectorizer(norm='l2', 
                                               preprocessor=StringPreprocessorAdapter('english.long'))
        else:
            self._model = VAE()
            self._model.load('vae.h5')
            self._vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        
    def unpublish(self, service):
        pass

    def _preprocess(self, bag_of_words):
        return bag_of_words.get_words_str()

    def _after_publish(self, documents):
        if self._train_model:

            # Custom objective function (Cosine similarity)
            def cos_distance(y_true, y_pred):
                y_true = K.l2_normalize(y_true, axis=-1)
                y_pred = K.l2_normalize(y_pred, axis=-1)
                return K.mean(1 - K.sum((y_true * y_pred), axis=-1))

            X = self._vectorizer.fit_transform(documents)
            pickle.dump(self._vectorizer, open('vectorizer.pkl', 'wb'))
            X_train, X_test, _, _ = train_test_split(X, np.zeros(X.shape), test_size=0.01, random_state=23)
            self._model.train(X_train, X_test, cos_distance)
            self._model.save('vae.h5')
        else:
            X = self._vectorizer.transform(documents)
        self._index = self._model.transform(X)

    def publish(self, service):
        pass

    def find(self, query):
        query = StringTransformer().transform(query)
        query_vector = self._vectorizer.transform([self._query_transformer.transform(query).get_words_str()])
        query_vae = self._model.transform(query_vector)
        results = cosine_similarity(query_vae, self._index)
        results = sorted(enumerate(results[0]), key=lambda item: -item[1])
        result_list = []
        for tuple_result in results:
            result_list.append(self._service_array[tuple_result[0]])
        return result_list

    def number_of_services(self):
        pass
