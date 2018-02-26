import ConfigParser
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from isistan.smartweb.algorithm.VAE import VAE


from isistan.smartweb.core.SearchEngine import SmartSearchEngine
from isistan.smartweb.preprocess.StringPreprocessor import StringPreprocessor
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
        self._vectorizer = TfidfVectorizer()
        self._preprocessor = StringPreprocessor('english.long')

    def load_configuration(self, configuration_file):
        super(MLearningSearchEngine, self).load_configuration(configuration_file)
        config = ConfigParser.ConfigParser()
        config.read(configuration_file)
        latent_dim = config.getint('RegistryConfigurations', 'latent_dim')
        intermediate_dim = config.getint('RegistryConfigurations', 'intermediate_dim')
        batch_size = config.getint('RegistryConfigurations', 'batch_size')
        epochs = config.getint('RegistryConfigurations', 'epochs')
        self._model = VAE(latent_dim, intermediate_dim, 1.0,
                          batch_size, epochs)

    def unpublish(self, service):
        pass

    def _preprocess(self, bag_of_words):
        return bag_of_words.get_words_str()

    def _after_publish(self, documents):
        X = self._vectorizer.fit_transform(documents)
        X_train, X_test, _, _ = train_test_split(X, np.zeros(X.shape), test_size=0.33)
        self._model.train(X_train, X_test)
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
