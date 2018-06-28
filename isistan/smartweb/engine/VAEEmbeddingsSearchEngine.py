import ConfigParser
import numpy as np

import pickle

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from gensim.corpora import Dictionary
from gensim import models, similarities


from isistan.smartweb.algorithm.VAE import VAE
from isistan.smartweb.algorithm.WMDDistance import WMDDistance
from isistan.smartweb.core.SearchEngine import SmartSearchEngine
from isistan.smartweb.preprocess.StringPreprocessor import StringPreprocessor
from isistan.smartweb.preprocess.StringPreprocessorAdapter import StringPreprocessorAdapter
from isistan.smartweb.preprocess.StringTransformer import StringTransformer

__author__ = 'ignacio'


class VAEEmbeddingsSearchEngine(SmartSearchEngine):
    #
    # Uses a Keras model as the base to compute document similarity

    def __init__(self):
        super(VAEEmbeddingsSearchEngine, self).__init__()
        self._service_array = []
        self._index = None
        self._corpus = None
        self._train_model = False
        self._load_wmd = False
        self._preprocessor = StringPreprocessor('english.long')

    def load_configuration(self, configuration_file):
        super(VAEEmbeddingsSearchEngine, self).load_configuration(configuration_file)
        config = ConfigParser.ConfigParser()
        config.read(configuration_file)
        latent_dim = config.getint('RegistryConfigurations', 'latent_dim')
        intermediate_dim = config.getint('RegistryConfigurations', 'intermediate_dim')
        batch_size = config.getint('RegistryConfigurations', 'batch_size')
        epochs = config.getint('RegistryConfigurations', 'epochs')
        learning_rate = config.getfloat('RegistryConfigurations', 'learning_rate')
        epsilon_std = config.getfloat('RegistryConfigurations', 'epsilon_std')
        self._precomputed_vectors_path = config.get('RegistryConfigurations', 'precomputed_vectors_path')
        if config.get('RegistryConfigurations', 'load_wmd_model').lower() == 'true':
            self._load_wmd = True
        if config.get('RegistryConfigurations', 'train_model').lower() == 'true':
            self._train_model = True
            if config.get('RegistryConfigurations', 'reproducible').lower() == 'true':
                self._model = VAE(latent_dim, intermediate_dim, epsilon_std,
                            batch_size, epochs, learning_rate, reproducible = True)
            else:
                self._model = VAE(latent_dim, intermediate_dim, epsilon_std,
                            batch_size, epochs, learning_rate)
        else:
            self._model = VAE()
            self._model.load('vae.h5')
            self._vectorizer = Dictionary.load(open('vectorizer.pkl', 'rb'))
        
    def _doc_to_nbow(self, document):
        vocab_len = len(self._vectorizer)
        d = np.zeros(vocab_len, dtype=np.double)
        nbow = self._vectorizer.doc2bow(document)  # Word frequencies.
        doc_len = len(document)
        for idx, freq in nbow:
            d[idx] = freq / float(doc_len)  # Normalized word frequencies.
        return d

    def _corpus_to_nbow(self, documents):
        corpus = np.zeros((len(documents),len(self._vectorizer)))
        for i in range(len(documents)):
            corpus[i, :] = self._doc_to_nbow(documents[i])
        return corpus

    def unpublish(self, service):
        pass

    def _preprocess(self, bag_of_words):
        words = bag_of_words.get_words_list()
        return self._preprocessor(words)

    def _after_publish(self, documents):
        if self._train_model:
            self._word2vec_model = models.KeyedVectors.load_word2vec_format(self._precomputed_vectors_path, binary=False)
            self._word2vec_model.init_sims(replace=True)
            documents = [filter(lambda x: x in self._word2vec_model.vocab, document) for document in documents]
            self._vectorizer = Dictionary(documents)
            self._vectorizer.save(open('vectorizer.pkl', 'wb'))
            X = self._corpus_to_nbow(documents)
            
            if self._load_wmd:
                distance = WMDDistance.load('distances.npy', self._vectorizer, self._word2vec_model)
            else:
                distance = WMDDistance(self._vectorizer, self._word2vec_model)    
                distance.save('distances')
            X_train, X_test, _, _ = train_test_split(X, np.zeros(X.shape), test_size=0.33, random_state=23)
            self._model.train(X_train, X_test, distance.distance)
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
