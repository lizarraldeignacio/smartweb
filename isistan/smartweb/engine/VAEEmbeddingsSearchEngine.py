import configparser
import numpy as np

import pickle

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from gensim.corpora import Dictionary
from gensim import models, similarities


from isistan.smartweb.algorithm.VAEWasserstein import VAEWasserstein
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
        config = configparser.ConfigParser()
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
                self._model = VAEWasserstein(latent_dim, intermediate_dim, epsilon_std,
                            batch_size, epochs, learning_rate, reproducible = True)
            else:
                self._model = VAEWasserstein(latent_dim, intermediate_dim, epsilon_std,
                            batch_size, epochs, learning_rate)
        else:
            self._model = VAEWasserstein()
            self._model.load('models/vae.h5')
            self._vectorizer = Dictionary.load('models/vectorizer.npy')
        
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

    def _save_obj(self, obj, name):
        with open('models/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def _load_obj(self, name):
        with open('models/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def _create_filter_vocab(self, documents, vocab):
        filter_set = set()
        for document in documents:
            for word in document:
                if word not in vocab:
                    filter_set.add(word)
        return filter_set

    def _after_publish(self, documents):
        if self._train_model:
            filter_set = None
            if self._load_wmd:
                filter_set = self._load_obj('word_filter')
                documents = [[x for x in document if x not in filter_set] for document in documents]
                self._vectorizer = Dictionary(documents)
                distance = WMDDistance.load('models/distances.npy', self._vectorizer)
            else:
                self._word2vec_model = models.KeyedVectors.load_word2vec_format(self._precomputed_vectors_path, binary=False)
                self._word2vec_model.init_sims(replace=True)
                filter_set = self._create_filter_vocab(documents, self._word2vec_model.vocab)
                self._save_obj(filter_set, 'word_filter')
                documents = [[x for x in document if x not in filter_set] for document in documents]
                self._vectorizer = Dictionary(documents)
                distance = WMDDistance(self._vectorizer, self._word2vec_model)    
                distance.save('models/distances')
            X = self._corpus_to_nbow(documents)
            self._vectorizer.save(open('models/vectorizer.npy', 'wb'))
            X_train, X_test, _, _ = train_test_split(X, np.zeros(X.shape), test_size=0.33, random_state=23)
            print(X_train)
            print(X_test)
            self._model.train(X_train, X_test, distance.get_distances())
            self._model.save('models/vae.h5')
        else:
            filter_set = self._load_obj('word_filter')
            documents = [[x for x in document if x not in filter_set] for document in documents]
            X = self._corpus_to_nbow(documents)
        self._index = self._model.transform(X)

    def publish(self, service):
        pass

    def find(self, query):
        query = StringTransformer().transform(query)
        query_vector = self._doc_to_nbow(self._query_transformer.transform(query).get_words_list())
        query_vector = np.expand_dims(query_vector, axis=0)
        query_vae = self._model.transform(query_vector)
        results = cosine_similarity(query_vae, self._index)
        results = sorted(enumerate(results[0]), key=lambda item: -item[1])
        result_list = []
        for tuple_result in results:
            result_list.append(self._service_array[tuple_result[0]])
        return result_list

    def number_of_services(self):
        pass
