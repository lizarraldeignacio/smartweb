import configparser

from isistan.smartweb.preprocess.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from isistan.smartweb.core.SearchEngine import SmartSearchEngine
from isistan.smartweb.preprocess.StringPreprocessorAdapter import StringPreprocessorAdapter
from isistan.smartweb.preprocess.StringTransformer import StringTransformer

__author__ = 'ignacio'


class PCASearchEngine(SmartSearchEngine):
    #
    # Registry implementation using Principal Component Analysis

    def __init__(self):
        super(PCASearchEngine, self).__init__()
        self._service_array = []        
        self._lsi_index = None
        self._tfidf_matrix = None
        self._svd_matrix = None

    def load_configuration(self, configuration_file):
        super(PCASearchEngine, self).load_configuration(configuration_file)
        config = configparser.ConfigParser()
        config.read(configuration_file)
        number_of_topics = config.getint('RegistryConfigurations', 'number_of_topics')
        self._pca = PCA(n_components=number_of_topics, svd_solver='full', random_state=23)
        self._metric = config.get('RegistryConfigurations', 'metric').lower()
        self._vectorizer = TfidfVectorizer(sublinear_tf=False,
                                           analyzer='word', lowercase=False, use_bm25idf=self._use_bm25idf,
                                           bm25_tf=self._use_bm25tf, k = self._bm25_k,
                                           preprocessor=StringPreprocessorAdapter('english.long'))

    def unpublish(self, service):
        pass

    def _preprocess(self, bag_of_words):
        return bag_of_words.get_words_str()

    def _after_publish(self, documents):
        self._tfidf_matrix = self._vectorizer.fit_transform(documents)
        self._svd_matrix = self._pca.fit_transform(self._tfidf_matrix.toarray())
        self._lsi_index = NearestNeighbors(len(self._service_array), algorithm='brute', metric=self._metric)
        self._lsi_index.fit(self._svd_matrix)

    def publish(self, service):
        pass

    def find(self, query):
        query = StringTransformer().transform(query)
        query_array = self._vectorizer.transform([self._query_transformer.transform(query).get_words_str()])
        query_array = self._pca.transform(query_array.toarray())
        result = self._lsi_index.kneighbors(query_array, return_distance=False)[0]
        result_list = []
        for index in result:
            result_list.append(self._service_array[index])
        return result_list

    def number_of_services(self):
        pass
