import ConfigParser
from os.path import join

from gensim import corpora, models, similarities

from isistan.smartweb.core.SearchEngine import SmartSearchEngine
from isistan.smartweb.persistence.WordBag import WordBag
from isistan.smartweb.preprocess.SimplePreprocessor import SimplePreprocessor
from isistan.smartweb.preprocess.StringTransformer import StringTransformer

__author__ = 'ignacio'


class FastTextSearchEngine(SmartSearchEngine):
    #
    # Registry implementation using FastText

    def __init__(self):
        super(FastTextSearchEngine, self).__init__()
        self._service_map = {}
        self._service_array = []
        self._fasttext_model = None
        self._index = None
        self._preprocessor = SimplePreprocessor('english.long')
        self._precomputed_vectors_path = None

    def load_configuration(self, configuration_file):
        super(FastTextSearchEngine, self).load_configuration(configuration_file)
        
        config = ConfigParser.ConfigParser()
        config.read(configuration_file)

        self._precomputed_vectors_path = config.get('RegistryConfigurations', 'precomputed_vectors_path')

    def unpublish(self, service):
        pass

    def _preprocess(self, bag_of_words):
        words = bag_of_words.get_words_list()
        return self._preprocessor(words)

    def _after_publish(self, documents):
        print "Loading Model..."
        self._fasttext_model = models.FastText.load_fasttext_format(self._precomputed_vectors_path)
        self._fasttext_model.init_sims(replace=True)
        print "[OK]"
        for document, service in zip(documents, self._service_array):
            print service
            self._service_map[service] = filter(lambda x: x in self._fasttext_model.vocab, document)

    def publish(self, service):
        pass

    def find(self, query):
        transformer = StringTransformer()
        query = self._preprocessor(transformer.transform(query).get_words_list())
        # Filter words from the query that aren't in the vocabulary
        query = list(filter(lambda x: x in self._fasttext_model.vocab, query))
        results = []
        for key in self._service_map.keys():
            # Assign 0 similarty for empty documents, otherwise calculate similarity
            if self._service_map[key]:
              results.append((key, self._fasttext_model.n_similarity(query, self._service_map[key])))
            else:
              results.append((key, 0))
        results = sorted(results, key=lambda item: -item[1])
        result_list = []
        for tuple_result in results:
            result_list.append(tuple_result[0])
        return result_list

    def number_of_services(self):
        pass
