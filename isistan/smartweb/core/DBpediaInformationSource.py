import urllib
import urllib2
import httplib
import json
from InformationSource import InformationSource
from isistan.smartweb.preprocess.StringTransformer import StringTransformer
from isistan.smartweb.util.HttpUtils import HttpUtils

__author__ = 'ignacio'


class DBpediaInformationSource(InformationSource):
    #
    # Obtains information about terms using DBpedia as a source

    _NUMBER_OF_RETRIES = 10
    _NUMBER_OF_SENTENCES = 3
    _SEARCH_SERVICE_URL = 'http://lookup.dbpedia.org/api/search/KeywordSearch'

    def __init__(self):
        self._cache = {}

    def _query_dbpedia(self, query):
        query = query.encode('utf-8')
        n_retries = 0
        retry = True
        params = {
                'QueryString': query
        }
        search_url = self._SEARCH_SERVICE_URL + '?' + urllib.urlencode(params)
        if query not in self._cache:
            while retry and n_retries < self._NUMBER_OF_RETRIES:
                try:
                    retry = False
                    response = json.loads(HttpUtils.http_request(search_url, headers=[('Accept', 'application/json')]))
                    if len(response) > 0:
                        print 'Query: ' + query
                        print 'Response:'
                        print json.dumps(response)
                        if len(response['results']) > 0:
                            result = response['results'][0]
                            self._cache[query] = result['description']
                        else:
                            self._cache[query] = None
                except (urllib2.HTTPError, httplib.BadStatusLine, urllib2.URLError):
                    print 'retry'
                    retry = True
                    n_retries += 1
        return self._cache[query]

    def get_description(self, query):
        additional_words = []
        description = self._query_dbpedia(query)
        if description is not None:
            sentences = description.split('.')
            print 'found information for query: ' + query
            for i in range(0, min(len(sentences), self._NUMBER_OF_SENTENCES)):
                transformer = StringTransformer()
                additional_sentence = transformer.transform(sentences[i]).get_words_list()
                additional_words.extend(additional_sentence)
        else:
            print 'information not found for query: ' + query

        return additional_words

    def get_type(self, query):
        pass

    def get_aka(self, query):
        pass
