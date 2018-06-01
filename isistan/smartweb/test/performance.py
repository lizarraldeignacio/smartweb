from isistan.smartweb.persistence.WordBag import WordBag
from isistan.smartweb.preprocess.StringPreprocessor import StringPreprocessor
from isistan.smartweb.preprocess.WADLTransformer import WADLTransformer
from isistan.smartweb.preprocess.WSDLTransformer import WSDLTransformer
from isistan.smartweb.preprocess.WSDLSimpleTransformer import WSDLSimpleTransformer
from isistan.smartweb.preprocess.PlainTextTransformer import PlainTextTransformer
from isistan.smartweb.preprocess.NERTransformer import NERTransformer
from isistan.smartweb.core.DBpediaInformationSource import DBpediaInformationSource
from isistan.smartweb.core.StandfordNER import StandfordNER



def _create_document_transformer(document_list):
        if len(document_list) > 0:
            document = document_list[0].split('.')
            extension = document[len(document)-1].lower()
            if extension == 'wadl':
                return WADLTransformer()
            elif extension == 'wsdl':
                return WSDLTransformer()
            else:
                return PlainTextTransformer()
        print 'Dataset: Invalid document format'

def _preprocess(bag_of_words):
        words = bag_of_words.get_words_list()
        return StringPreprocessor('english.long')(words)

def timeit_indexing_NER(service_list):
        transformer = _create_document_transformer(service_list)
        documents = []
        current_document = 1
        knowledge_source = DBpediaInformationSource(base_url='http://localhost:1111/api/search/KeywordSearch')
        standford_ner = StandfordNER()
        document_transformer = NERTransformer(knowledge_source, standford_ner)
        print 'Loading documents'
        for document in service_list:
            bag_of_words = document_transformer.transform(transformer.transform(document))
            documents.append(_preprocess(bag_of_words))
            current_document += 1

def timeit_indexing(service_list):
    transformer = _create_document_transformer(service_list)
    documents = []
    current_document = 1
    print 'Loading documents'
    for document in service_list:
        bag_of_words = transformer.transform(document)
        documents.append(_preprocess(bag_of_words))
        current_document += 1