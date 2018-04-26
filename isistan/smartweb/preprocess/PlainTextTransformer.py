from Transformer import Transformer
from isistan.smartweb.persistence.WordBag import WordBag
import csv

__author__ = 'ignacio'


class PlainTextTransformer(Transformer):
    #
    # Process a Text Document

    def __init__(self):
        super(PlainTextTransformer, self).__init__()

    def transform(self, filepath):
        """Transform a wsdl file into a string"""
        with open(filepath[7:], 'rb') as words_file:
            try:
                words = csv.reader(words_file, delimiter=' ').next()
            except StopIteration:
                words = ''
        return WordBag(words)
