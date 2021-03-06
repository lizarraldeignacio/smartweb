import re
from .Transformer import Transformer
from isistan.smartweb.persistence.XMLReader import XMLReader
from isistan.smartweb.persistence.WordBag import WordBag

__author__ = 'ignacio'


class WSDLSimpleTransformer(Transformer):
    #
    # Process the WSDL Document

    WSDL_TAG_MESSAGE = 'message'
    WSDL_TAG_TYPE = 'type'
    WSDL_TAG_ELEMENT = 'element'
    WSDL_TAG_SCHEMA = 'schema'
    WSDL_TAG_COMPLEX_TYPE = 'complexType'
    WSDL_TAG_SIMPLE_TYPE = 'simpleType'
    WSDL_TAG_SEQUENCE = 'sequence'
    WSDL_TAG_PART = 'part'
    WSDL_TAG_BINDING = 'binding'
    WSDL_TAG_PORT = 'port'
    WSDL_TAG_OPERATION = 'operation'
    WSDL_TAG_PORT_TYPE = 'portType'
    WSDL_TAG_INPUT = 'input'
    WSDL_TAG_OUTPUT = 'output'
    WSDL_TAG_DOCUMENTATION = 'documentation'
    WSDL_TAG_EXTENSION = 'extension'
    WSDL_TAG_RESTRICTION = 'restriction'
    WSDL_TAG_FAULT = 'fault'
    WSDL_TAG_SERVICE = 'service'
    WSDL_TAG_PORT = 'port'
    WSDL_TAG_DEFINITIONS = 'definitions'

    WSDL_ATTRIBUTE_NAME = 'name'
    WSDL_ATTRIBUTE_TYPE = 'type'
    WSDL_ATTRIBUTE_BASE = 'base'
    WSDL_ATTRIBUTE_ELEMENT = 'element'
    WSDL_ATTRIBUTE_MESSAGE = 'message'
    WSDL_ATTRIBUTE_SOAP_ACTION = 'soapAction'
    WSDL_ATTRIBUTE_BINDING = 'binding'

    def __init__(self):
        super(WSDLSimpleTransformer, self).__init__()
        self._reader = XMLReader()

    def _process_service(self):
        terms = []
        service = self._reader.get_elements([self.WSDL_TAG_SERVICE])[0]
        service_name_list = WSDLSimpleTransformer._process_name(service, self.WSDL_ATTRIBUTE_NAME)
        if service_name_list is not None:
            terms.extend(service_name_list)
        port_type_list = self._reader.get_elements([self.WSDL_TAG_PORT_TYPE])
        terms.extend(self._process_port_types(port_type_list))
        return terms

    def _process_port_types(self, port_type_list):
        terms = []
        if len(port_type_list) > 0:
            for port_type in port_type_list:
                operation_list = self._reader.get_elements_from(port_type, [self.WSDL_TAG_OPERATION])
                if operation_list is not None:
                    for operation in operation_list:
                        terms.extend(WSDLSimpleTransformer._process_name(operation, self.WSDL_ATTRIBUTE_NAME))
        return terms

    @staticmethod
    def _process_name(operation, attribute):
        if attribute in operation.attrib:
            operation_name = operation.attrib[attribute]
            if len(operation_name.split(':')) > 1:
                operation_name = operation_name.split(':')[1]
            return re.sub(' +', ' ', re.sub("([a-z])([A-Z])", "\g<1> \g<2>",
                                            re.sub(' +', ' ',
                                                   re.sub('[^a-zA-Z ]', ' ',
                                                          operation_name
                                                          )))).split(' ')

    def transform(self, url):
        """Transform a wsdl file into a string"""
        self._reader.load_from_url(url)
        return WordBag(self._process_service())
