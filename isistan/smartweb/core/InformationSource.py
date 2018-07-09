import abc

__author__ = 'ignacio'


class InformationSource(object, metaclass=abc.ABCMeta):
    #
    # Information source abstract class

    @abc.abstractmethod
    def get_description(self, query):
        pass

    @abc.abstractmethod
    def get_type(self, query):
        pass

    @abc.abstractmethod
    def get_aka(self, query):
        pass
