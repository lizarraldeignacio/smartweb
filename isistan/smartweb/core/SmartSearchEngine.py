import abc


__author__ = 'ignacio'


class SmartSearchEngine(object, metaclass=abc.ABCMeta):
    #
    # Search engine abstract class

    @abc.abstractmethod
    def publish(self, service):
        pass

    @abc.abstractmethod
    def publish_services(self, service_list):
        pass
        
    @abc.abstractmethod
    def unpublish(self, service):
        pass

    @abc.abstractmethod
    def number_of_services(self):
        pass

    @abc.abstractmethod
    def find(self, query):
        pass
