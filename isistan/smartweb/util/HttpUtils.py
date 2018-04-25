import urllib2


class HttpUtils:

    @staticmethod
    def http_request(url, headers=None):
        proxy_handler = urllib2.ProxyHandler({})
        opener = urllib2.build_opener(proxy_handler)
        if headers is not None:
            opener.addheaders = headers
        return opener.open(url).read()
