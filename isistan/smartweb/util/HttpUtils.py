import urllib.request, urllib.error, urllib.parse


class HttpUtils:

    @staticmethod
    def http_request(url):
        proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy_handler)
        return opener.open(url, timeout=10000).read()
