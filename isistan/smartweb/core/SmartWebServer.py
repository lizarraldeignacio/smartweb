import sys
import argparse
import urllib.request, urllib.parse, urllib.error

from isistan.smartweb.core.SearchEngine import SearchEngine

from flask import Flask
from flask import request

app = Flask(__name__)

__author__ = 'ignacio'


BASE_RESOURCE_LOCATION = '/isistan-smartweb/smartweb'
PUBLISH_SERVICES_PARAMETER = 'service_list'

engone = None

@app.route(BASE_RESOURCE_LOCATION + '/services', methods=['POST'])
def publish_services():
    services = request.form['service_list'].split(' ')
    engine.publish_services(services)
    return 'services published'

@app.route(BASE_RESOURCE_LOCATION + '/services/<query>', methods=['GET'])
def find_services(query):
    result = ' '.join(engine.find(query))
    return result

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-engine", help="loads a specific engine, default=WSQBESearchEngine", type=str, default='WSQBESearchEngine')
    parser.add_argument("-port", help="listen on a custom port, default=8080", type=int, default=8080)
    parser.add_argument("-conf", help="specifies a configuration file that will be passed to the engine, default=wsqbe_properties.cfg", 
    					type=str, default='wsqbe_properties.cfg')
    args = parser.parse_args()
    module_fullname = "isistan.smartweb.engine." + args.engine
    module = __import__(module_fullname, fromlist=[args.engine])
    global engine
    engine = getattr(module, args.engine)()
    engine.load_configuration(args.conf)
    
    app.run(port =  args.port, debug = False)

if __name__ == '__main__':
    run()



