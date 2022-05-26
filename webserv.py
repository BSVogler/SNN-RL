import asyncio
import json
import pickle
from typing import Optional

import numpy as np
from numpy import random
import tornado.web
import tornado.httpserver
import tornado.ioloop
import tornado.websocket as ws
from tornado.options import define, options
import time

define('port', default=4041, help='port to listen on')

data = None

class Web_socket_handler(ws.WebSocketHandler):
    '''
    This class handles the websocket channel
    '''
    live_web_sockets = set()

    @classmethod
    def route_urls(cls):
        return [(r'/', cls, {}), ]

    def simple_init(self):
        self.last = time.time()
        self.stop = False

    def open(self):
        '''
            client opens a connection
        '''
        self.simple_init()
        self.live_web_sockets.add(self)
        #print("New client connected")
        # gen gakedata
        # weights = np.random.random(size=(1024)).tolist()
        if data is not None:
            self.write_message(json.dumps(data.tolist()))

    def on_message(self, message):
        '''
            Message received on the handler
        '''
        #print("received message {}".format(message))
        #self.write_message("said {}".format(message))
        if message == "sender":
            self.live_web_sockets.remove(self)
        else:
            newdata = pickle.loads(message)
            Web_socket_handler.push_update(newdata)
        self.last = time.time()

    @classmethod
    def push_update(cls, datanew):
        global data
        data=datanew
        cls.send_message(json.dumps(data.tolist()))

    @classmethod
    def send_message(cls, message):
        """broadcasts to all clients"""
        removable = set()
        for ws in cls.live_web_sockets:
            if not ws.ws_connection or not ws.ws_connection.stream.socket:
                removable.add(ws)
            else:
                ws.write_message(message)
        for ws in removable:
            cls.live_web_sockets.remove(ws)

    def on_close(self):
        '''
            Channel is closed
        '''
        #print("connection is closed")
        # self.loop.stop()

    def check_origin(self, origin):
        return True


def initiate_server(loop=None, experiment=None):
    # allows beeing run as a thread
    if loop:
        asyncio.set_event_loop(loop)
    expinst = experiment
    # create a tornado application and provide the urls
    handler = Web_socket_handler.route_urls()
    app = tornado.web.Application(handler)

    # setup the server
    global server
    server = tornado.httpserver.HTTPServer(app)
    server.listen(options.port)

    # start io/event loop
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    initiate_server()
