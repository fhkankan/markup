'''
Created on 2013-8-12

@author: Changlong
'''

import logging
logging.basicConfig(
    filename='example_client.log',
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s")

from Utils import Util
if Util.isWindows():
    from twisted.internet import iocpreactor
    iocpreactor.install()
elif Util.isMac():
    from twisted.internet import kqreactor
    kqreactor.install()
else:
    from twisted.internet import epollreactor
    epollreactor.install()

from twisted.internet.protocol import Protocol, ClientFactory
from twisted.internet import reactor
connection_count = 0


class PoetryProtocol(Protocol):

    poem = ''

    def dataReceived(self, data):
        print(data)
        reactor.callLater(3 * 60, self.transport.write, str(self.index))

    def connectionMade(self):
        global connection_count
        Protocol.connectionMade(self)
        connection_count += 1
        print("connection made:", connection_count)
        logging.info("connection made: %d", id(connection_count))
        self.index = connection_count
        self.transport.write(str(self.index))
        if self.index < 10000:
            oppo = self.transport.getPeer()
            reactor.callLater(0, reactor.connectTCP, oppo.host, oppo.port,
                              self.factory)

    def connectionLost(self, reason):
        logging.info("connection closed: %d", id(self.transport))


class PoetryClientFactory(ClientFactory):

    protocol = PoetryProtocol

    def __init__(self):
        pass


def get_poetry(host, port):
    """
    Download a poem from the given host and port and invoke

      callback(poem)

    when the poem is complete.
    """
    factory = PoetryClientFactory()
    reactor.callLater(0.11, reactor.connectTCP, host, port, factory)
    #reactor.connectTCP(host, port, factory)


def poetry_main():
    get_poetry("42.96.156.14", 9632)

    reactor.run()


if __name__ == '__main__':
    poetry_main()
