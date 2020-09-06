'''
Created on 2013-7-31

@author: Changlong
'''
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

import SBPS.ProtocolReactor as ProtocolReactor
import logging
logging.basicConfig(
    filename='example.log',
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s")

if __name__ == '__main__':
    logging.info("Relayer Server starting...")
    print("Relayer Server starting...")
    ProtocolReactor.Run()  # run until stop
