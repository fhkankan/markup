'''
Created on 2013-7-31

@author: Changlong
'''

from twisted.internet.protocol import Protocol, ServerFactory
from twisted.internet import reactor, threads, ssl
from twisted.internet.defer import DeferredLock
import twisted.internet.error as twistedError
import traceback
import struct
import sys
import os
sys.path.insert(
    0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.path.pardir))
import Command
import logging
import time
from DB import SBDB
import threading
from Utils import Util, Config
from SBPS import InternalMessage


class SBProtocol(Protocol):
    '''
    Relayer Protocol
    '''

    connection_count = 0
    countPendingCmd = 0

    def __init__(self):
        '''
        Constructor
        '''
        self.m_buffer = b""
        self.lockBuffer = DeferredLock()
        self.tmActivate = time.time()
        self.dictWaitResp = {}
        self.lock_dictWaitResp = threading.RLock()
        self.dictControlling = {}
        self.cond_dictControlling = threading.Condition()
        # self.timer=threading.Timer(Config.time_heartbeat,self.timeout)
        # self.timer.start()
        self.timer = reactor.callLater(Config.time_heartbeat, self.timeout)
        self.lockCmd = threading.RLock()
        self.HeaderTagType = -1  # -1: not decided, 0: no header_tag, 1: has header_tag
        self.rcv_alarm = "False"
        self.role = ""

    def dataReceived(self, data):
        try:
            Protocol.dataReceived(self, data)
            self.lockBuffer.acquire().addCallback(self.AddDataAndDecode, data)
        except:
            print(traceback.format_exc())

    def connectionMade(self):
        #print("a connection made: ", id(self.transport), self.transport.getPeer().host)
        ip = self.transport.getPeer().host
        #         if ip.find("10.")!=0:
        #             logging.info("a connection made:%s,%s ", id(self.transport), ip)
        #         pass
        #
        with self.factory.lockPendingCmd:
            SBProtocol.connection_count = SBProtocol.connection_count + 1
            if SBProtocol.connection_count > Config.count_connection:
                self.transport.loseConnection()
                print("close connection due to reaching connection limit.")

    def RunCommand(self, command):
        try:
            with self.factory.lockPendingCmd:
                SBProtocol.countPendingCmd = SBProtocol.countPendingCmd - 1
            command.Run()
        except:
            print(traceback.format_exc())

    def AddDataAndDecode(self, lock, data):
        try:
            print("data received in transport %d : %s (%s)" % (id(
                self.transport), Util.asscii_string(data), data))
            self.m_buffer += data
            while len(self.m_buffer
                      ) >= Command.BaseCommand.CBaseCommand.HEAD_LEN:
                self.m_buffer, command, = self.Decode(self.m_buffer)
                if command is None:
                    break

                # the maximum pending command is set to equal with connection count, one command for one connection by average
                if SBProtocol.countPendingCmd < Config.count_connection / 100:
                    threads.deferToThread(self.RunCommand, command)
                    with self.factory.lockPendingCmd:
                        SBProtocol.countPendingCmd = SBProtocol.countPendingCmd + 1
                else:
                    cmd_resp = command.GetResp()
                    cmd_resp.SetErrorCode(Command.BaseCommand.CS_SERVERBUSY)
                    cmd_resp.Send()
        except:
            print(traceback.format_exc())
        finally:
            lock.release()

    def Decode(self, data):
        '''
        return a tuple: new data,command
        '''
        if self.HeaderTagType < 0:  # not decide
            if data[:4] == self.factory.SBMP_HEADERTAG:
                self.HeaderTagType = 1
            else:
                self.HeaderTagType = 0

        if self.HeaderTagType == 1:
            tag_position = data.find(self.factory.SBMP_HEADERTAG)
            if tag_position < 0:
                return (data, None)
            data = data[tag_position + 4:]  # remove head tag
        length, command_id = struct.unpack("!2I", data[:8])
        command = None
        if length <= len(data):
            command_data = data[:length]
            if command_id in Command.dicInt_Type:
                try:
                    command = Command.dicInt_Type[command_id](command_data,
                                                              self)
                except Exception as e:
                    print(traceback.format_exc())
                    logging.error(
                        "build command exception in transport %d: %s :%s",
                        id(self.transport), str(e),
                        Util.asscii_string(command_data))
                    command = None
            else:
                command = Command.BaseCommand.CMesscodeCommand(
                    command_data, self)
            data = data[length:]
        else:
            if self.HeaderTagType == 1:
                # if command is not completed, add the head tag again
                data = self.factory.SBMP_HEADERTAG + data
        return (data, command)

    def connectionLost(self, reason=twistedError.ConnectionDone):

        if self.role == Command.BaseCommand.PV_ROLE_HUMAN:
            InternalMessage.UnregistFilter(InternalMessage.TTYPE_HUMAN,
                                           self.client_id)
            InternalMessage.NotifyTerminalStatus(
                InternalMessage.TTYPE_HUMAN, self.client_id, id(
                    self.transport), InternalMessage.OPER_OFFLINE, 'n')
        elif self.role == Command.BaseCommand.PV_ROLE_RELAYER:
            InternalMessage.UnregistFilter(InternalMessage.TTYPE_GATEWAY,
                                           self.relayer_id)
            InternalMessage.NotifyTerminalStatus(InternalMessage.TTYPE_GATEWAY,
                                                 self.relayer_id, 0,
                                                 InternalMessage.OPER_OFFLINE)

        try:
            self.timer.cancel()
        except Exception:
            pass
        #print ("connection lost:",id(self.transport),reason)
        self.releaseFromDict()

        with self.factory.lockPendingCmd:
            SBProtocol.connection_count = SBProtocol.connection_count - 1
        Protocol.connectionLost(self, reason=reason)

    def timeout(self):
        if self.role != Command.BaseCommand.PV_ROLE_INTERNAL:
            self.transport.loseConnection()

    def isDeadSession(self):
        return time.time() - self.tmActivate > Config.time_heartbeat

    def releaseFromDict(self):
        with self.factory.lockDict:
            if 'role' not in dir(self):
                return
            if self.role == Command.BaseCommand.PV_ROLE_RELAYER:
                if self.relayer_id in self.factory.dictRelayer:
                    if self.factory.dictRelayer[self.relayer_id] == self:
                        self.factory.dictRelayer.pop(self.relayer_id)
            elif self.role == Command.BaseCommand.PV_ROLE_HUMAN:
                for relayerId in SBDB.GetRelayerIDsByAccountId(
                        self.account_id):
                    if relayerId in self.factory.dictAccounts:
                        listAccount = self.factory.dictAccounts[relayerId]
                        if self in listAccount:
                            listAccount.remove(self)
                            if len(listAccount) <= 0:
                                self.factory.dictAccounts.pop(relayerId)


class SBProtocolFactory(ServerFactory):
    protocol = SBProtocol

    def __init__(self):
        self.lockDict = DeferredLock()
        self.dictRelayer = {}  # key:relayerid,value:SBProtocol
        self.dictAccounts = {}  # key:relayerid,value:array of SBProtocol
        self.lockDict = threading.RLock()
        self.SBMP_HEADERTAG = struct.pack("2B", 0x01, 0xBB)

        self.lockPendingCmd = threading.RLock()

    def GetAccountProtocol(self, relayer_id, client_id):
        with self.lockDict:
            if relayer_id in self.dictAccounts:
                for clientProtocol in self.dictAccounts[relayer_id]:
                    if clientProtocol.client_id == client_id:
                        return clientProtocol
        return None


class EchoProtocol(Protocol):
    '''
    Echo Protocol
    '''

    def __init__(self):
        '''
        Constructor
        '''

    def dataReceived(self, data):
        Protocol.dataReceived(self, data)
        print("data received: ", data, ",", id(self.transport))
        self.transport.write(data)

    def connectionMade(self):
        ip = self.transport.getPeer().host
#         if ip.find("10.")!=0:
#             print ("a connection made: ", id(self.transport))

    def connectionLost(self, reason=None):
        ip = self.transport.getPeer().host


#         if ip.find("10.")!=0:
#             print ("a connection lost: ", id(self.transport))


class EchoProtocolFactory(ServerFactory):
    protocol = EchoProtocol

    def __init__(self):
        pass


instance_SBProtocolFactory = None
bReactorStopped = False


def Run(withListen=True):

    global instance_SBProtocolFactory, bReactorStopped
    instance_SBProtocolFactory = SBProtocolFactory()

    InternalMessage.protocolInternal = SBProtocol()
    InternalMessage.protocolInternal.role = "internal"
    InternalMessage.protocolInternal.factory = instance_SBProtocolFactory
    threading.Thread(target=InternalMessage.traced_Run, daemon=True).start()

    if withListen:
        reactor.listenTCP(9630,
                          instance_SBProtocolFactory)  # @UndefinedVariable

        cert = None
        with open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), os.path.pardir,
                    'server.pem')) as keyAndCert:
            cert = ssl.PrivateCertificate.loadPEM(keyAndCert.read())
        reactor.listenSSL(9631, instance_SBProtocolFactory, cert.options())
        reactor.listenTCP(9632, EchoProtocolFactory())  # @UndefinedVariable

    try:
        reactor.run()  # @UndefinedVariable
    except:
        pass
    InternalMessage.Stop()
    bReactorStopped = True
    try:
        # reactor.stop()
        import sys
        sys.exit(0)
    except:
        pass


if __name__ == '__main__':
    pro = SBProtocolFactory().buildProtocol("127.0.0.1")
    pro.dataReceived("abc__")
    pro.dataReceived("abc=============================")
    print(type(12)(34))
