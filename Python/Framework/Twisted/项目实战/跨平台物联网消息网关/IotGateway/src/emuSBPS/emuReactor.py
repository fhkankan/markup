'''
Created on 2013-8-16

@author: Changlong
'''

from twisted.internet.protocol import Protocol, ClientFactory
from twisted.internet import reactor, threads
from twisted.internet.defer import DeferredLock
import twisted.internet.error as twistedError
import threading
import time
import struct
import Utils.Util as Util
import Command
import logging
from Utils import Config
protocolActive = None
waitList = {}

import traceback


def trace_err(e):
    print(traceback.format_exc())


class SBProtocol(Protocol):
    '''
    Relayer Protocol
    '''

    m_buffer = ""
    lockBuffer = DeferredLock()

    def __init__(self):
        '''
        Constructor
        '''
        self.HeaderTagType = 0

        self.m_buffer = b""
        self.lockBuffer = DeferredLock()
        self.tmActivate = time.time()
        self.dictWaitResp = {}
        self.lock_dictWaitResp = threading.RLock()
        self.dictControlling = {}
        self.cond_dictControlling = threading.Condition()
        self.timer = None

        self.lockCmd = threading.RLock()
        self.HeaderTagType = 0  # -1: not decided, 0: no header_tag, 1: has header_tag
        self.rcv_alarm = "False"
        self.role = ""

    def dataReceived(self, data):
        Protocol.dataReceived(self, data)
        self.lockBuffer.acquire().addCallback(self.AddDataAndDecode, data)

    def connectionMade(self):
        print("a connection made: ", id(self.transport))
        global protocolActive
        protocolActive = self

    def AddDataAndDecode(self, lock, data):
        print("data received in transport %d : %s (%s)" % (id(
            self.transport), Util.asscii_string(data), data))
        self.m_buffer += data
        while len(self.m_buffer) >= Command.BaseCommand.CBaseCommand.HEAD_LEN:
            self.m_buffer, command, = self.Decode(self.m_buffer)
            if command is None:
                break
            if command.command_seq in waitList:
                requestCmd = waitList[command.command_seq]
                requestCmd.respond = command
                requestCmd.lock.release()
                requestCmd.cond.acquire()
                requestCmd.cond.notifyAll()
                requestCmd.cond.release()

            d = threads.deferToThread(command.Run)
            d.addErrback(trace_err)
        lock.release()

    def Decode(self, data):
        '''
        return a tuple: new data,command
        '''
        length, command_id = struct.unpack("!2I", data[:8])
        command = None
        if length <= len(data):
            command_data = data[:length]
            if command_id in Command.dicInt_Type:
                try:
                    command = Command.dicInt_Type[command_id](command_data,
                                                              self)
                except Exception as e:
                    logging.error(
                        "build command exception in transport %d: %s :%s",
                        id(self.transport), str(e),
                        Util.asscii_string(command_data))
                    command = None
            else:
                command = Command.BaseCommand.CMesscodeCommand(
                    command_data, self)
            data = data[length:]
        return (data, command)

    def connectionLost(self, reason=twistedError.ConnectionDone):
        Protocol.connectionLost(self, reason=reason)
        print("connection lost:", id(self.transport), reason)


class SBProtocolFactory(ClientFactory):
    protocol = SBProtocol

    def __init__(self):
        pass


def waitLock(lock, cond, timeout=0x7fffffff):
    with cond:
        current_time = start_time = time.time()
        while current_time < start_time + timeout:
            if lock.acquire(False):
                return True
            else:
                second = timeout - current_time + start_time
                cond.wait(second)
                current_time = time.time()
    return False


def SendAndVerify(requestCmd):
    global protocolActive, waitList
    waitList[requestCmd.command_seq] = requestCmd
    requestCmd.protocol = protocolActive
    requestCmd.lock = threading.Lock()
    requestCmd.cond = threading.Condition()  # threading.Lock()
    requestCmd.lock.acquire()
    requestCmd.Send()
    bRet = False
    if waitLock(requestCmd.lock, requestCmd.cond,
                Config.timeout_relayer_control * 5):
        bRet = requestCmd.IsOKResp(requestCmd.respond)
        if bRet:
            print("pass: ", requestCmd.__class__)
        else:
            print("wrong Resp:", requestCmd.__class__)
    else:
        print("timeout Resp", requestCmd.__class__)

    waitList.pop(requestCmd.command_seq)
    return bRet


def Run():
    reactor.connectTCP("localhost", 9630,
                       SBProtocolFactory())  # @UndefinedVariable
    reactor.run()  # @UndefinedVariable

def Stop():
    protocolActive.transport.loseConnection()
    reactor.stop()