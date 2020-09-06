'''
Created on Aug 25, 2014

@author: Changlong
'''

import zmq
import threading
import struct
import logging
import datetime
import time
import traceback
from twisted.internet import threads
from twisted.internet.defer import DeferredLock
from DB import SBDB, SBDB_ORM
# from SBPS import ProtocolReactor
from Utils import Util, Config
import Command
from Command import RedirectNotify

PORT_PUBSUB = 5557
TTYPE_GATEWAY = b'g'
TTYPE_HUMAN = b'h'
TTYPE_SERVER = b's'
TTYPE_ONLINEOFFLINE = b'o'
TTYPE_MANAGEMENT = b'm'
OPER_ONLINE = b'on'
OPER_OFFLINE = b'off'
OPER_REQUEST = b'req'
OPER_RESPONSE = b'res'
OPER_REDIRECT = b'direct'
OPER_LOAD = b'load'
OPER_SIMPLECONTROL = b'sc'
OPER_SIMPLECONTROLRESP = b'scr'

context = None
socketSubscribe = None
socketPublish = None
listSubscribedServer = []

lockPublish = DeferredLock()
protocolInternal = None

# online status,map of relayer_id-->CConnectionIterm
dictRelayerServer = {}
# online status,map of client_id-->CConnectionIterm
dictClientServer = {}
# server connection count, map of server_id-->connection count
dictServerLoad = {}
# server information, map of server_id-->ip address
dictServerInfo = {}
MyServerID = 0
MyServerAddress = Config.domain_name
MyServerType = SBDB.CV_TYPE_SERVER_FUNCTION
lockMaps = threading.RLock()
Max_Server_Load = 5000


def str_to_bytes(o):
    if isinstance(o, str):
        return o.encode('utf-8')
    else:
        return o


def bytes_to_str(o):
    if isinstance(o, bytes):
        return o.decode('utf-8')
    else:
        return o


def AddSubscribeServer(listServer, serverType=SBDB.CV_TYPE_SERVER_FUNCTION):
    global MyServerID, MyServerAddress, MyServerType, dictServerInfo
    listMyIPs = Util.GetMachineIPs()
    for server in listServer:
        if server[1] not in listSubscribedServer:
            socketSubscribe.connect("tcp://%s:%d" % (server[1], PORT_PUBSUB))
            listSubscribedServer.append(server[1])
        dictServerInfo[server[0]] = server[2]
        if server[1] in listMyIPs:
            MyServerID = server[0]
            MyServerAddress = server[1]
            if MyServerType != SBDB.CV_TYPE_SERVER_SUPERVISION:
                MyServerType = serverType


def PublishMessageCallback(lock, data):
    try:
        data = list(map(str_to_bytes, data))
        socketPublish.send_multipart(data)
        print("data sent of InternalMessage %s:%s:%s (%s%s%s)" %
              (data[0], data[1], data[2], Util.asscii_string(data[0]),
               Util.asscii_string(data[1]), Util.asscii_string(data[2])))
    except:
        print(traceback.format_exc())
    finally:
        lock.release()


def PublishMessage(head, fromInfo, body):
    lockPublish.acquire().addCallback(PublishMessageCallback,
                                      [head, fromInfo, body])


class CInternalMessage(object):
    '''
    classdocs
    '''

    def __init__(self, head=b"|0|0||", from_filter=b"|0|0", body=b""):
        '''
        Constructor
        '''
        [
            self.destType, self.destId, self.destSock, self.operation,
            self.addition
        ] = head.split(b'|')
        self.destId = int(self.destId)
        self.destSock = int(self.destSock)
        [self.fromType, self.fromId, self.fromSock] = from_filter.split(b'|')
        self.fromId = int(self.fromId)
        self.fromSock = int(self.fromSock)
        self.body = body

    def SetParam(self, destType, destId, destSock, operation, addition,
                 fromType, fromId, fromSock):
        self.destType, self.destId, self.destSock, self.operation, self.addition, self.fromType, self.fromId, self.fromSock = destType, destId, destSock, operation, addition, fromType, fromId, fromSock

    def Send(self):
        PublishMessage(
            "%s|%d|%d|%s|%s" % (bytes_to_str(self.destType), self.destId,
                                self.destSock, bytes_to_str(self.operation),
                                bytes_to_str(self.addition)),
            "%s|%d|%d" % (bytes_to_str(self.fromType), self.fromId,
                          self.fromSock), self.body)


class CConnectionItem(object):
    def __init__(self, server_id):
        '''
        Constructor
        '''
        self.dt_active = datetime.datetime.now()
        self.server_id = server_id


def NotifyTerminalStatus(terminal_type,
                         terminal_id,
                         terminal_sock,
                         operation,
                         balance="y"):
    global MyServerID
    message = CInternalMessage()
    message.SetParam(TTYPE_ONLINEOFFLINE + terminal_type, terminal_id,
                     terminal_sock, operation, balance, TTYPE_SERVER,
                     MyServerID, 0)
    message.Send()


def RedirectHumanTo(client_id, server_id):
    message = CInternalMessage()
    message.SetParam(TTYPE_HUMAN, client_id, 0, OPER_REDIRECT,
                     dictServerInfo[server_id], TTYPE_SERVER, MyServerID, 0)
    message.Send()


def RegistFilter(destType, destId=None):
    head = ""
    if isinstance(destType, bytes):
        destType = destType.decode('utf-8')
    if destId is None:
        head = destType
    else:
        head = "%s|%d|" % (destType, destId)
    print("RegistFilter %s......................................" % (head))
    socketSubscribe.setsockopt(zmq.SUBSCRIBE, head.encode('utf-8'))


def UnregistFilter(destType, destId=None):
    head = ""
    if isinstance(destType, bytes):
        destType = destType.decode('utf-8')
    if destId is None:
        head = destType
    else:
        head = "%s|%d|" % (destType, destId)
    print("UnRegistFilter %s ***************************************" % (head))
    socketSubscribe.setsockopt(zmq.UNSUBSCRIBE, head.encode('utf-8'))


def CheckMapsByActiveTime():
    with lockMaps:
        for relayer_id in dictRelayerServer.keys():
            if dictRelayerServer[relayer_id].dt_active < datetime.datetime.now(
            ) - datetime.timedelta(seconds=Config.time_heartbeat):
                # timeout
                item = dictRelayerServer.pop(relayer_id)
                dictServerLoad[
                    item.server_id] = dictServerLoad[item.server_id] - 1
        for client_id in dictClientServer.keys():
            if dictClientServer[client_id].dt_active < datetime.datetime.now(
            ) - datetime.timedelta(seconds=Config.time_heartbeat):
                # timeout
                item = dictClientServer.pop(client_id)
                dictServerLoad[
                    item.server_id] = dictServerLoad[item.server_id] - 1


def LoadMapsFromDatabase():

    dictServerLoad.clear()
    dictClientServer.clear()
    dictRelayerServer.clear()

    with SBDB.session_scope() as session:
        for client in session.query(SBDB_ORM.Client).filter(
                SBDB_ORM.Client.dt_active > datetime.datetime.now() -
                datetime.timedelta(seconds=Config.time_heartbeat)).all():
            item = CConnectionItem(client.server_id)
            item.dt_active = client.dt_active
            dictClientServer[client.id] = item
            dictServerLoad[item.server_id] = dictServerLoad.get(
                item.server_id, 0) + 1

        for relayer in session.query(SBDB_ORM.Relayer).filter(
                SBDB_ORM.Relayer.dt_active > datetime.datetime.now() -
                datetime.timedelta(seconds=Config.time_heartbeat)).all():
            item = CConnectionItem(relayer.server_id)
            item.dt_active = relayer.dt_active
            dictRelayerServer[relayer.id] = item
            dictServerLoad[item.server_id] = dictServerLoad.get(
                item.server_id, 0) + 1


def ThreadCheckMaps():
    from SBPS import ProtocolReactor
    LoadMapsFromDatabase()

    waitTimes = Config.time_heartbeat / 2
    n = 0
    while not ProtocolReactor.bReactorStopped:
        if 0 == n:
            CheckMapsByActiveTime()

            # notify all server to load config
            message = CInternalMessage()
            message.SetParam(TTYPE_SERVER, 0, 0, OPER_LOAD, "", TTYPE_SERVER,
                             MyServerID, 0)
            message.Send()
        n = (n + 1) % waitTimes
        time.sleep(2)


def traced_ThreadCheckMaps():
    import traceback
    try:
        ThreadCheckMaps()
    except:
        print(traceback.format_exc())


def RunOnlineMessage(message):
    global dictRelayerServer, dictClientServer, dictServerLoad, dictServerInfo, lockMaps
    with lockMaps:
        if len(message.destType) < 2:
            return
        if (message.destType[1] == TTYPE_HUMAN):
            session_key = message.destId
            if message.operation == OPER_ONLINE:
                if session_key not in dictClientServer:
                    dictServerLoad[message.fromId] = dictServerLoad.get(
                        message.fromId, 0) + 1
                    dictClientServer[session_key] = CConnectionItem(
                        message.fromId)
                else:
                    dictClientServer[session_key].server_id = message.fromId
                    dictClientServer[
                        session_key].dt_active = datetime.datetime.now()
                if message.addition.lower() == 'y':  # need redirect checking
                    listRelayeres = SBDB.GetRelayeresByAccountId(
                        message.fromId)
                    target_server_id = -1
                    for relayer_id in listRelayeres:
                        if relayer_id in dictRelayerServer:
                            # if there is a relayer connected to that account is connected on the same server with the account connection, don't redirect the account connection
                            if dictRelayerServer[
                                    relayer_id].server_id == message.fromId:
                                return
                            else:
                                target_server_id = relayer_id
                    if target_server_id > -1:
                        RedirectHumanTo(message.destId, target_server_id)
            elif message.operation == OPER_OFFLINE:
                if session_key in dictClientServer:
                    dictServerLoad[message.fromId] = dictServerLoad.get(
                        message.fromId, 0) - 1
                    dictClientServer.pop(session_key)
        elif (message.destType[1] == TTYPE_GATEWAY):
            if message.operation == OPER_ONLINE:
                if message.destId not in dictRelayerServer:
                    dictServerLoad[message.fromId] = dictServerLoad.get(
                        message.fromId, 0) + 1
                    dictRelayerServer[message.destId] = CConnectionItem(
                        message.fromId)
                else:
                    dictRelayerServer[
                        message.destId].server_id = message.fromId
                    dictRelayerServer[
                        message.destId].dt_active = datetime.datetime.now()
            elif message.operation == OPER_OFFLINE:
                if message.destId in dictRelayerServer:
                    dictServerLoad[message.fromId] = dictServerLoad.get(
                        message.fromId, 0) - 1
                    dictRelayerServer.pop(message.destId)


def RunTransmitMessage(message):
    if message.operation in [OPER_REQUEST, OPER_RESPONSE]:
        length, command_id = struct.unpack("!2I", message.body[:8])
        command = None
        try:
            command = Command.dicInt_Type[command_id](message.body,
                                                      protocolInternal)
        except Exception as e:
            print(traceback.format_exc())
            logging.error(
                "build command exception in protocolInternal transport %d: %s :%s",
                id(protocolInternal.transport), str(e),
                Util.asscii_string(message.body))
            command = None
        command.internalMessage = message
        # threads.deferToThread(command.Run)
        command.Run()
    elif message.operation == OPER_REDIRECT:
        notify = RedirectNotify.CRedirectNotify(
            client_id=message.destId, addr=message.addition)
        notify.Notify()


def RunServerMessage(message):
    if message.operation == OPER_LOAD:
        AddSubscribeServer(SBDB.GetServers(SBDB.CV_TYPE_SERVER_FUNCTION))
        AddSubscribeServer(
            SBDB.GetServers(SBDB.CV_TYPE_SERVER_SUPERVISION),
            SBDB.CV_TYPE_SERVER_SUPERVISION)
        if message.fromType == TTYPE_MANAGEMENT:
            message.fromType = TTYPE_SERVER
            message.Send()
        SBDB.UpdateActiveTimeServer(MyServerID)


dictProcessor = {
    ord('o'): RunOnlineMessage,
    ord('g'): RunTransmitMessage,
    ord('h'): RunTransmitMessage,
    ord('s'): RunServerMessage
}

countPendingCmd = 0
lockPendingCmd = threading.RLock()


def ProcessMessage(head, from_filter, body):
    try:
        global countPendingCmd, lockPendingCmd
        print("data received of InternalMessage %s:%s:%s (%s%s%s)" %
              (head, from_filter, body, Util.asscii_string(head),
               Util.asscii_string(from_filter), Util.asscii_string(body)))
        message = CInternalMessage(head, from_filter, body)
        typeMessage = message.destType[0]
        print("44444", message, typeMessage, type(typeMessage), dictProcessor)
        if typeMessage in dictProcessor:
            print("55555", typeMessage)
            dictProcessor[typeMessage](message)
        with lockPendingCmd:
            countPendingCmd = countPendingCmd - 1
    except:
        print(traceback.format_exc())


def Run():
    from SBPS import ProtocolReactor
    global socketSubscribe, socketPublish, MyServerType, context, countPendingCmd, lockPendingCmd

    context = zmq.Context()
    socketSubscribe = context.socket(zmq.SUB)
    socketPublish = context.socket(zmq.PUB)

    AddSubscribeServer(SBDB.GetServers(SBDB.CV_TYPE_SERVER_FUNCTION))
    AddSubscribeServer(
        SBDB.GetServers(SBDB.CV_TYPE_SERVER_SUPERVISION),
        SBDB.CV_TYPE_SERVER_SUPERVISION)

    socketPublish.bind("tcp://*:%d" % (PORT_PUBSUB))
    RegistFilter(TTYPE_SERVER, 0)
    RegistFilter(TTYPE_SERVER, MyServerID)

    if MyServerType == SBDB.CV_TYPE_SERVER_SUPERVISION:
        threading.Thread(target=traced_ThreadCheckMaps, daemon=True).start()
        RegistFilter(TTYPE_ONLINEOFFLINE)

    while not ProtocolReactor.bReactorStopped:
        try:
            [head, from_filter, body] = socketSubscribe.recv_multipart()
            with lockPendingCmd:
                if countPendingCmd < Config.count_connection:
                    countPendingCmd = countPendingCmd + 1
                    threads.deferToThread(ProcessMessage, head, from_filter,
                                          body)
        except:
            print(traceback.format_exc())

    print("InternalMessage.Run() returned")


def traced_Run():
    try:
        Run()
    except:
        print(traceback.format_exc())


def Stop():
    try:
        if protocolInternal:
            protocolInternal.timer.cancel()
        socketPublish.close()
        socketSubscribe.close()
        context.term()
    except:
        print(traceback.format_exc())
