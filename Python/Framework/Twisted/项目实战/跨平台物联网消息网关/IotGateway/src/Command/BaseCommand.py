'''
Created on 2013-8-5

@author: Changlong
'''
from Utils import Util, Config
import logging
import struct
import threading
import time
import json
import traceback
from twisted.internet import reactor
#from SBPS import InternalMessage

PN_ERRORSTRING = "error"
PN_DEVICEID = "dev_id"
PN_VERSION = 'version'
PN_RELAYERID = "relayer_id"
PN_DEVMODEL = "dev_model"
PN_DEVCODE = "dev_code"
PN_DEVNAME = "name"
PN_RESULT = "result"
PN_TERMINALTYPE = "terminal_type"
PN_SB_CODE = "sb_code"
PN_USERNAME = "user_name"
PN_PASSWORD = "password"
PN_EMAIL = 'email'
PN_APARTMENTNAME = "name"
PN_ID = "id"
PN_NAME = "name"
PN_APARTMENTID = "apartment_id"
PN_FLAGNOTIFICATION = "flag_notification"
PN_DEVKEYS = "dev_keys"
PN_DEVSEQ = "dev_key_seq"
PN_DEVVALUE = "value"
PN_UPDATEDATA = "update_data"
PN_CONTACTORNAME = "name"
PN_MOBLEPHONE = "mobile_phone"
PN_SCENENAME = "name"
PN_SCENEID = "id"
PN_SCENECONTENTS = "scene_contents"
PN_ARMSTATE = "arm_state"
PN_SCENE_ID = "scene_id"
PN_DEVTYPE = "dev_type"
PN_DEVICEKEYS = "device_keys"
PN_DEVICES = "devices"
PN_RELAYERS = "relayers"
PN_SCENES = "scenes"
PN_CONTACTORS = "contactors"
PN_APARTMENTINFO = "apartment_info"
PN_LANGUAGENAME = "language_name"
PN_APARTMENTS = "apartments"
PN_SPECIALSCENE = "function"
PN_OS = "os"
PN_TOKEN = "token"
PN_LASTTOKEN = "last_token"
PN_ISALARM = "is_alarm"
PN_RCVALARM = "rcv_alarm"
PN_BALANCE = "balance"
PN_TERMINALCODE = "terminal_code"
PN_ADDR = "addr"
PN_STARTDT = "start_dt"
PN_ENDDT = "end_dt"
PN_ALARMTYPES = "alarm_types"
PN_DEVICEIDS = "device_ids"
PN_APARTMENTIDS = "apartment_ids"
PN_PAGE = "page"
PN_ALARMS = "alarms"
PN_DEVICETYPENAME = "device_type_name"
PN_DT = "dt"

PV_ROLE_RELAYER = "relayer"
PV_ROLE_HUMAN = "human"
PV_ROLE_INTERNAL = "internal"
PV_SCENE_ALLLIGHTON = "all_light_on"
PV_SCENE_ALLLIGHTOFF = "all_light_off"
PV_SCENE_GASSENSOR = "gas_sensor"
PV_SCENE_SPECIFIED = "spcified"

PV_E_OK = 0
PV_E_USERPASS = 1
PV_E_ROLE = 2

PV_ARM_OFF = 1
PV_ARM_ON = 2

PV_OS_ANDROID = "android"
PV_OS_IOS = "ios"

ALARM_TYPE_NO = 0
ALARM_TYPE_SET = 10
ALARM_TYPE_ALWAYS = 20

gas_sensor_model = "2690S"
gas_actuator_model = "2141A"
gas_actuator_value = 1

CS_OK = 0
CS_DEVICEEXIST = 1
CS_DEVICEMODEL = 2
CS_DBEXCEPTION = 3
CS_PARAMLACK = 4
CS_USERNAME = 5
CS_EMAIL = 6
CS_PASSWORD = 7
CS_RELAYEROFFLINE = 8
CS_CONTROLFAIL = 9
CS_NODEVICE = 10
CS_MOBILEPHONE = 11
CS_NORELAYER = 12
CS_UNAUTHORIZED = 13
CS_LOGINFAIL = 14
CS_TRYLATER = 15
CS_NOTFOUNDEMAIL = 16
CS_SERVERBUSY = 17
CS_RELAYERRESPTIMEOUT = 18

dictErrorString = {
    CS_DEVICEEXIST: "device has exist",
    CS_DEVICEMODEL: "wrong model",
    CS_DBEXCEPTION: "database exception",
    CS_PARAMLACK: "parameter do not enough",
    CS_USERNAME: "user_name has been existed",
    CS_EMAIL: "email has been existed",
    CS_PASSWORD: "password un-correct",
    CS_RELAYEROFFLINE: "relayer off-line",
    CS_CONTROLFAIL: "control fail",
    CS_NODEVICE: "no device",
    CS_MOBILEPHONE: "phone number existed",
    CS_NORELAYER: "no relayer",
    CS_UNAUTHORIZED: "un-authenticated",
    CS_LOGINFAIL: "authentication fail",
    CS_TRYLATER: "try later...",
    CS_NOTFOUNDEMAIL: "email not found",
    CS_SERVERBUSY: "server busy",
    CS_RELAYERRESPTIMEOUT: "relayer resp timeout"
}


class CBaseCommand(object):
    '''
    classdocs
    '''

    sequence_latest = 0
    lock_sequence = threading.RLock()
    HEAD_LEN = 16

    def __init__(self, data=None, protocol=None):
        '''
        Constructor
        '''
        self.protocol = protocol
        self.role = None
        self.tmActivate = time.time()
        self.body = {}
        self.relayer_id = 0
        if data is not None:
            if isinstance(data, str):
                data = data.encode('utf-8')
            self.data = data
            self.command_len, self.command_id, self.command_status, self.command_seq = struct.unpack(
                "!4I", data[:CBaseCommand.HEAD_LEN])
            if self.command_len > CBaseCommand.HEAD_LEN:
                self.body = json.loads(data[CBaseCommand.HEAD_LEN:])
        else:
            self.command_len = CBaseCommand.HEAD_LEN
            self.command_status = 0
            self.command_id = type(self).command_id
            self.command_seq = self.GetNextSeq()
        self.internalMessage = None

    #################need override by sub-class###############
    command_id = 0x0
    TypeResp = object

    def Run(self):
        print("run: ", self.__class__)
        if self.protocol is not None and self.protocol.role != PV_ROLE_INTERNAL:
            try:
                self.protocol.timer.cancel()
                # self.protocol.timer=threading.Timer(Config.time_heartbeat,self.protocol.timeout)
                # self.protocol.timer
                self.protocol.timer = reactor.callLater(
                    Config.time_heartbeat, self.protocol.timeout)
            except Exception:
                pass
            self.protocol.tmActivate = time.time()

    ##########################################################

    def Authorized(self):
        return 'role' in dir(
            self.protocol) and self.protocol.role is not None and len(
                self.protocol.role) > 0

    def GetResp(self):
        TypeResp = type(self).TypeResp
        command_id = Util.int32_to_uint32(self.command_id) | 0x80000000
        return TypeResp(
            protocol=self.protocol, request=self, command_id=command_id)

    def Send_Real(self):
        reactor.callFromThread(self.protocol.transport.write, self.data)
        print("data sent in transport %d : %s (%s)" % (id(
            self.protocol.transport), Util.asscii_string(self.data),
            self.data))

    def Send(self, internalMessage=None):
        body_string = ""
        if len(self.body) > 0:
            body_string = json.dumps(self.body)
        self.command_len = CBaseCommand.HEAD_LEN + len(body_string)
        if isinstance(self, CBaseRespCommand):
            self.command_seq = self.request.command_seq
        self.data = struct.pack("!4I", self.command_len, self.command_id,
                                self.command_status, self.command_seq)
        self.data = self.data + body_string.encode('utf-8')

        if internalMessage is None:
            if self.protocol.HeaderTagType == 1:
                self.data = self.protocol.factory.SBMP_HEADERTAG + self.data
            self.Send_Real()
            return

        internalMessage.body = self.data
        internalMessage.Send()

    def SendResp(self):
        cmd_resp = self.GetResp()
        cmd_resp.Send()

    def SendUnauthorizedResp(self):
        cmd_resp = self.GetResp()
        cmd_resp.SetErrorCode(CS_UNAUTHORIZED)
        cmd_resp.Send()

    def GetNextSeq(self):
        CBaseCommand.lock_sequence.acquire()
        if CBaseCommand.sequence_latest >= 0x7fffffff:
            CBaseCommand.sequence_latest = 0
        next_sequence = CBaseCommand.sequence_latest + 1
        CBaseCommand.sequence_latest = next_sequence
        CBaseCommand.lock_sequence.release()
        return next_sequence

    def __str__(self, *args, **kwargs):
        return "<%s(%d,%d,%d,%d,%s>" % (
            self.__class__, self.command_len, self.command_id,
            self.command_status, self.command_seq, json.dumps(self.body))

    def IsOKResp(self, resp):
        return self.command_seq == resp.command_seq and resp.command_id - self.command_id == 0x80000000 and resp.command_status == 0

    def SetErrorCode(self, command_status, error_string=None):
        self.command_status = command_status
        if command_status != CS_OK:
            if error_string is None:
                error_string = dictErrorString.get(command_status,
                                                   "unknown error")
            self.body[PN_ERRORSTRING] = error_string


class CBaseRespCommand(CBaseCommand):
    command_id = 0x80000000

    def __init__(self, data=None, protocol=None, request=None,
                 command_id=None):
        CBaseCommand.__init__(self, data, protocol)
        self.request = request
        if command_id is not None:
            self.command_id = command_id


CBaseCommand.TypeResp = CBaseRespCommand


class CMesscodeCommand(CBaseCommand):
    '''
    this enclose a stream which could not be decode
    '''

    def __init__(self, data, protocol=None):
        '''
        Constructor
        '''
        CBaseCommand.__init__(self, data, protocol)

    def Run(self):

        logging.error("MesscodeCommand in transport %d : %s",
                      id(self.protocol.transport),
                      Util.asscii_string(self.data))
