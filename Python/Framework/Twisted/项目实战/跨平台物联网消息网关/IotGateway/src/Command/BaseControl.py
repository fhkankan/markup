'''
Created on 2013-8-26

@author: Changlong
'''

from BaseCommand import CBaseCommand
from sqlalchemy.exc import SQLAlchemyError
from Command import BaseCommand
from DB import SBDB_ORM, SBDB
from sqlalchemy import and_
import threading
import logging
from Utils import Config
from SBPS import InternalMessage
from sqlalchemy import distinct
from twisted.internet import reactor


class CDeviceCmd(object):
    def __init__(self, dev_model, dev_code, dev_seq, value):
        self.dev_model = dev_model
        self.dev_code = dev_code
        self.dev_seq = dev_seq
        self.value = value
        self.body = {}
        self.result = -1
        self.bTriedRelayerMain = False

    def __eq__(self, other):
        if isinstance(other, SBDB_ORM.DeviceKeyCode):
            return self.dev_model == other.device_key.device_model.name and self.dev_code == other.device.uni_code and self.dev_seq == other.device_key.seq
        elif isinstance(other, CDeviceCmd):
            return self.dev_model == other.dev_model and self.dev_code == other.dev_code and self.dev_seq == other.dev_seq and self.value == other.value
        else:
            object._eq__(self, other)

    def __hash__(self):
        return hash(self.dev_model + self.dev_code + str(self.dev_seq))


class CBaseControl(CBaseCommand):
    '''
    classdocs
    '''

    def __init__(self, data=None, protocol=None):
        '''
        Constructor
        '''
        CBaseCommand.__init__(self, data, protocol)
        self.dictWaitingRelayerControls = {
        }  # the map of relayer_id-->array of CDeviceCmd
        self.dictSendingRelayerControls = {}
        self.dictFinishedRelayerControls = {}
        self.lock = threading.RLock()
        self.bFinished = False
        self.requireCommand = None
        self.timer = None

    def Run(self):
        with self.protocol.lockCmd:
            if not self.Authorized():
                self.SendUnauthorizedResp()
                return
            CBaseCommand.Run(self)
            self.initDictRelayerControls()
            with self.lock:
                self.FeedbackIfFinished()

    def initByDeviceCmdList(self, listDC):
        self.DeviceKeyCode = ""
        self.dictWaitingRelayerControls = {}
        setUnique = set()
        with SBDB.session_scope() as session:

            if self.internalMessage:
                self.account_id = self.internalMessage.fromId
            else:
                self.account_id = self.protocol.account_id
            try:
                for dc in listDC:
                    if dc in setUnique:
                        continue
                    setUnique.update((dc, ))
                    if not self.internalMessage:
                        for s, in session.query(
                                SBDB_ORM.Apartment_Relayer.relayer_id
                        ).join(SBDB_ORM.Apartment).join(
                                SBDB_ORM.ApartmentDevice).join(
                                    SBDB_ORM.Device).filter(
                                        and_(
                                            SBDB_ORM.Apartment.account_id ==
                                            self.account_id,
                                            SBDB_ORM.Device.uni_code ==
                                            dc.dev_code)).all():
                            print("tracing: ", s, self.account_id, dc.dev_code)
                            if s in self.dictWaitingRelayerControls:
                                self.dictWaitingRelayerControls[s].append(dc)
                            else:
                                self.dictWaitingRelayerControls[s] = [
                                    dc,
                                ]
                    else:
                        relayer_id = self.internalMessage.destId
                        if relayer_id in self.dictWaitingRelayerControls:
                            self.dictWaitingRelayerControls[relayer_id].append(
                                dc)
                        else:
                            self.dictWaitingRelayerControls[relayer_id] = [
                                dc,
                            ]
            except SQLAlchemyError as e:
                logging.error("transport %d:%s", id(self.protocol.transport),
                              e)
                session.rollback()

    def SendBatch(self):
        for relayer_id in self.dictWaitingRelayerControls.keys():
            deviceCmds = self.dictWaitingRelayerControls.pop(relayer_id)
            if len(deviceCmds) <= 0:
                continue

            if self.internalMessage:
                with self.protocol.factory.lockDict:
                    sb_protocol = self.protocol.factory.dictRelayer.get(
                        relayer_id)
                if sb_protocol is None:
                    self.dictFinishedRelayerControls[relayer_id] = deviceCmds
                    continue
            else:
                sb_protocol = InternalMessage.protocolInternal
            self.dictSendingRelayerControls[relayer_id] = deviceCmds
            bSend = False
            for deviceCmd in deviceCmds:
                if deviceCmd.result == 0:
                    continue
                bSend = True
                deviceCmd.result = -1
                #from ControlDevice import CControlDevice
                # control_device=CControlDevice(protocol=sb_protocol)
                control_device = self.getCommand(deviceCmd)
                control_device.protocol = sb_protocol
                control_device.relayer_id = relayer_id
                if sb_protocol.role == BaseCommand.PV_ROLE_INTERNAL:
                    with sb_protocol.lock_dictWaitResp:
                        # sb_protocol.dictWaitResp[(relayer_id<<32)+control_device.command_seq]=control_device
                        control_device.requireTransport = id(
                            self.protocol.transport)
                        sb_protocol.dictWaitResp[
                            (control_device.requireTransport << 32) +
                            control_device.command_seq] = control_device
                else:
                    with sb_protocol.lock_dictWaitResp:
                        sb_protocol.dictWaitResp[
                            control_device.command_seq] = control_device
                control_device.requireCommand = self
                control_device.Send()
                # threading.Timer(Config.timeout_relayer_control,control_device.timeout).start()
                # control_device.timer=reactor.callLater(Config.timeout_relayer_control,control_device.timeout)
            if not bSend:
                self.dictFinishedRelayerControls[
                    relayer_id] = self.dictSendingRelayerControls.pop(
                        relayer_id)
                continue
            break

    def FinishOne(self, relayer_id, control_device, respond):
        with self.lock:
            # deviceCmds=self.dictSendingRelayerControls.get(relayer)
            deviceCmds = None
            for key in self.dictSendingRelayerControls.keys():
                if key == relayer_id:
                    deviceCmds = self.dictSendingRelayerControls[key]
            if deviceCmds is not None:
                for deviceCmd in deviceCmds:
                    if control_device.body[BaseCommand.
                                           PN_DEVMODEL] == deviceCmd.dev_model and control_device.body[BaseCommand.
                                                                                                       PN_DEVCODE] == deviceCmd.dev_code and control_device.body[BaseCommand.
                                                                                                                                                                 PN_DEVSEQ] == deviceCmd.dev_seq:
                        if deviceCmd.result != 0:
                            deviceCmd.result = respond.command_status
                            deviceCmd.body = respond.body

                            # if it's failed, and have not try other relayer in the same account: add these relayer to dictWaitingRelayerControls
                            if deviceCmd.result != 0 and not deviceCmd.bTriedRelayerMain and self.protocol.role == BaseCommand.PV_ROLE_HUMAN:
                                with SBDB.session_scope() as session:
                                    deviceCmd.bTriedRelayerMain = True

                                    listRelayerId = []
                                    for s in session.query(
                                            SBDB_ORM.Relayer
                                    ).join(SBDB_ORM.Apartment_Relayer).join(
                                            SBDB_ORM.Apartment
                                    ).join(SBDB_ORM.Account).join(
                                            SBDB_ORM.ApartmentDevice
                                    ).join(SBDB_ORM.Device).join(
                                            SBDB_ORM.DeviceModel).filter(
                                                and_(
                                                    SBDB_ORM.Account.id ==
                                                    self.protocol.account_id,
                                                    SBDB_ORM.Device.uni_code ==
                                                    deviceCmd.dev_code,
                                                    SBDB_ORM.DeviceModel.name
                                                    == deviceCmd.dev_model,
                                                    SBDB_ORM.Relayer.id !=
                                                    relayer_id)):
                                        if s.id not in listRelayerId:
                                            listRelayerId.append(s.id)
                                    if len(listRelayerId) > 0:
                                        deviceCmd.result = -1
                                        deviceCmds.remove(deviceCmd)
                                        for s in listRelayerId:
                                            print("relayer:", s)
                                            if s in self.dictWaitingRelayerControls:
                                                self.dictWaitingRelayerControls[
                                                    s].append(deviceCmd)
                                            else:
                                                self.dictWaitingRelayerControls[
                                                    s] = [
                                                        deviceCmd,
                                                ]

                        # if it's succeed finished by other relayer, set this relayer as this device's mainSueprbox
                        if deviceCmd.result == 0 and deviceCmd.bTriedRelayerMain:
                            with SBDB.session_scope() as session:
                                for apartment_device in session.query(
                                        SBDB_ORM.ApartmentDevice
                                ).join(SBDB_ORM.Device).join(
                                        SBDB_ORM.Apartment).join(
                                            SBDB_ORM.Account).filter(
                                                and_(
                                                    SBDB_ORM.Account.id ==
                                                    self.protocol.account_id,
                                                    SBDB_ORM.Device.uni_code ==
                                                    deviceCmd.dev_code)):
                                    apartment_device.relayer_id = relayer_id
                                session.commit()
                        break
            self.FeedbackIfFinished()

    def CheckFinished(self):
        if len(self.dictSendingRelayerControls.keys()) > 0:
            for relayer_id in list(self.dictSendingRelayerControls.keys()):
                for deviceCmd in self.dictSendingRelayerControls[relayer_id]:
                    if deviceCmd.result < 0:
                        return False
                self.dictFinishedRelayerControls[
                    relayer_id] = self.dictSendingRelayerControls.pop(
                        relayer_id)
        if len(self.dictWaitingRelayerControls.keys()) <= 0:
            return True
        else:
            self.SendBatch()
            return self.CheckFinished()

    def FeedbackIfFinished(self):
        if not self.CheckFinished():
            return
        self.Feedback()

    def Feedback(self):
        if self.bFinished:
            return
        respond = self.GetResp()
        #         if self.protocol is not None and self.protocol.role ==BaseCommand.PV_ROLE_HUMAN:
        if self.protocol is not None:
            result = -1
            # if one control command fail, the total command fail
            for relayer_id in self.dictFinishedRelayerControls.keys():
                for dc in self.dictFinishedRelayerControls[relayer_id]:
                    if result <= 0:
                        result = dc.result
                        respond.body = dc.body

            if result == -1:
                respond.SetErrorCode(BaseCommand.CS_RELAYEROFFLINE)
            else:
                respond.SetErrorCode(result)

            interMessage = None
            if self.protocol.role == BaseCommand.PV_ROLE_INTERNAL:
                interMessage = InternalMessage.CInternalMessage()
                interMessage.SetParam(
                    self.internalMessage.fromType, self.internalMessage.fromId,
                    self.internalMessage.fromSock,
                    InternalMessage.OPER_RESPONSE, "",
                    self.internalMessage.destType, self.internalMessage.destId,
                    self.internalMessage.destSock)

            respond.Send(interMessage)
        self.bFinished = True
        return respond

    def timeout(self):
        with self.protocol.cond_dictControlling:
            request = self.protocol.dictControlling.pop(self.command_seq, None)
            if request is not None:
                logging.debug(
                    "call self.protocol.cond_dictControlling.notify() due to timeout in protocol %d",
                    id(self.protocol.transport))
                self.protocol.cond_dictControlling.notify()
            else:
                logging.debug(
                    "fail to self.protocol.dictControlling.pop(%d) in protocol %d",
                    self.command_seq, id(self.protocol.transport))
            self.timer = None
        relayer_id = None
        if self.protocol.role == BaseCommand.PV_ROLE_INTERNAL:
            with self.protocol.lock_dictWaitResp:
                # request=self.protocol.dictWaitResp.pop((self.relayer_id<<32)+self.command_seq,None)
                request = self.protocol.dictWaitResp.pop(
                    (self.requireTransport << 32) + self.command_seq, None)
            relayer_id = self.relayer_id
        else:
            with self.protocol.lock_dictWaitResp:
                request = self.protocol.dictWaitResp.pop(
                    self.command_seq, None)
            relayer_id = self.protocol.relayer_id
        if request is None:
            return  # this request has been feedbacked
        respond = request.GetResp()
        respond.command_status = BaseCommand.CS_RELAYERRESPTIMEOUT
        respond.body = {}
        requireCommand = request.requireCommand
        requireCommand.FinishOne(relayer_id, request, respond)
        request.requireCommand = None

    # ----------------subclass override--------------------
    def initDictRelayerControls(self):
        pass

    def getCommand(self, deviceCmd):
        return None

    # -----------------------------------------------------

    def Send(self):
        maxcmd_flow_control = 1000
        if self.protocol.role == BaseCommand.PV_ROLE_RELAYER:
            maxcmd_flow_control = Config.maxcmd_relayer_control
        while True:
            with self.protocol.cond_dictControlling:
                if len(self.protocol.dictControlling.keys()
                       ) > maxcmd_flow_control:
                    logging.debug(
                        "call self.protocol.cond_dictControlling.wait() due to reach maxcmd in protocol %d",
                        id(self.protocol.transport))
                    self.protocol.cond_dictControlling.wait()
                elif self.command_seq in self.protocol.dictControlling:
                    logging.debug(
                        "call self.protocol.cond_dictControlling.wait() due to same command_seq in protocol %d",
                        id(self.protocol.transport))
                    self.protocol.cond_dictControlling.wait()
                else:
                    interMessage = None
                    self.protocol.dictControlling[self.command_seq] = self
                    self.timer = reactor.callLater(
                        Config.timeout_relayer_control, self.timeout)

                    # if isinstance(self.requireCommand,BaseCommand.CBaseCommand):
                    if self.requireCommand is not None and self.requireCommand.internalMessage is None:
                        interMessage = InternalMessage.CInternalMessage()
                        interMessage.SetParam(
                            InternalMessage.TTYPE_GATEWAY, self.relayer_id, 0,
                            InternalMessage.OPER_REQUEST, "",
                            InternalMessage.TTYPE_HUMAN,
                            self.requireCommand.protocol.client_id,
                            id(self.requireCommand.protocol.transport))

                    CBaseCommand.Send(self, interMessage)

                    break
