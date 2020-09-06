# -*- coding: utf-8 -*-
'''
Created on 2013-8-12

@author: Changlong
'''

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, and_
from sqlalchemy.orm import scoped_session, sessionmaker, undefer
from Utils import Config

engine = create_engine(Config.db_connection_string, echo=False)

SessionType = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))
import threading
import datetime
import logging
import string
from Utils import Util

CV_TYPE_SERVER_FUNCTION = 1
CV_TYPE_SERVER_SUPERVISION = 2


def GetSession():
    return SessionType()


from contextlib import contextmanager


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = GetSession()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


import SBDB_ORM
from sqlalchemy import or_, distinct
SBDB_ORM.metadata.bind = engine
SBDB_ORM.metadata.create_all()


def AddMetaData():
    with session_scope() as session:
        # Add Languages...
        lang = SBDB_ORM.Language(1, 'en-US')
        session.merge(lang)
        lang = SBDB_ORM.Language(2, 'zh-CN')
        session.merge(lang)
        lang = SBDB_ORM.Language(3, 'zh-TW')
        session.merge(lang)

        # Add Device Types...
        dt = SBDB_ORM.DeviceType(1, "灯光")
        session.merge(dt)
        dt = SBDB_ORM.DeviceType(2, "窗帘")
        session.merge(dt)
        dt = SBDB_ORM.DeviceType(3, "空调")
        session.merge(dt)

        # Add Device Models....
        dm = SBDB_ORM.DeviceModel(1, 1, '2111S')
        session.merge(dm)
        dm = SBDB_ORM.DeviceModel(2, 1, 'light-0203')
        session.merge(dm)
        dm = SBDB_ORM.DeviceModel(3, 2, 'curtain-3901')
        session.merge(dm)

        # Add Server
        server = SBDB_ORM.Server(1)
        session.merge(server)
        session.commit()


AddMetaData()


def GetRelayerIdForcely(relayer_code):
    with session_scope() as session:
        try:
            relayer = session.query(SBDB_ORM.Relayer).filter_by(
                uni_code=relayer_code).with_lockmode('update').first()
            if relayer is None:
                relayer = SBDB_ORM.Relayer()
                relayer.uni_code = relayer_code
                session.add(relayer)
            session.commit()
            return relayer.id
        except Exception as e:
            session.rollback()
            raise e


def GetDeviceForcely(session, device_code, model_name=None):

    device = session.query(SBDB_ORM.Device).filter_by(
        uni_code=device_code).with_lockmode('update').first()
    if device is None:
        model = GetDeviceModelByName(session, model_name)
        if model is None:
            return None
        device = SBDB_ORM.Device()
        device.uni_code = device_code
        device.device_model_id = model.id

        session.add(device)
    session.commit()
    return device


def GetAccount(session, username, password=None):
    account = None
    if password is None:
        account = session.query(SBDB_ORM.Account).filter(
            or_(SBDB_ORM.Account.user_name == username,
                SBDB_ORM.Account.email == username,
                SBDB_ORM.Account.mobile_phone == username)).first()
    else:
        account = session.query(SBDB_ORM.Account).filter(
            SBDB_ORM.Account.password == password).filter(
                or_(SBDB_ORM.Account.user_name == username,
                    SBDB_ORM.Account.email == username,
                    SBDB_ORM.Account.mobile_phone == username)).first()

    return account


def GetRelayerIDsByAccountId(account_id, session=None):
    release = False
    if session is None:
        session = GetSession()
        release = True
    RelayerIDs = []
    for relayerId, in session.query(
            SBDB_ORM.Apartment_Relayer.relayer_id).join(
                SBDB_ORM.Apartment).join(SBDB_ORM.Account).filter(
                    SBDB_ORM.Account.id == account_id).all():
        RelayerIDs.append(relayerId)
    if release:
        session.close()
    return RelayerIDs


def GetDeviceModelByName(session, model_name):
    for model in session.query(SBDB_ORM.DeviceModel).all():
        if model.name == model_name:
            return model
    return None


def GetDeviceKeyByModelAndSeq(obj_device_model, seq_number):
    for dev_key in obj_device_model.device_keys:
        if dev_key.seq == seq_number:
            return dev_key
    return None


def GetDeviceKeyCodeByDeviceCode(session, dev_code):
    return session.query(SBDB_ORM.DeviceKeyCode).filter(
        SBDB_ORM.DeviceKeyCode.key_code == dev_code).first()


#     return session.query(SBDB_ORM.DeviceModel).join(SBDB_ORM.Device).filter(SBDB_ORM.Device.uni_code==dev_code).first()


def IncreaseVersion(session, apartment_id):
    apartment = session.query(
        SBDB_ORM.Apartment).with_lockmode('update').filter(
            SBDB_ORM.Apartment.id == apartment_id).first()
    apartment.version = apartment.version + 1
    return apartment


def IncreaseVersions(session, relayer_id, apartment_id):
    if relayer_id == 0:
        return IncreaseVersion(session, apartment_id)
    else:
        apartments = session.query(
            SBDB_ORM.Apartment).with_lockmode('update').join(
                SBDB_ORM.Apartment_Relayer).filter(
                    SBDB_ORM.Apartment_Relayer.relayer_id == relayer_id).all()
    for apartment in apartments:
        apartment.version = apartment.version + 1
    for apartment in apartments:
        if apartment.id == apartment_id:
            return apartment
    return None


def GetServers(server_type=CV_TYPE_SERVER_FUNCTION):
    with session_scope() as session:
        listServer = []
        for server in session.query(SBDB_ORM.Server).filter(
                SBDB_ORM.Server.type == server_type).with_lockmode('update'):
            listServer.append((server.id, server.address,
                               server.extern_address))
    return listServer


def GetAccountIdByClientId(clientId):
    with session_scope() as session:
        account_id, = session.query(SBDB_ORM.Client.account_id).filter(
            SBDB_ORM.Client.id == clientId).first()
    return account_id


def GetRelayeresByAccountId(accountId):
    with session_scope() as session:
        listRelayeres = []
        for apartment_relayer in session.query(
                SBDB_ORM.Apartment_Relayer).join(SBDB_ORM.Apartment).filter(
                    SBDB_ORM.Apartment.account_id == accountId).all():
            listRelayeres.append(apartment_relayer.relayer_id)
    return listRelayeres


def GetActiveClientIdsByAccountId(accountId):
    with session_scope() as session:
        listClientIds = []
        for client_id, in session.query(SBDB_ORM.Client.id).join(
                SBDB_ORM.Account).filter(
                    and_(
                        SBDB_ORM.Account.id == accountId,
                        SBDB_ORM.Client.dt_active >
                        (datetime.datetime.now() - datetime.timedelta(
                            seconds=Config.time_heartbeat)))).all():
            listClientIds.append(client_id)
    print("GetActiveClientIdsByAccountId:", accountId, listClientIds)
    return listClientIds


def UpdateActiveTime(role_session, terminal_id, sock_=0):
    from Command import BaseCommand
    from SBPS import InternalMessage
    with session_scope() as session:
        if role_session == BaseCommand.PV_ROLE_HUMAN:
            session.query(SBDB_ORM.Client).filter(
                SBDB_ORM.Client.id == terminal_id).update({
                    SBDB_ORM.Client.dt_active:
                    datetime.datetime.now()
                })
            session.commit()
            InternalMessage.NotifyTerminalStatus(InternalMessage.TTYPE_HUMAN,
                                                 terminal_id, sock_,
                                                 InternalMessage.OPER_ONLINE)
        elif role_session == BaseCommand.PV_ROLE_RELAYER:
            session.query(SBDB_ORM.Relayer).filter(
                SBDB_ORM.Relayer.id == terminal_id).update({
                    SBDB_ORM.Relayer.dt_active:
                    datetime.datetime.now()
                })
            session.commit()
            InternalMessage.NotifyTerminalStatus(InternalMessage.TTYPE_GATEWAY,
                                                 terminal_id, 0,
                                                 InternalMessage.OPER_ONLINE)


def UpdateAuthTimeRelayer(relayer_id):
    from SBPS import InternalMessage
    InternalMessage.RegistFilter(InternalMessage.TTYPE_GATEWAY, relayer_id)
    with session_scope() as session:
        session.query(
            SBDB_ORM.Relayer).filter(SBDB_ORM.Relayer.id == relayer_id).update(
                {
                    SBDB_ORM.Relayer.dt_auth: datetime.datetime.now(),
                    SBDB_ORM.Relayer.server_id: InternalMessage.MyServerID
                })
        session.commit()
    InternalMessage.NotifyTerminalStatus(InternalMessage.TTYPE_GATEWAY,
                                         relayer_id, 0,
                                         InternalMessage.OPER_ONLINE)


def UpdateAuthTimeHuman(client_id, balance, sock_):
    from SBPS import InternalMessage
    InternalMessage.RegistFilter(InternalMessage.TTYPE_HUMAN, client_id)
    with session_scope() as session:
        session.query(
            SBDB_ORM.Client).filter(SBDB_ORM.Client.id == client_id).update({
                SBDB_ORM.Client.dt_auth:
                datetime.datetime.now()
            })
        session.commit()
    InternalMessage.NotifyTerminalStatus(InternalMessage.TTYPE_HUMAN,
                                         client_id, sock_,
                                         InternalMessage.OPER_ONLINE, balance)


def UpdateActiveTimeServer(serverId):
    with session_scope() as session:
        session.query(
            SBDB_ORM.Server).filter(SBDB_ORM.Server.id == serverId).update({
                SBDB_ORM.Server.dt_active:
                datetime.datetime.now()
            })
        session.commit()
