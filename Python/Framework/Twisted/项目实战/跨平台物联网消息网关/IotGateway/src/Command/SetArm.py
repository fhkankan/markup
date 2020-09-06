# -*- coding: utf-8 -*-
'''
Created on 2013-9-5

@author: Changlong
'''

from BaseCommand import CBaseCommand
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_
from DB import SBDB, SBDB_ORM
from Command import BaseCommand
import logging
import datetime
import threading
from Utils import Util


class CSetArm(CBaseCommand):
    '''
    classdocs 
    '''
    command_id = 0x00060002

    def __init__(self, data=None, protocol=None):
        '''
        Constructor
        '''
        CBaseCommand.__init__(self, data, protocol)

    def Run(self):
        with self.protocol.lockCmd:
            if not self.Authorized():
                self.SendUnauthorizedResp()
                return
            CBaseCommand.Run(self)
            with SBDB.session_scope() as session:
                arm_state = self.body[BaseCommand.PN_ARMSTATE]
                apartment_id = self.body.get(BaseCommand.PN_APARTMENTID)
                respond = self.GetResp()
                try:
                    apartment = session.query(SBDB_ORM.Apartment).filter(
                        SBDB_ORM.Apartment.id == apartment_id).first()
                    apartment.arm_state = arm_state
                    apartment.dt_arm = datetime.datetime.now()
                    session.commit()
                    self.PushToClients(session, apartment.dt_arm)
                except SQLAlchemyError as e:
                    respond.SetErrorCode(BaseCommand.CS_DBEXCEPTION)
                    logging.error("transport %d:%s",
                                  id(self.protocol.transport), e)
                    session.rollback()
            respond.Send()

    def PushToClients(self, session, time_):
        arm_state = self.body[BaseCommand.PN_ARMSTATE]
        listIOS = []
        language = 0
        for client in session.query(SBDB_ORM.Client).filter(
                and_(SBDB_ORM.Client.account_id == self.protocol.account_id,
                     SBDB_ORM.Client.os == BaseCommand.PV_OS_IOS)):
            if len(client.device_token) > 10:
                listIOS.append(client.device_token)
            if language == 0:
                language = client.account.language_id
        session = None

        strTime = time_.strftime("%Y-%m-%d %H:%M:%S")
        if language == 2:
            if arm_state == BaseCommand.PV_ARM_OFF:
                message = "房间于%s撤防!" % strTime
            else:
                message = "房间于%s布防!" % strTime
        else:
            if arm_state == BaseCommand.PV_ARM_OFF:
                message = "The apartment is armed ON at %s !" % strTime
            else:
                message = "The apartment is armed OFF at %s !" % strTime

        if len(listIOS) > 0:
            threading.Thread(
                target=Util.push_ios,
                args=(listIOS, "notification", message),
                daemon=True).start()
