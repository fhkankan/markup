#!/usr/bin/python
# coding=utf-8
'''
Created on 2013-8-21

@author: Changlong
'''

from BaseCommand import CBaseCommand
from sqlalchemy.exc import SQLAlchemyError
from DB import SBDB, SBDB_ORM
from Command import BaseCommand
from sqlalchemy import and_
import logging


class CAddRelayer(CBaseCommand):
    '''
    classdocs
    '''
    command_id = 0x00020007

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
            apartment_id = self.body[BaseCommand.PN_APARTMENTID]
            sb_code = self.body[BaseCommand.PN_SB_CODE]
            sb_name = self.body[BaseCommand.PN_NAME]
            # account_id=self.protocol.account_id
            respond = self.GetResp()
            with SBDB.session_scope() as session:
                if session.query(SBDB_ORM.Relayer).join(
                        SBDB_ORM.Apartment_Relayer).filter(
                            and_(
                                SBDB_ORM.Relayer.uni_code == sb_code,
                                SBDB_ORM.Apartment_Relayer.apartment_id ==
                                apartment_id)).first() is not None:
                    respond.SetErrorCode(BaseCommand.CS_DEVICEEXIST)
                try:
                    relayer_id = SBDB.GetRelayerIdForcely(sb_code)
                    apartment = SBDB.IncreaseVersion(session, apartment_id)
                    apartment_relayer = SBDB_ORM.Apartment_Relayer()
                    apartment_relayer.apartment_id = apartment_id
                    apartment_relayer.relayer_id = relayer_id
                    apartment_relayer.name = sb_name
                    '''
                    relayer=SBDB_ORM.Relayer()
                    relayer.apartment_id=apartment_id
                    relayer.name=sb_name
                    relayer.uni_code=sb_code            
                    session.add(relayer)
                    '''
                    session.add(apartment_relayer)
                    respond.body[BaseCommand.PN_VERSION] = apartment.version
                    session.commit()
                    #                 if session.query(SBDB_ORM.Device).join(SBDB_ORM.Relayer).filter(SBDB_ORM.Relayer.id==relayer.id).first():
                    #                     respond.body[BaseCommand.PN_UPDATEDATA]=True
                    respond.body[BaseCommand.PN_ID] = relayer_id
                except SQLAlchemyError as e:
                    respond.SetErrorCode(BaseCommand.CS_DBEXCEPTION)
                    logging.error("transport %d:%s",
                                  id(self.protocol.transport), e)
                    session.rollback()
            respond.Send()
