'''
Created on 2013-8-21

@author: Changlong
'''
from BaseCommand import CBaseCommand
from sqlalchemy.exc import SQLAlchemyError
from DB import SBDB, SBDB_ORM
from Command import BaseCommand
import logging


class CAddApartment(CBaseCommand):
    '''
    classdocs
    '''
    command_id = 0x00020006

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
            name = self.body[BaseCommand.PN_APARTMENTNAME]
            respond = self.GetResp()
            with SBDB.session_scope() as session:
                try:
                    #SBDB.IncreaseVersion(session, self.protocol.account)
                    apartment = SBDB_ORM.Apartment()
                    apartment.account_id = self.protocol.account_id
                    apartment.arm_state = BaseCommand.PV_ARM_OFF
                    apartment.name = name
                    apartment.scene_id = None
                    apartment.version = 0
                    session.add(apartment)
                    respond.body[BaseCommand.PN_VERSION] = apartment.version
                    session.commit()
                    respond.body[BaseCommand.PN_ID] = apartment.id
                except SQLAlchemyError as e:
                    respond.SetErrorCode(BaseCommand.CS_DBEXCEPTION)
                    logging.error("transport %d:%s",
                                  id(self.protocol.transport), e)
                    session.rollback()
            respond.Send()
