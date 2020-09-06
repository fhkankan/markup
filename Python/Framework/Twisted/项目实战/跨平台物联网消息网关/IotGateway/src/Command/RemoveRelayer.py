'''
Created on 2013-8-12

@author: Changlong
'''
from BaseCommand import CBaseCommand
from sqlalchemy.exc import SQLAlchemyError
from DB import SBDB, SBDB_ORM
from Command import BaseCommand
import logging
from sqlalchemy import and_


class CRemoveRelayer(CBaseCommand):
    '''
    classdocs 
    '''
    command_id = 0x00030007

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
                sb_id = self.body[BaseCommand.PN_RELAYERID]
                apartment_id = self.body[BaseCommand.PN_APARTMENTID]
                respond = self.GetResp()
                if sb_id is None:
                    respond.SetErrorCode(BaseCommand.CS_PARAMLACK)
                else:
                    try:
                        apartment = SBDB.IncreaseVersion(session, apartment_id)
                        session.query(SBDB_ORM.Apartment_Relayer).filter(
                            and_(
                                SBDB_ORM.Apartment_Relayer.relayer_id == sb_id,
                                SBDB_ORM.Apartment_Relayer.apartment_id ==
                                apartment_id)).delete()
                        respond.body[
                            BaseCommand.PN_VERSION] = apartment.version
                        session.commit()
                    except SQLAlchemyError as e:
                        respond.SetErrorCode(BaseCommand.CS_DBEXCEPTION)
                        logging.error("transport %d:%s",
                                      id(self.protocol.transport), e)
                        session.rollback()
            respond.Send()
