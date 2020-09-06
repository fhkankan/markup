'''
Created on 2013-8-12

@author: Changlong
'''
from BaseCommand import CBaseCommand
from sqlalchemy.exc import SQLAlchemyError
from DB import SBDB, SBDB_ORM
from Command import BaseCommand
import logging


class CModifyApartment(CBaseCommand):
    '''
    classdocs 
    '''
    command_id = 0x00040006

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
                apartment_id = self.body[BaseCommand.PN_ID]
                apartment_name = self.body[BaseCommand.PN_NAME]
                respond = self.GetResp()
                if apartment_id is None or apartment_name is None:
                    respond.SetErrorCode(BaseCommand.CS_PARAMLACK)
                else:
                    try:
                        SBDB.IncreaseVersion(session, apartment_id)
                        apartment = session.query(SBDB_ORM.Apartment).filter(
                            SBDB_ORM.Apartment.id == apartment_id).first()
                        apartment.name = apartment_name
                        respond.body[
                            BaseCommand.PN_VERSION] = apartment.version
                        session.commit()
                    except SQLAlchemyError as e:
                        respond.SetErrorCode(BaseCommand.CS_DBEXCEPTION)
                        logging.error("transport %d:%s",
                                      id(self.protocol.transport), e)
                        session.rollback()
            respond.Send()
