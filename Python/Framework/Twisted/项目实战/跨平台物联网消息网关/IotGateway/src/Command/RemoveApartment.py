'''
Created on 2013-8-12

@author: Changlong
'''
from BaseCommand import CBaseCommand
from sqlalchemy.exc import SQLAlchemyError
from DB import SBDB, SBDB_ORM
from Command import BaseCommand
import logging
import traceback


class CRemoveApartment(CBaseCommand):
    '''
    classdocs 
    '''
    command_id = 0x00030006

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
                respond = self.GetResp()
                if apartment_id is None:
                    respond.SetErrorCode(BaseCommand.CS_PARAMLACK)
                else:
                    try:
                        apartment = session.query(SBDB_ORM.Apartment).join(
                            SBDB_ORM.Account, SBDB_ORM.Account.id ==
                            self.protocol.account_id).filter(
                                SBDB_ORM.Apartment.id == apartment_id).first()
                        session.delete(apartment)
                        respond.body[BaseCommand.PN_VERSION] = 0
                        session.commit()
                    except SQLAlchemyError as e:
                        print(traceback.format_exc())
                        respond.SetErrorCode(BaseCommand.CS_DBEXCEPTION)
                        logging.error("transport %d:%s",
                                      id(self.protocol.transport), e)
                        session.rollback()
            respond.Send()
