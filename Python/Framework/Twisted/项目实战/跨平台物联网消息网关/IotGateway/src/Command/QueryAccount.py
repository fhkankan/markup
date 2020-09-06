'''
Created on 2013-9-3

@author: Changlong
'''

from BaseCommand import CBaseCommand
from sqlalchemy.exc import SQLAlchemyError
from DB import SBDB, SBDB_ORM
from Command import BaseCommand
import logging


class CQueryAccount(CBaseCommand):
    '''
    classdocs 
    '''
    command_id = 0x00010003

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
                account_id = self.protocol.account_id
                respond = self.GetResp()
                try:
                    account = session.query(SBDB_ORM.Account).filter(
                        SBDB_ORM.Account.id == account_id).one()
                    respond.body[BaseCommand.
                                 PN_LANGUAGENAME] = account.language.language
                    respond.body[BaseCommand.PN_USERNAME] = account.user_name
                    respond.body[
                        BaseCommand.PN_MOBLEPHONE] = account.mobile_phone
                    respond.body[BaseCommand.PN_EMAIL] = account.email
                    '''
                    # move this parameter to Authorize Command
                    listApartment=[]
                    for apartment in account.apartments:
                        elementApartment={}
                        elementApartment[BaseCommand.PN_ID]=apartment.id
                        listApartment.append(elementApartment)
                    respond.body[BaseCommand.PN_APARTMENTS]=listApartment
                    '''

                except SQLAlchemyError as e:
                    respond.SetErrorCode(BaseCommand.CS_DBEXCEPTION)
                    logging.error("transport %d:%s",
                                  id(self.protocol.transport), e)
                    session.rollback()
            respond.Send()
