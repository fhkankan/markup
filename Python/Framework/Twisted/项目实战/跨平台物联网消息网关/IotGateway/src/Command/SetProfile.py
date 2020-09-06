'''
Created on 2013-9-5

@author: Changlong
'''

from BaseCommand import CBaseCommand
from sqlalchemy.exc import SQLAlchemyError
from DB import SBDB, SBDB_ORM
from Command import BaseCommand
import logging


class CSetProfile(CBaseCommand):
    '''
    classdocs 
    '''
    command_id = 0x00050001

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
                password = self.body.get(BaseCommand.PN_PASSWORD)
                email = self.body.get(BaseCommand.PN_EMAIL)
                language_name = self.body.get(BaseCommand.PN_LANGUAGENAME)
                mobile_phone = self.body.get(BaseCommand.PN_MOBLEPHONE)
                respond = self.GetResp()
                try:
                    account = session.query(SBDB_ORM.Account).filter(
                        SBDB_ORM.Account.id == self.protocol.account_id).one()
                    if password is not None:
                        account.password = password
                    if email is not None:
                        account.email = email
                    if language_name is not None:
                        for language in session.query(SBDB_ORM.Language).all():
                            if language.language == language_name:
                                account.language_id = language.id
                    if mobile_phone is not None:
                        account.mobile_phone = mobile_phone
                    session.commit()
                except SQLAlchemyError as e:
                    respond.SetErrorCode(BaseCommand.CS_DBEXCEPTION)
                    logging.error("transport %d:%s",
                                  id(self.protocol.transport), e)
                    session.rollback()
            respond.Send()
