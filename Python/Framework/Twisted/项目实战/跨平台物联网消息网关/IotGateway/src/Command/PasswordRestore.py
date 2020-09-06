'''
Created on Jul 21, 2014

@author: Changlong
'''

from BaseCommand import CBaseCommand
from sqlalchemy.exc import SQLAlchemyError
from DB import SBDB, SBDB_ORM
from Command import BaseCommand
import logging
import datetime
import os
import threading
from Utils import Util, Config, emaillib


class CPasswordRestore(CBaseCommand):
    '''
    classdocs 
    '''
    command_id = 0x00060003

    def __init__(self, data=None, protocol=None):
        '''
        Constructor
        '''
        CBaseCommand.__init__(self, data, protocol)

    def Run(self):
        with self.protocol.lockCmd:
            CBaseCommand.Run(self)
            with SBDB.session_scope() as session:
                respond = self.GetResp()
                email = self.body.get(BaseCommand.PN_EMAIL, None)
                account_id, = session.query(SBDB_ORM.Account.id).filter(
                    SBDB_ORM.Account.email == email).first()
                if email is None:
                    respond.SetErrorCode(BaseCommand.CS_NOTFOUNDEMAIL)
                elif account_id is None:
                    respond.SetErrorCode(BaseCommand.CS_NOTFOUNDEMAIL)

    #             elif 'dt_restore_require' in dir(self.protocol) and (datetime.datetime.now()-self.protocol.dt_restore_require).seconds<Config.second_restore_require:
    #                 respond.SetErrorCode(BaseCommand.CS_TRYLATER)
                else:
                    try:
                        content = open(
                            os.path.join(Config.dir_local_static,
                                         "restore_confirm.html"), 'r').read()
                        url = Util.GenRestoreURL(account_id)
                        content = content.replace("{{url_restore}}", url)
                        #emaillib.SendEmail("Honeywell Smart Home: reset password confirm", content, [self.protocol.account.email])
                        threading.Thread(
                            target=emaillib.SendEmail,
                            args=(
                                "Honeywell Smart Home: reset password confirm",
                                content, [email]),
                            daemon=True).start()
                        self.protocol.dt_restore_require = datetime.datetime.now(
                        )
                    except SQLAlchemyError as e:
                        respond.SetErrorCode(BaseCommand.CS_DBEXCEPTION)
                        logging.error("transport %d:%s",
                                      id(self.protocol.transport), e)
                        session.rollback()
            respond.Send()
