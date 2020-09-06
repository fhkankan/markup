'''
Created on 2013-8-13

@author: Changlong
'''

from BaseCommand import CBaseCommand
import BaseCommand
from DB import SBDB, SBDB_ORM
from sqlalchemy import and_
from sqlalchemy.exc import SQLAlchemyError
import logging
import datetime
import random
from Utils import Util
from SBPS import InternalMessage
from twisted.internet import threads


class CAuthorize(CBaseCommand):
    '''
    classdocs
    '''
    command_id = 0x00000001

    def __init__(self, data=None, protocol=None):
        '''
        Constructor
        '''
        CBaseCommand.__init__(self, data, protocol)

    def Run(self):
        with self.protocol.lockCmd:
            CBaseCommand.Run(self)
            if 'role' in dir(self.protocol):
                self.protocol.releaseFromDict()
            role = self.body[BaseCommand.PN_TERMINALTYPE]
            resp = self.GetResp()
            resp.body[BaseCommand.PN_RESULT] = BaseCommand.PV_E_OK
            if role == BaseCommand.PV_ROLE_RELAYER:
                sb_code = self.body[BaseCommand.PN_SB_CODE]
                self.protocol.relayer_id = SBDB.GetRelayerIdForcely(sb_code)
                self.protocol.role = role
                with self.protocol.factory.lockDict:
                    self.protocol.factory.dictRelayer[
                        self.protocol.relayer_id] = self.protocol
                #logging.info("transport %d: relayer %s login pass",id(self.protocol.transport),sb_code)
                threads.deferToThread(SBDB.UpdateAuthTimeRelayer,
                                      self.protocol.relayer_id)
            elif role == BaseCommand.PV_ROLE_HUMAN:
                #self.protocol.account=SBDB.GetAccount(self.body[BaseCommand.PN_USERNAME], self.body[BaseCommand.PN_PASSWORD])
                with SBDB.session_scope() as session:
                    account = SBDB.GetAccount(
                        session, self.body[BaseCommand.PN_USERNAME])
                    if account is not None and not Util.check_password(
                            self.body[BaseCommand.PN_PASSWORD],
                            account.password):
                        account = None
                    if account is None:
                        resp.body[
                            BaseCommand.PN_RESULT] = BaseCommand.PV_E_USERPASS
                        resp.body[BaseCommand.
                                  PN_ERRORSTRING] = "user/password mismatch"
                        resp.SetErrorCode(BaseCommand.CS_LOGINFAIL)
                    else:
                        self.protocol.account_id = account.id
                        self.protocol.role = role
                        self.protocol.client_id = -1
                        self.protocol.rcv_alarm = self.body.get(
                            BaseCommand.PN_RCVALARM, "False")
                        listApartment = []
                        for apartment in account.apartments:
                            elementApartment = {}
                            elementApartment[BaseCommand.PN_ID] = apartment.id
                            listApartment.append(elementApartment)
                        resp.body[BaseCommand.PN_APARTMENTS] = listApartment
                        dictAccount = self.protocol.factory.dictAccounts
                        for relayerId in SBDB.GetRelayerIDsByAccountId(
                                self.protocol.account_id):
                            with self.protocol.factory.lockDict:
                                if relayerId in dictAccount:
                                    dictAccount[relayerId].append(
                                        self.protocol)
                                else:
                                    dictAccount[relayerId] = [
                                        self.protocol,
                                    ]

                        # set client information
                        os = self.body.get(BaseCommand.PN_OS,
                                           BaseCommand.PV_OS_IOS)
                        token = self.body.get(BaseCommand.PN_TOKEN)
                        last_token = self.body.get(BaseCommand.PN_LASTTOKEN)
                        balance = self.body.get(BaseCommand.PN_BALANCE)
                        # terminal_code=self.body.get(BaseCommand.PN_TERMINALCODE,datetime.datetime.now().strftime("%Y%m%d%H%M%S")+str(random.uniform(0,4000)))
                        if token == '' or token is None:
                            terminal_code = self.body.get(
                                BaseCommand.PN_TERMINALCODE,
                                datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                                + str(random.uniform(0, 4000)))
                        else:
                            terminal_code = self.body.get(
                                BaseCommand.PN_TERMINALCODE, token)

                        try:
                            # for temply use
                            if token is not None:
                                session.query(SBDB_ORM.Client).filter(
                                    and_(
                                        SBDB_ORM.Client.account_id !=
                                        self.protocol.account_id,
                                        SBDB_ORM.Client.device_token ==
                                        token.strip())).delete()
                            # -------------
                            if last_token is not None and token != last_token:
                                # session.query(SBDB_ORM.Client).filter(and_(SBDB_ORM.Client.account_id==self.protocol.account_id,SBDB_ORM.Client.device_token==last_token.strip())).delete()
                                session.query(SBDB_ORM.Client).filter(
                                    SBDB_ORM.Client.device_token ==
                                    last_token.strip()).delete()

                            client = session.query(SBDB_ORM.Client).filter(
                                SBDB_ORM.Client.terminal_code ==
                                terminal_code.strip()).first()
                            if client is None:
                                if token is not None:
                                    session.query(SBDB_ORM.Client).filter(
                                        SBDB_ORM.Client.device_token ==
                                        token.strip()).delete()
                                client = SBDB_ORM.Client()
                                client.device_token = token
                                client.enable_alarm = True
                                client.os = os
                                session.add(client)
                            client.account_id = self.protocol.account_id
                            client.terminal_code = terminal_code
                            client.dt_auth = client.dt_active = datetime.datetime.now(
                            )
                            client.server_id = InternalMessage.MyServerID
                            session.commit()
                            self.protocol.client_id = client.id
                            #logging.info("transport %d: user %s login pass ",id(self.protocol.transport),self.body[BaseCommand.PN_USERNAME])
                            if balance is None:
                                balance = 'n'
                            threads.deferToThread(SBDB.UpdateAuthTimeHuman,
                                                  client.id, balance,
                                                  id(self.protocol.transport))
                        except SQLAlchemyError as e:
                            resp.SetErrorCode(BaseCommand.CS_DBEXCEPTION)
                            logging.error("transport %d:%s",
                                          id(self.protocol.transport), e)
                            session.rollback()

            else:
                resp.body[BaseCommand.PN_RESULT] = BaseCommand.PV_E_ROLE
                resp.SetErrorCode(BaseCommand.CS_LOGINFAIL)

            resp.Send()

    def IsOKResp(self, resp):
        if not CBaseCommand.IsOKResp(self, resp):
            return False
        return resp.body[BaseCommand.PN_RESULT] == BaseCommand.PV_E_OK
