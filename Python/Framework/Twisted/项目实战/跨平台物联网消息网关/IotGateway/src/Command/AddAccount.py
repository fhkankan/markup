'''
Created on 2013-8-21

@author: Changlong
'''
from BaseCommand import CBaseCommand
from sqlalchemy.exc import SQLAlchemyError
from DB import SBDB, SBDB_ORM
from Command import BaseCommand
import logging
from Utils import Util
from sqlalchemy import or_


class CAddAccount(CBaseCommand):
    '''
    classdocs
    '''
    command_id = 0x00020005

    def __init__(self, data=None, protocol=None):
        '''
        Constructor
        '''
        CBaseCommand.__init__(self, data, protocol)

    def Run(self):
        with self.protocol.lockCmd:
            CBaseCommand.Run(self)
            user_name = self.body.get(BaseCommand.PN_USERNAME)
            if user_name is not None:
                user_name = user_name.strip()
            password = self.body[BaseCommand.PN_PASSWORD]
            email = self.body.get(BaseCommand.PN_EMAIL)
            if email is not None:
                email = email.strip()
            mobile_phone = self.body.get(BaseCommand.PN_MOBLEPHONE)
            if mobile_phone is not None:
                mobile_phone = mobile_phone.strip()
            respond = self.GetResp()
            with SBDB.session_scope() as session:
                if user_name is None and password is None and email is None:
                    respond.SetErrorCode(BaseCommand.CS_PARAMLACK)
                elif user_name is not None and (
                        session.query(SBDB_ORM.Account).filter(
                            or_(SBDB_ORM.Account.user_name == user_name,
                                SBDB_ORM.Account.email == user_name,
                                SBDB_ORM.Account.mobile_phone == user_name))
                        .first() is not None or len(user_name) < 2):
                    respond.SetErrorCode(BaseCommand.CS_USERNAME)
                elif email is not None and (
                        session.query(SBDB_ORM.Account).filter(
                            or_(SBDB_ORM.Account.user_name == email,
                                SBDB_ORM.Account.email == email,
                                SBDB_ORM.Account.mobile_phone == email))
                        .first() is not None or not Util.validateEmail(email)):
                    respond.SetErrorCode(BaseCommand.CS_EMAIL)
                elif mobile_phone is not None and (
                        session.query(SBDB_ORM.Account).filter(
                            or_(SBDB_ORM.Account.user_name == mobile_phone,
                                SBDB_ORM.Account.email == mobile_phone,
                                SBDB_ORM.Account.mobile_phone == mobile_phone))
                        .first() is not None
                        or not Util.validateMobilePhone(mobile_phone)):
                    respond.SetErrorCode(BaseCommand.CS_MOBILEPHONE)
                else:
                    try:
                        account = SBDB_ORM.Account()
                        account.language_id = 2
                        account.email = email
                        account.password = Util.hash_password(password)
                        account.user_name = user_name
                        account.mobile_phone = mobile_phone
                        account.version = 0
                        apartment = SBDB_ORM.Apartment()
                        apartment.arm_state = BaseCommand.PV_ARM_OFF
                        apartment.name = "Home"
                        apartment.scene_id = None
                        apartment.version = 0
                        account.apartments.append(apartment)
                        session.add(account)
                        session.commit()
                        respond.body[
                            BaseCommand.PN_VERSION] = apartment.version
                        respond.body[BaseCommand.PN_APARTMENTID] = apartment.id
                        respond.body[BaseCommand.PN_NAME] = apartment.name
                    except SQLAlchemyError as e:
                        respond.SetErrorCode(BaseCommand.CS_DBEXCEPTION)
                        logging.error("transport %d:%s",
                                      id(self.protocol.transport), e)
                        session.rollback()
            respond.Send()
