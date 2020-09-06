'''
Created on 2013-8-12

@author: Changlong
'''
from BaseCommand import CBaseCommand
from sqlalchemy.exc import SQLAlchemyError
from DB import SBDB, SBDB_ORM
from Command import BaseCommand
from sqlalchemy import and_
import logging


class CModifyRelayer(CBaseCommand):
    '''
    classdocs 
    '''
    command_id = 0x00040007

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
                apartment_id = self.body[BaseCommand.PN_APARTMENTID]
                sb_id = self.body[BaseCommand.PN_RELAYERID]
                sb_code = self.body.get(BaseCommand.PN_SB_CODE, None)
                name = self.body.get(BaseCommand.PN_NAME, None)
                respond = self.GetResp()
                if sb_id is None:
                    respond.SetErrorCode(BaseCommand.CS_PARAMLACK)
                else:
                    try:
                        apartment = None
                        if sb_code is None:
                            apartment = SBDB.IncreaseVersion(
                                session, apartment_id)
                        else:
                            apartment = SBDB.IncreaseVersions(
                                session, 0, apartment_id)
                        relayer, apartment_relayer = session.query(
                            SBDB_ORM.Relayer, SBDB_ORM.Apartment_Relayer).join(
                                SBDB_ORM.Apartment_Relayer).filter(
                                    and_(
                                        SBDB_ORM.Relayer.id == sb_id,
                                        SBDB_ORM.Apartment_Relayer.apartment_id
                                        == apartment_id)).first()
                        if name is not None:
                            apartment_relayer.name = name
                        if sb_code is not None:
                            relayer.uni_code = sb_code
                        respond.body[
                            BaseCommand.PN_VERSION] = apartment.version
                        session.commit()
                    except SQLAlchemyError as e:
                        respond.SetErrorCode(BaseCommand.CS_DBEXCEPTION)
                        logging.error("transport %d:%s",
                                      id(self.protocol.transport), e)
                        session.rollback()
            respond.Send()
