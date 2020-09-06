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


class CRemoveDevice(CBaseCommand):
    '''
    classdocs 
    '''
    command_id = 0x00030001

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
                dev_id = self.body[BaseCommand.PN_DEVICEID]
                apartment_id = self.body[BaseCommand.PN_APARTMENTID]
                respond = self.GetResp()
                if dev_id is None:
                    respond.SetErrorCode(BaseCommand.CS_PARAMLACK)
                else:
                    try:
                        apartment = SBDB.IncreaseVersions(
                            session, 0, apartment_id)
                        # session.query(SBDB_ORM.ApartmentDevice).join(SBDB_ORM.Apartment).join(SBDB_ORM.Device).filter(and_(SBDB_ORM.Apartment.id==apartment_id,SBDB_ORM.Device.id==dev_id)).delete()
                        session.query(SBDB_ORM.ApartmentDevice).filter(
                            and_(
                                SBDB_ORM.ApartmentDevice.apartment_id ==
                                apartment_id,
                                SBDB_ORM.ApartmentDevice.device_id ==
                                dev_id)).delete()
                        respond.body[
                            BaseCommand.PN_VERSION] = apartment.version
                        session.commit()

                    except SQLAlchemyError as e:
                        respond.SetErrorCode(BaseCommand.CS_DBEXCEPTION)
                        logging.error("transport %d:%s",
                                      id(self.protocol.transport), e)
                        session.rollback()
            respond.Send()
