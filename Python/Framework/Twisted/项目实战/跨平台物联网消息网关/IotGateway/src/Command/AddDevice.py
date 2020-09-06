'''
Created on 2013-8-12

@author: Changlong
'''
from BaseCommand import CBaseCommand
from sqlalchemy.exc import SQLAlchemyError
from DB import SBDB, SBDB_ORM
from sqlalchemy import and_
from Command import BaseCommand
import logging
import string


class CAddDevice(CBaseCommand):
    '''
    classdocs 
    '''
    command_id = 0x00020001

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

            apartment_id = self.body[BaseCommand.PN_APARTMENTID]
            relayer_id = self.body.get(BaseCommand.PN_RELAYERID)
            dev_model = self.body[BaseCommand.PN_DEVMODEL]
            dev_code = self.body[BaseCommand.PN_DEVCODE]
            dev_keys = self.body.get(BaseCommand.PN_DEVKEYS)
            respond = self.GetResp()
            with SBDB.session_scope() as session:
                model = SBDB.GetDeviceModelByName(session, dev_model)
                try:
                    if relayer_id is None:
                        relayer_id, = session.query(SBDB_ORM.Relayer.id).join(
                            SBDB_ORM.Apartment_Relayer).filter(
                                SBDB_ORM.Apartment_Relayer.apartment_id ==
                                apartment_id).order_by(
                                    SBDB_ORM.Relayer.id).first()
                    else:
                        relayer_id = int(relayer_id)
                    if model is None:
                        respond.SetErrorCode(BaseCommand.CS_DEVICEMODEL)
                    elif relayer_id is None:
                        respond.SetErrorCode(BaseCommand.CS_NORELAYER)
                    elif session.query(SBDB_ORM.ApartmentDevice).join(
                            SBDB_ORM.Device).filter(
                                and_(
                                    SBDB_ORM.Device.uni_code == dev_code,
                                    SBDB_ORM.ApartmentDevice.apartment_id ==
                                    apartment_id)).first() is not None:
                        respond.SetErrorCode(BaseCommand.CS_DEVICEEXIST)
                    else:
                        apartment = SBDB.IncreaseVersions(
                            session, 0, apartment_id)
                        device = SBDB.GetDeviceForcely(session, dev_code,
                                                       dev_model)
                        if device is None:
                            device = SBDB_ORM.Device()
                            device.device_model_id = model.id
                            device.uni_code = dev_code

                            session.add(device)

                        else:
                            apartment_device = SBDB_ORM.ApartmentDevice()
                            apartment_device.apartment_id = apartment_id
                            apartment_device.name = model.device_type.name
                            apartment_device.relayer_id = relayer_id
                            apartment_device.device_id = device.id

                            respond.body[
                                BaseCommand.PN_VERSION] = apartment.version
                            session.add(apartment_device)
                            session.commit()
                            respond.body[BaseCommand.PN_DEVICEID] = device.id
                except SQLAlchemyError as e:
                    respond.SetErrorCode(BaseCommand.CS_DBEXCEPTION)
                    logging.error("transport %d:%s",
                                  id(self.protocol.transport), e)
                    session.rollback()
            respond.Send()
