'''
Created on 2013-9-3

@author: Changlong
'''

from BaseCommand import CBaseCommand
from sqlalchemy.exc import SQLAlchemyError
from DB import SBDB, SBDB_ORM
from Command import BaseCommand
import logging


class CQueryApartment(CBaseCommand):
    '''
    classdocs 
    '''
    command_id = 0x00010002

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
                version = self.body[BaseCommand.PN_VERSION]
                apartment_id = self.body[BaseCommand.PN_APARTMENTID]
                respond = self.GetResp()
                try:
                    apartment = session.query(SBDB_ORM.Apartment).filter(
                        SBDB_ORM.Apartment.id == apartment_id).one()
                    respond.body[BaseCommand.PN_VERSION] = apartment.version
                    respond.body[BaseCommand.PN_SCENE_ID] = apartment.scene_id
                    respond.body[BaseCommand.PN_ARMSTATE] = apartment.arm_state
                    if version != apartment.version:
                        apartment_info = {}
                        apartment_info[BaseCommand.PN_ID] = apartment.id
                        apartment_info[BaseCommand.PN_NAME] = apartment.name
                        apartment_info[
                            BaseCommand.PN_VERSION] = apartment.version

                        bDeviceInserted = False
                        listDevice = []
                        for apartment_device in apartment.apartment_devices:
                            elementDevice = {}
                            elementDevice[
                                BaseCommand.PN_ID] = apartment_device.device.id
                            elementDevice[
                                BaseCommand.
                                PN_DEVTYPE] = apartment_device.device.device_model.device_type.name
                            elementDevice[
                                BaseCommand.
                                PN_DEVMODEL] = apartment_device.device.device_model.name
                            elementDevice[
                                BaseCommand.
                                PN_DEVCODE] = apartment_device.device.uni_code
                            elementDevice[
                                BaseCommand.PN_DEVNAME] = apartment_device.name
                            listDeviceKey = []
                            listDevice.append(elementDevice)

                        listRelayer = []
                        for apartment_relayer in apartment.apartment_relayers:
                            relayer = apartment_relayer.relayer
                            elementRelayer = {}
                            elementRelayer[BaseCommand.PN_ID] = relayer.id
                            elementRelayer[
                                BaseCommand.PN_SB_CODE] = relayer.uni_code
                            if not bDeviceInserted:
                                elementRelayer[
                                    BaseCommand.PN_DEVICES] = listDevice
                                bDeviceInserted = True
                            listRelayer.append(elementRelayer)
                        apartment_info[BaseCommand.PN_RELAYERS] = listRelayer

                        respond.body[
                            BaseCommand.PN_APARTMENTINFO] = apartment_info

                except SQLAlchemyError as e:
                    respond.SetErrorCode(BaseCommand.CS_DBEXCEPTION)
                    logging.error("transport %d:%s",
                                  id(self.protocol.transport), e)
                    session.rollback()
            respond.Send()
