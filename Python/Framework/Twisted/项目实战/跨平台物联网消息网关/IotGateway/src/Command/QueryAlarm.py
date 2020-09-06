'''
Created on 2013-9-3

@author: Changlong
'''

from BaseCommand import CBaseCommand
from sqlalchemy.exc import SQLAlchemyError
from DB import SBDB, SBDB_ORM
from sqlalchemy import and_
from Command import BaseCommand
import logging


class CQueryAlarm(CBaseCommand):
    '''
    classdocs 
    '''
    command_id = 0x00010007
    PAGELIMIT = 10

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
                start_dt = self.body.get(BaseCommand.PN_STARTDT,
                                         "1999-01-01 00:00:00")
                end_dt = self.body.get(BaseCommand.PN_ENDDT,
                                       "2059-01-01 00:00:00")
                alarm_types = self.body.get(BaseCommand.PN_ALARMTYPES, None)
                device_ids = self.body.get(BaseCommand.PN_DEVICEIDS, None)
                apartment_ids = self.body.get(BaseCommand.PN_APARTMENTIDS,
                                              None)
                page = self.body[BaseCommand.PN_PAGE]

                respond = self.GetResp()
                try:
                    theQuery = session.query(SBDB_ORM.Alarm, SBDB_ORM.ApartmentDeviceKey.name).join(SBDB_ORM.Event).\
                        join(SBDB_ORM.DeviceKeyCode, SBDB_ORM.DeviceKeyCode.id == SBDB_ORM.Event.device_key_code_id).\
                        outerjoin(SBDB_ORM.ApartmentDeviceKey, and_(SBDB_ORM.ApartmentDeviceKey.device_key_code_id == SBDB_ORM.DeviceKeyCode.id, SBDB_ORM.ApartmentDeviceKey.apartment_device_id == SBDB_ORM.Alarm.apartment_device_id)).\
                        join(SBDB_ORM.ApartmentDevice, SBDB_ORM.ApartmentDeviceKey.apartment_device_id == SBDB_ORM.ApartmentDevice.id).\
                        join(SBDB_ORM.Apartment, SBDB_ORM.Apartment.id == SBDB_ORM.ApartmentDevice.apartment_id).filter(and_(
                            SBDB_ORM.Event.dt.between(start_dt, end_dt), SBDB_ORM.Apartment.account_id == self.protocol.account_id))
                    if apartment_ids:
                        theQuery = theQuery.filter(
                            SBDB_ORM.ApartmentDevice.apartment_id.in_(
                                apartment_ids))
                    if device_ids:
                        theQuery = theQuery.filter(
                            SBDB_ORM.DeviceKeyCode.device_id.in_(device_ids))
                    if alarm_types:
                        theQuery = theQuery.join(SBDB_ORM.Device).join(
                            SBDB_ORM.DeviceModel).join(
                                SBDB_ORM.DeviceType).filter(
                                    SBDB_ORM.DeviceType.name.in_(alarm_types))

                    theQuery = theQuery.order_by(
                        SBDB_ORM.Alarm.id.desc()).offset(
                            CQueryAlarm.PAGELIMIT * page).limit(
                                CQueryAlarm.PAGELIMIT)
                    listAlarms = []
                    for row in theQuery:
                        alarm = row.Alarm
                        dev_key_name = row.name
                        if dev_key_name is None:
                            dev_key_name = "N/A"
                        alarm_item = {}
                        alarm_item[BaseCommand.PN_ID] = alarm.id
                        alarm_item[
                            BaseCommand.
                            PN_APARTMENTID] = alarm.apartment_device.apartment_id
                        alarm_item[
                            BaseCommand.
                            PN_DEVICETYPENAME] = alarm.event.device_key_code.device.device_model.device_type.name
                        alarm_item[
                            BaseCommand.
                            PN_DEVICEID] = alarm.event.device_key_code.device_id
                        alarm_item[
                            BaseCommand.PN_DT] = alarm.event.dt.strftime(
                                "%Y-%m-%d %H:%M:%S")
                        alarm_item[
                            BaseCommand.
                            PN_DEVCODE] = alarm.event.device_key_code.device.uni_code
                        alarm_item[BaseCommand.PN_DEVNAME] = dev_key_name
                        alarm_item[
                            BaseCommand.
                            PN_DEVMODEL] = alarm.event.device_key_code.device.device_model.name
                        listAlarms.append(alarm_item)
                    respond.body[BaseCommand.PN_ALARMS] = listAlarms

                except SQLAlchemyError as e:
                    respond.SetErrorCode(BaseCommand.CS_DBEXCEPTION)
                    logging.error("transport %d:%s",
                                  id(self.protocol.transport), e)
                    session.rollback()
            respond.Send()
