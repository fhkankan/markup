import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

dicInt_Type = {}
import AddAccount
dicInt_Type[AddAccount.CAddAccount.command_id] = AddAccount.CAddAccount

import AddApartment
dicInt_Type[AddApartment.CAddApartment.command_id] = AddApartment.CAddApartment

import AddDevice
dicInt_Type[AddDevice.CAddDevice.command_id] = AddDevice.CAddDevice

import AddRelayer
dicInt_Type[AddRelayer.CAddRelayer.command_id] = AddRelayer.CAddRelayer

import Authorize
dicInt_Type[Authorize.CAuthorize.command_id] = Authorize.CAuthorize

import BaseCommand
dicInt_Type[BaseCommand.CBaseCommand.command_id] = BaseCommand.CBaseCommand

import HeartBeat
dicInt_Type[HeartBeat.CHeartBeat.command_id] = HeartBeat.CHeartBeat

import ModifyApartment
dicInt_Type[ModifyApartment.CModifyApartment.
            command_id] = ModifyApartment.CModifyApartment

import ModifyRelayer
dicInt_Type[
    ModifyRelayer.CModifyRelayer.command_id] = ModifyRelayer.CModifyRelayer

import RemoveApartment
dicInt_Type[RemoveApartment.CRemoveApartment.
            command_id] = RemoveApartment.CRemoveApartment

import RemoveDevice
dicInt_Type[RemoveDevice.CRemoveDevice.command_id] = RemoveDevice.CRemoveDevice

import RemoveRelayer
dicInt_Type[
    RemoveRelayer.CRemoveRelayer.command_id] = RemoveRelayer.CRemoveRelayer

import ControlDevice
dicInt_Type[
    ControlDevice.CControlDevice.command_id] = ControlDevice.CControlDevice
dicInt_Type[ControlDevice.CControlDeviceResp.
            command_id] = ControlDevice.CControlDeviceResp

import QueryDevice
dicInt_Type[QueryDevice.CQueryDevice.command_id] = QueryDevice.CQueryDevice
dicInt_Type[
    QueryDevice.CQueryDeviceResp.command_id] = QueryDevice.CQueryDeviceResp

import QueryAccount
dicInt_Type[QueryAccount.CQueryAccount.command_id] = QueryAccount.CQueryAccount

import QueryApartment
dicInt_Type[
    QueryApartment.CQueryApartment.command_id] = QueryApartment.CQueryApartment

import SetArm
dicInt_Type[SetArm.CSetArm.command_id] = SetArm.CSetArm

import SetProfile
dicInt_Type[SetProfile.CSetProfile.command_id] = SetProfile.CSetProfile
