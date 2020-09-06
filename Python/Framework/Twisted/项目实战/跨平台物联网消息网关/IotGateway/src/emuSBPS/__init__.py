
import Command
from emuSBPS import ControlDevice
Command.dicInt_Type[ControlDevice.CControlDevice.command_id]=ControlDevice.CControlDevice

from emuSBPS import QueryDevice
Command.dicInt_Type[QueryDevice.CQueryDevice.command_id]=QueryDevice.CQueryDevice
Command.dicInt_Type[QueryDevice.CQueryDeviceResp.command_id]=QueryDevice.CQueryDeviceResp


