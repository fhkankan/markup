'''
Created on 2013-8-29

@author: Changlong
'''
'''
Created on 2013-8-26

@author: Changlong
'''

from BaseControl import CBaseControl,CDeviceCmd
from Command import BaseCommand
from Command.BaseCommand import CBaseCommand
import logging

class CQueryDevice(CBaseControl):
    '''
    classdocs
    '''

    command_id=0x00010001
    
    def __init__(self,data=None,protocol=None):
        '''
        Constructor
        '''
        CBaseControl.__init__(self, data, protocol)
    
    #----------------subclass override--------------------
    def initDictRelayerControls(self):
        dev_model=self.body[BaseCommand.PN_DEVMODEL]
        dev_code=self.body[BaseCommand.PN_DEVCODE]
        dev_key_seq=self.body[BaseCommand.PN_DEVSEQ]
        self.initByDeviceCmdList([CDeviceCmd(dev_model,dev_code, dev_key_seq, None),])
        
    def getCommand(self,deviceCmd):
        control_device=CQueryDevice()
        control_device.body[BaseCommand.PN_DEVMODEL]=deviceCmd.dev_model
        control_device.body[BaseCommand.PN_DEVCODE]=deviceCmd.dev_code
        control_device.body[BaseCommand.PN_DEVSEQ]=deviceCmd.dev_seq
        return control_device
    
    #-----------------------------------------------------
      


    
class CQueryDeviceResp(CBaseCommand):
    '''
    classdocs
    '''

    command_id=0x80010001
    
    def __init__(self,data=None,protocol=None):
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
            with self.protocol.cond_dictControlling:
                request=self.protocol.dictControlling.pop(self.command_seq,None)
                if request is not None: 
                    logging.debug("call self.protocol.cond_dictControlling.notify() due to CQueryDeviceResp in protocol %d",id(self.protocol.transport))
                    if request.timer!=None:    
                        request.timer.cancel()
                        request.timer=None
                    self.protocol.cond_dictControlling.notify()
                    
                else:
                    logging.debug("fail to self.protocol.dictControlling.pop(%d) due to CQueryDeviceResp in protocol %d",self.command_seq,id(self.protocol.transport))
                #if request is not None: self.protocol.cond_dictControlling.notify()
            relayer_id=None
            if self.protocol.role==BaseCommand.PV_ROLE_INTERNAL:
                with self.protocol.lock_dictWaitResp:
                    #request=self.protocol.dictWaitResp.pop((self.internalMessage.fromId<<32)+self.command_seq,None)
                    request=self.protocol.dictWaitResp.pop((self.internalMessage.destSock<<32)+self.command_seq,None)
                relayer_id=self.internalMessage.fromId
            else:
                with self.protocol.lock_dictWaitResp:
                    request=self.protocol.dictWaitResp.pop(self.command_seq,None)
                relayer_id=self.protocol.relayer_id
            if request is None: return
            requireCommand=request.requireCommand
            requireCommand.FinishOne(relayer_id,request,self)
            request.requireCommand=None
        
