'''
Created on 2013-8-12

@author: Changlong
'''
from BaseCommand import CBaseCommand
from twisted.internet import threads
import BaseCommand
from DB import SBDB
class CHeartBeat(CBaseCommand):
    '''
    classdocs 
    '''
    command_id=0x00000002
    def __init__(self,data=None,protocol=None):
        '''
        Constructor
        '''
        CBaseCommand.__init__(self, data, protocol)
    
    def Run(self):
        with self.protocol.lockCmd:
            if self.Authorized():
                CBaseCommand.Run(self)
                self.SendResp()
                if self.protocol.role==BaseCommand.PV_ROLE_HUMAN:
                    threads.deferToThread(SBDB.UpdateActiveTime,self.protocol.role,self.protocol.client_id,id(self.protocol.transport))
                elif self.protocol.role==BaseCommand.PV_ROLE_RELAYER:
                    threads.deferToThread(SBDB.UpdateActiveTime,self.protocol.role,self.protocol.relayer_id,id(self.protocol.transport))
            else:
                self.SendUnauthorizedResp()

    

