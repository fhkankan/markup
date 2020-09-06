'''
Created on 2013-8-27

@author: Changlong
'''
from Command.BaseCommand import CBaseCommand
class CControlDevice(CBaseCommand):
    '''
    classdocs
    '''

    command_id=0x00060001
    
    def __init__(self,data=None,protocol=None):
        '''
        Constructor
        '''
        CBaseCommand.__init__(self, data, protocol)
        
    def Run(self):
        #if not self.Authorized(): return
        CBaseCommand.Run(self)
        self.SendResp()
