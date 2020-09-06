'''
Created on 2013-9-18

@author: Changlong
'''

from Command.BaseCommand import CBaseCommand

class CQueryDevice(CBaseCommand):
    '''
    classdocs
    '''

    command_id=0x00010001
    
    def __init__(self,data=None,protocol=None):
        '''
        Constructor
        '''
        CBaseCommand.__init__(self, data, protocol)
        
    def Run(self):
        #if not self.Authorized(): return
        CBaseCommand.Run(self)
        self.SendResp()
        
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
        pass
        
