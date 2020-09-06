'''
Created on Jun 3, 2014

@author: Changlong
'''

from BaseSimpleControl import CBaseSimpleControl,CBaseRespSimpleControl


class CQueryRelayerStatus(CBaseSimpleControl):
    '''
    classdocs
    '''

    command_id=0x00010008
    def __init__(self,data=None,protocol=None):
        '''
        Constructor
        '''
        CBaseSimpleControl.__init__(self, data, protocol)
        
        
        
    
class CQueryRelayerStatusResp(CBaseRespSimpleControl):
    '''
    classdocs
    '''

    command_id=0x80010008
    
    def __init__(self,data=None,protocol=None):
        '''
        Constructor
        '''
        CBaseRespSimpleControl.__init__(self, data, protocol)
        
CQueryRelayerStatus.TypeResp=CQueryRelayerStatusResp
