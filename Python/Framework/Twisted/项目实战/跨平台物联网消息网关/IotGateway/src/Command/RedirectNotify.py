'''
Created on Sep 9, 2014

@author: Changlong
'''

from BaseNotify import CBaseNotify
import BaseCommand
from Utils import Config

class CRedirectNotify(CBaseNotify):
    '''
    classdocs
    '''

    command_id=0x00060007

    def __init__(self,data=None,protocol=None,client_id=0,addr=Config.domain_name):
        '''
        Constructor
        '''
        CBaseNotify.__init__(self, data, protocol,client_id)
        self.command_id=CRedirectNotify.command_id
        self.body[BaseCommand.PN_ADDR]=addr
        
    
