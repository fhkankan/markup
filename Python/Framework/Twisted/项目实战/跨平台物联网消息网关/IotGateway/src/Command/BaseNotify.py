'''
Created on Jun 3, 2014

@author: Changlong
'''

from BaseCommand import CBaseCommand


class CBaseNotify(CBaseCommand):
    '''
    classdocs
    '''

    def __init__(self, data=None, protocol=None, client_id=0):
        '''
        Constructor
        '''
        CBaseCommand.__init__(self, data, protocol)
        self.client_id = client_id

    def Notify(self, internalMessage=None):

        if internalMessage:
            print("notify :", internalMessage)
            self.Send(internalMessage)
            return
        if self.relayer_id == 0:
            return
        with self.protocol.factory.lockDict:
            dictAccount = self.protocol.factory.dictAccounts
            if self.relayer_id in dictAccount:
                self.command_seq = self.GetNextSeq()
                for clientProtocol in dictAccount[self.relayer_id]:
                    # if clientProtocol.rcv_alarm=="True" and clientProtocol.client_id==self.client_id:
                    if clientProtocol.client_id == self.client_id:
                        self.protocol = clientProtocol
                        self.Send()
                        break
