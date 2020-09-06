'''
Created on Jun 3, 2014

@author: Changlong
'''

from BaseCommand import CBaseCommand,CBaseRespCommand
import BaseCommand
import threading
from Utils import Config
from SBPS import InternalMessage
from twisted.internet import reactor

class CBaseSimpleControl(CBaseCommand):
    '''
    classdocs
    '''


    def __init__(self,data=None,protocol=None):
        '''
        Constructor
        '''
        CBaseCommand.__init__(self, data, protocol)
        
    def Run(self):
        if self.protocol.role==BaseCommand.PV_ROLE_INTERNAL:
            sb_protocol=None
            with self.protocol.factory.lockDict:
                sb_protocol=self.protocol.factory.dictRelayer.get(self.internalMessage.destId)
            
            if sb_protocol is None:
                self.Finish(False)
                return
                
            with sb_protocol.lock_dictWaitResp:
                sb_protocol.dictWaitResp[self.command_seq]=self
            self.Execute()
            return 
        
        with self.protocol.lockCmd:
            if self.protocol.role==BaseCommand.PV_ROLE_HUMAN:
                if not self.Authorized(): 
                    self.SendUnauthorizedResp()
                    return
                CBaseCommand.Run(self)
                
                #check whether contain parameter relayer_id
                relayer_id=self.body.get(BaseCommand.PN_RELAYERID,None)
                if relayer_id is None:
                    respond=self.GetResp()
                    respond.SetErrorCode(BaseCommand.CS_PARAMLACK)
                    respond.Send()
                    return
                
                with self.protocol.lock_dictWaitResp:
                    self.protocol.dictWaitResp[self.command_seq]=self
                
                interMessage=InternalMessage.CInternalMessage()
                interMessage.SetParam(InternalMessage.TTYPE_GATEWAY,relayer_id,0,InternalMessage.OPER_REQUEST,"",
                                  InternalMessage.TTYPE_HUMAN,self.protocol.client_id,id(self.protocol.transport))
                
                self.Send(interMessage)
                #threading.Timer(Config.timeout_relayer_control,self.timeout).start()
                reactor.callLater(Config.timeout_relayer_control,self.timeout)
    
    def Finish(self,status,resp_body=None):
        resp=self.GetResp()
        resp.command_status=status
        if resp_body:   resp.body=resp_body
        
        if self.protocol.role==BaseCommand.PV_ROLE_HUMAN:
            with self.protocol.lock_dictWaitResp:
                me=self.protocol.dictWaitResp.pop(self.command_seq,None)
                if me is None:  return  #this request has been feedbacked or timeout
                
            resp.Send()
            
        elif self.protocol.role==BaseCommand.PV_ROLE_INTERNAL:
            sb_protocol=None
            with self.protocol.factory.lockDict:
                sb_protocol=self.protocol.factory.dictRelayer.get(self.internalMessage.destId)
                
            if sb_protocol:
                with sb_protocol.lock_dictWaitResp:
                    me=sb_protocol.dictWaitResp.pop(self.command_seq,None)
                    if me is None:  return #this request has been feedbacked or timeout
                
            interMessage=InternalMessage.CInternalMessage()
            interMessage.SetParam(self.internalMessage.fromType,self.internalMessage.fromId,self.internalMessage.fromSock,InternalMessage.OPER_RESPONSE,"",
                              self.internalMessage.destType,self.internalMessage.destId,self.internalMessage.destSock)
            
            resp.Send(interMessage)
        
    def timeout(self):
        self.Finish(BaseCommand.CS_RELAYERRESPTIMEOUT)
    
    #--subclass override this member to realize it's special feature--
    def Execute(self):
        self.Finish(BaseCommand.CS_OK)
    #-----------------------------------------------------------------
       
       

class CBaseRespSimpleControl(CBaseRespCommand):
    '''
    classdocs
    '''


    def __init__(self,data=None,protocol=None):
        '''
        Constructor
        '''
        CBaseCommand.__init__(self, data, protocol)
        
    def Run(self):
        if self.protocol.role==BaseCommand.PV_ROLE_INTERNAL:
            account_protocol=self.protocol.factory.GetAccountProtocol(relayer_id=self.internalMessage.fromId,client_id=self.internalMessage.destId)
            if account_protocol is None:    return
            
            with account_protocol.lock_dictWaitResp:
                request=account_protocol.dictWaitResp.pop(self.command_seq,None)
                if request is None:  return  #this request has been feedbacked or timeout           
            self.Send()
        else:
            with self.protocol.lock_dictWaitResp:
                request=self.protocol.dictWaitResp.pop(self.command_seq,None)
                if request is None:  return #this request has been feedbacked or timeout
                
            interMessage=InternalMessage.CInternalMessage()
            interMessage.SetParam(request.internalMessage.fromType,request.internalMessage.fromId,request.internalMessage.fromSock,InternalMessage.OPER_RESPONSE,"",
                              request.internalMessage.destType,request.internalMessage.destId,request.internalMessage.destSock)
            self.Send(interMessage)
                
        
            
