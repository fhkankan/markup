'''
Created on Aug 13, 2014

@author: Changlong
'''

from DB import SBDB, SBDB_ORM
from Utils import Util
import sys

def hash_password():
    session =SBDB.GetSession()
    
    accounts=session.query(SBDB_ORM.Account).with_lockmode('update').all()
    for account in accounts:
        account.password=Util.hash_password(account.password)
    
    session.commit()

if __name__ == '__main__':
    run_mode=None
    if len(sys.argv)>1: run_mode=sys.argv[1]
    
    if run_mode is not None:
        eval(run_mode+'()')
    
    
    
