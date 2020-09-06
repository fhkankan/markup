'''
Created on 2013-8-15

@author: Changlong
'''

import logging
import threading
import time
import emuSBPS.emuReactor as emuReactor
import emuSBPS
from emuSBPS import *
from emuSBPS import ControlDevice
from Command import Authorize, BaseCommand, HeartBeat  # ,EventDev
logging.basicConfig(
    filename='example_relayer.log',
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s")


def main_loop():
    time.sleep(1)
    if emuReactor.protocolActive is None:
        print("can't connect server")
        return
    emuReactor.protocolActive.role = BaseCommand.PV_ROLE_RELAYER

    counting = 0
    while True:
        request = None
        if counting <= 0:
            command = "Authorize"
        else:
            time.sleep(60)
            command = "HeartBeat"
            # command = input('Enter Command: ')
        counting += 1
        if command == "quit":
            break
        try:
            request = eval(command + ".C" + command + "()")
            request.protocol = emuReactor.protocolActive
        except Exception as e:
            print("unknown command :", command, e)
            continue

        if isinstance(request, Authorize.CAuthorize):
            request.body[BaseCommand.PN_SB_CODE] = "test_relayer"
            request.body[
                BaseCommand.PN_TERMINALTYPE] = BaseCommand.PV_ROLE_RELAYER

        emuReactor.SendAndVerify(request)


def traced_main_loop():
    import traceback
    try:
        main_loop()
    except:
        print(traceback.format_exc())


def main():
    # thread.start_new_thread(emuReactor.Run)
    t = threading.Thread(target=traced_main_loop, daemon=True)
    t.start()
    emuReactor.Run()


if __name__ == '__main__':
    main()
