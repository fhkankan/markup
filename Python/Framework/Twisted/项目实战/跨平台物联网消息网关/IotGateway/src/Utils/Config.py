'''
Created on 2013-8-28

@author: Changlong
'''

import configparser
import os
cf = configparser.ConfigParser()

listConfigs = cf.read(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.path.pardir,
        "sbs.conf"))
if len(listConfigs) > 0:
    timeout_relayer_control = cf.getint("default", "timeout_relayer_control")
    time_heartbeat = cf.getint("default", "time_heartbeat")
    interval_patroller = cf.getint("default", "interval_patroller")
    timeout_buffered_state = cf.getint("default", "timeout_buffered_state")
    maxcmd_relayer_control = cf.getint("default", "maxcmd_relayer_control")
    db_connection_string = cf.get("default", "db_connection_string")
    count_connection = cf.getint("default", "count_connection")

second_restore_require = 15 * 60
dir_local_root = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.path.pardir, os.path.pardir)
dir_local_mainserver = dir_local_root + '\\src'
dir_local_webserver = dir_local_root + '\\webapp'
dir_local_static = dir_local_webserver + '\\static'
dir_local_templates = dir_local_webserver + '\\templates'
domain_name = "www.honhome.com"
