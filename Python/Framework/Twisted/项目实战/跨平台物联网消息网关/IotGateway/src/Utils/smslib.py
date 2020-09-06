#!/usr/bin/python
# smslib.py

import urllib
import time
import datetime
from DB import SBDB, SBDB_ORM


def SendAlarmBySMSbyMdao(SMSmessage, listPhone):
    """send alram message to user by SMS

    SMSmessage: message of SMS
    listPhone: phone number list which used to receive alarm message

    return 
        True if sms is sent successfully. Or it will return False. 
    """
    # organization ID
    entid = 'entid=8000805&'
    # user name
    uid = 'uid=admin&'
    # password
    pwd = 'pwd=130821&'
    urllink = 'http://qmsg2.mdao.com/sendSms.do?'
    urllink += entid + uid + pwd + 'mobs='

    for phonenumber in listPhone:
        urllink += phonenumber + ','

    urllink += '&msg=' + SMSmessage + '&returnflag=text'
    print("Will send sms by:", urllink)

    # send SMS by http get method
    fp = urllib.urlopen(urllink)
    # get response from SMS server
    urlresponse = fp.readlines()

    # check the respnose to see whether SMS has been sent out
    if urlresponse[1][0] == '0':
        fp.close()
        return True, 0
    else:
        fp.close()
        return False, int(urlresponse[1][0])


def SendAlarmBySMS(sms_template, apartment_name, alarm_type, listPhone,
                   device_name):
    """send alram message to user by SMS

    sms_template: detailed alarm message
    apartment_name: name of apartment
    alarm_type: type of alarm
    listPhone: phone number list which used to receive alarm message

    return 
        True if sms is sent successfully. Or it will return False. 

    note: The maximum of phone number is 50. 

    """
    ISOTIMEFORMAT = '%Y-%m-%d %X'

    #alarmmsg = time.strftime(ISOTIMEFORMAT, time.localtime())
    #alarmmsg += ' [' + apartment_name + '] ' + alarm_type +' ' + sms_template
    #alarmmsg = urllib.quote(alarmmsg)
    # return SendAlarmBySMSbyMdao(alarmmsg, listPhone)
    strTime = time.strftime(ISOTIMEFORMAT, time.localtime())
    sms_template = sms_template.replace("[apartment]", apartment_name)
    sms_template = sms_template.replace("[time]", strTime)
    sms_template = sms_template.replace("[type]", alarm_type)
    sms_template = sms_template.replace("[device]", device_name)
    sms_template = urllib.quote(sms_template.encode('gbk'))

    return (sms_template, ) + SendAlarmBySMSbyMdao(sms_template, listPhone)


import threading
mutex = threading.Lock()


def SendAndSave(sms_template, apartment, alarm_type, listPhone, device_name):

    global mutex
    with mutex:

        # message,result,error=SendAlarmBySMS(sms_template,apartment.name,alarm_type,listPhone,device_name)
        message, result, error = SendAlarmBySMS(
            sms_template, apartment.account.user_name, alarm_type, listPhone,
            device_name)

        # insert into database
        sms_head = SBDB_ORM.SmsSenderHead()
        sms_head.apartment_id = apartment.id
        sms_head.content = message
        sms_head.dt = datetime.datetime.now()
        for phone in listPhone:
            sms_list = SBDB_ORM.SmsSenderList()
            sms_list.mobile_phone = phone
            sms_list.result = error
            sms_head.sms_sender_lists.append(sms_list)

        with SBDB.session_scope() as session:
            session.add(sms_head)
            session.commit()
        return result


if __name__ == '__main__':
    result = SendAlarmBySMS('Please call 119', '101-102', 'fire alarm',
                            ['18901688801', '18930519689'])
    if result == True:
        print('SMS was sent successfully')
    else:
        print('SMS can not be sent')
