# -*- coding: utf-8 -*-
'''
Created on 2013-9-9

@author: Changlong
'''

from DB import SBDB, SBDB_ORM
from sqlalchemy import and_


def GetTemplate_old(apartment, event):
    default = temp = GetDefaultTemplate()
    for message_template in event.device_key.device_model.MessageTemplates:
        temp = ChooseTemplate(temp, message_template, apartment, event)
    if default != temp:
        return temp

    for message_template in apartment.account.MessageTemplates:
        temp = ChooseTemplate(temp, message_template, apartment, event)
    if default != temp:
        return temp

    for message_template in apartment.account.language.MessageTemplates:
        temp = ChooseTemplate(temp, message_template, apartment, event)
    if default != temp:
        return temp

    return default


def GetTemplate(session, apartment, event):
    print("1111111111...............................")
    default = temp = GetDefaultTemplate()
    if apartment not in session:
        apartment = session.query(SBDB_ORM.Apartment).get(apartment.id)
    print("222222222..............................")
    for message_template in apartment.account.language.MessageTemplates:
        temp = ChooseTemplateByLanguage(temp, message_template, apartment,
                                        event)
    print("3333333333..............................")
    if default != temp:
        return temp
    print("444444444444..............................")
    for message_template in apartment.account.MessageTemplates:
        temp = ChooseTemplateByAccount(temp, message_template, apartment,
                                       event)
    print("5555555555555..............................")
    if default != temp:
        return temp

    print("66666666666666.............................")
    for message_template in event.device_key.device_model.MessageTemplates:
        temp = ChooseTemplateByModel(temp, message_template, apartment, event)
    print("2677777777777..............................")
    if default != temp:
        return temp

    return default


def GetDefaultTemplate():
    with SBDB.session_scope() as session:
        defaultTemplate = session.query(SBDB_ORM.MessageTemplate).filter(
            and_(SBDB_ORM.MessageTemplate.language_id is None,
                 SBDB_ORM.MessageTemplate.account_id is None,
                 SBDB_ORM.MessageTemplate.sensor_model_id is None)).first()
        if defaultTemplate is None:
            defaultTemplate = SBDB_ORM.MessageTemplate()
            defaultTemplate.template = "[apartment]的[device]于[time]发生[type]告警"
            session.add(defaultTemplate)
            session.commit()
    return defaultTemplate


def ChooseTemplate(old, new, apartment, event):
    # if a template match the device_model of the event, choose template by device_model and return
    if old.sensor_model_id == event.device_key.device_model_id:
        if new.sensor_model_id != event.device_key.device_model_id:
            return old
    else:
        if new.sensor_model_id == event.device_key.device_model_id:
            return new

    # now the device_model are in the same compare level, so compare account
    # if a template match the account of the event, choose template by account and return
    if old.account_id == apartment.account_id:
        if new.account_id != apartment.account_id:
            return old
    else:
        if new.account_id == apartment.account_id:
            return new

    # now the device_model and account are in the same compare level, so compare language
    # if a template match the account's language of the event, choose template by language and return
    if old.language_id == apartment.account.language_id:
        if new.language_id != apartment.account.language_id:
            return old
    else:
        if new.language_id == apartment.account.language_id:
            return new

    return old


def ChooseTemplateByModel(old, new, apartment, event):
    if old.sensor_model_id == event.device_key.device_model_id:
        if new.sensor_model_id != event.device_key.device_model_id:
            return old
    else:
        if new.sensor_model_id == event.device_key.device_model_id:
            return new


def ChooseTemplateByAccount(old, new, apartment, event):
    if old.account_id == apartment.account_id:
        if new.account_id != apartment.account_id:
            return old
    else:
        if new.account_id == apartment.account_id:
            return new


def ChooseTemplateByLanguage(old, new, apartment, event):
    if old.language_id == apartment.account.language_id:
        if new.language_id != apartment.account.language_id:
            return old
    else:
        if new.language_id == apartment.account.language_id:
            return new
