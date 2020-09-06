/*==============================================================*/
/* DBMS name:      PostgreSQL 8                                 */
/* Created on:     2013/8/21 15:44:24                           */
/*==============================================================*/






/*==============================================================*/
/* Table: Apartment_Superbox                                    */
/*==============================================================*/
create table Apartment_Superbox (
   apartment_id         INT4                 not null,
   superbox_id          INT4                 not null,
   name                 VARCHAR(50)          null,
   constraint PK_APARTMENT_SUPERBOX primary key (apartment_id, superbox_id)
);

/*==============================================================*/
/* Table: account                                               */
/*==============================================================*/
create table account (
   id                   SERIAL not null,
   language_id          INT4                 not null,
   user_name            VARCHAR(20)          null,
   password             VARCHAR(50)          null,
   email                VARCHAR(255)         null,
   mobile_phone         VARCHAR(50)          null,
   version              INT4                 null,
   constraint PK_ACCOUNT primary key (id)
);

comment on table account is
'email: 用于取回帐户密码
version: 本帐号的配置版本（用于与客户端同步信息）';

/*==============================================================*/
/* Table: alarm_name                                            */
/*==============================================================*/
create table alarm_name (
   language_id          INT4                 not null,
   device_type_id       INT4                 not null,
   name                 VARCHAR(50)          null,
   constraint PK_ALARM_NAME primary key (language_id, device_type_id)
);

comment on table alarm_name is
'用于按“语言-设备类型”对保存告警名称';

/*==============================================================*/
/* Table: apartment                                             */
/*==============================================================*/
create table apartment (
   id                   SERIAL not null,
   account_id           INT4                 not null,
   name                 VARCHAR(50)          null,
   arm_state            INT4                 null,
   scene_id             INT4                 null,
   dt_arm               TIMESTAMP            null,
   version              INT4                 null,
   constraint PK_APARTMENT primary key (id)
);

comment on table apartment is
'arm_state:
1. disarmed,不布防
2. armed,布防


scene_id: 当前所处场景的id
';

/*==============================================================*/
/* Index: apartment_PK                                          */
/*==============================================================*/
create unique index apartment_PK on apartment (
id
);



/*==============================================================*/
/* Table: client                                                */
/*==============================================================*/
CREATE TABLE client
(
  id serial NOT NULL,
  account_id integer NOT NULL,
  device_token character varying(100),
  enable_alarm boolean,
  os character varying(50),
  CONSTRAINT pk_client PRIMARY KEY (id),
  CONSTRAINT fk_client_reference_account FOREIGN KEY (account_id)
      REFERENCES account (id) MATCH SIMPLE
      ON UPDATE CASCADE ON DELETE CASCADE,
  CONSTRAINT ak_client_ur_client UNIQUE (account_id, device_token)
)
WITH (OIDS=FALSE);
ALTER TABLE client OWNER TO postgres;
COMMENT ON TABLE client IS '终端信息表：
device_token：设备码
os：操作系统，取值android/ios';


/*==============================================================*/
/* Table: contactor                                             */
/*==============================================================*/
create table contactor (
   id                   SERIAL not null,
   apartment_id         INT4                 not null,
   name                 VARCHAR(50)          null,
   mobile_phone         VARCHAR(50)          null,
   email_addr           VARCHAR(200)         null,
   constraint PK_CONTACTOR primary key (id)
);

/*==============================================================*/
/* Index: contactor_PK                                          */
/*==============================================================*/
create unique index contactor_PK on contactor (
id
);

/*==============================================================*/
/* Table: device                                                */
/*==============================================================*/
create table device (
   id                   SERIAL not null,
   device_model_id      INT4                 not null,
   superbox_id          INT4                 not null,
   uni_code             VARCHAR(50)          null,
   name                 VARCHAR(50)          null,
   flag_notification    VARCHAR(32)          null,
   constraint PK_DEVICE primary key (id)
);

comment on table device is
'unicode: 设备二维码
name:设备人性化名称
flag_notification用于按字节标示
是否在变化时发送短信/邮件/IM..
低位开始：
第一位：是否不发送短信
第二位：是否不发送邮件
';

/*==============================================================*/
/* Index: device_PK                                             */
/*==============================================================*/
create unique index device_PK on device (
id
);

/*==============================================================*/
/* Table: device_cmd                                            */
/*==============================================================*/
create table device_cmd (
   id                   INT4                 not null,
   device_key_id        INT4                 null,
   value                INT4                 null,
   name                 VARCHAR(50)          null,
   constraint PK_DEVICE_CMD primary key (id)
);

comment on table device_cmd is
'设备控制命令表';

/*==============================================================*/
/* Table: device_key                                            */
/*==============================================================*/
create table device_key (
   id                   INT4                 not null,
   device_model_id      INT4                 not null,
   seq                  INT4                 null,
   name                 VARCHAR(50)          null,
   can_enum             BOOL                 null,
   max_state_value      INT4                 null,
   min_state_value      INT4                 null,
   alarm_type           INT4                 null,
   constraint PK_DEVICE_KEY primary key (id)
);

comment on table device_key is
'can_enum:
true: 状态，有actuator_state
false:连续值表达

max_value&min_value: 本key的取值范围

alarm_type: 
0：不是防区
10：只在布防状态下告警
20：任何时候都告警';

/*==============================================================*/
/* Table: device_key_code                                       */
/*==============================================================*/
create table device_key_code (
   id                   SERIAL not null,
   device_id            INT4                 not null,
   device_key_id        INT4                 not null,
   key_code             VARCHAR(50)          null,
   name                 VARCHAR(50)          null,
   constraint PK_DEVICE_KEY_CODE primary key (id)
);

comment on table device_key_code is
'维护设备每个key/channel的属性';

/*==============================================================*/
/* Table: device_model                                          */
/*==============================================================*/
create table device_model (
   id                   INT4                 not null,
   device_type_id       INT4                 null,
   name                 VARCHAR(50)          null,
   constraint PK_DEVICE_MODEL primary key (id)
);

comment on table device_model is
'设备型号表
name:如 HRMS-2012S';

/*==============================================================*/
/* Index: sensor_type_PK                                        */
/*==============================================================*/
create unique index sensor_type_PK on device_model (
id
);

/*==============================================================*/
/* Table: device_state                                          */
/*==============================================================*/
create table device_state (
   id                   INT4                 not null,
   device_key_id        INT4                 not null,
   value_begin          INT4                 null,
   value_end            INT4                 null,
   name                 VARCHAR(50)          null,
   alarm_level          INT4                 null,
   constraint PK_DEVICE_STATE primary key (id)
);

comment on table device_state is
'alarm_level:
0-不是告警
>0是告警';

/*==============================================================*/
/* Index: sensor_state_PK                                       */
/*==============================================================*/
create unique index sensor_state_PK on device_state (
id
);

/*==============================================================*/
/* Table: device_type                                           */
/*==============================================================*/
create table device_type (
   id                   INT4                 not null,
   name                 VARCHAR(50)          null,
   constraint PK_DEVICE_TYPE primary key (id)
);

comment on table device_type is
'设备类型表：
name:如“fire”/''gas''';

/*==============================================================*/
/* Table: event                                                 */
/*==============================================================*/
create table event (
   id                   SERIAL not null,
   device_key_id        INT4                 null,
   device_uni_code      VARCHAR(50)          null,
   value                INT4                 null,
   dt                   TIMESTAMP            null,
   alarm_level          INT4                 null,
   constraint PK_EVENT primary key (id)
);

comment on table event is
'设备事件表

superbox_id:
上报事件的superbox

alarm_level:参见device_state表的alarm_level描述';

/*==============================================================*/
/* Table: language                                              */
/*==============================================================*/
create table language (
   id                   INT4                 not null,
   language             VARCHAR(50)          null,
   constraint PK_LANGUAGE primary key (id)
);

/*==============================================================*/
/* Index: language_PK                                           */
/*==============================================================*/
create unique index language_PK on language (
id
);

/*==============================================================*/
/* Table: language_device_state                                 */
/*==============================================================*/
create table language_device_state (
   language_id          INT4                 not null,
   device_state_id      INT4                 not null,
   name                 VARCHAR(50)          null,
   constraint PK_LANGUAGE_DEVICE_STATE primary key (language_id, device_state_id)
);

/*==============================================================*/
/* Table: message_template                                      */
/*==============================================================*/
create table message_template (
   id                   SERIAL not null,
   language_id          INT4                 null,
   account_id           INT4                 null,
   sensor_model_id      INT4                 null,
   template             VARCHAR(500)          null,
   constraint PK_MESSAGE_TEMPLATE primary key (id)
);

comment on table message_template is
'XX_id如果是0，则代表全局。

template: 如：[apartment]于[time]发生[type]告警，请速回！（其中[xxx]是关键字）';

/*==============================================================*/
/* Index: message_template_PK                                   */
/*==============================================================*/
create unique index message_template_PK on message_template (
id
);

/*==============================================================*/
/* Index: language_id_FK                                        */
/*==============================================================*/
create  index language_id_FK on message_template (
language_id
);

/*==============================================================*/
/* Table: param                                                 */
/*==============================================================*/
create table param (
   param_name           VARCHAR(500)         not null,
   param_value          VARCHAR(5000)        null,
   constraint PK_PARAM primary key (param_name)
);


/*==============================================================*/
/* Table: restore_require                                       */
/*==============================================================*/
CREATE TABLE restore_require
(
  id serial NOT NULL,
  account_id integer NOT NULL,
  dt timestamp without time zone NOT NULL,
  uuid character varying(50) NOT NULL,
  finished boolean NOT NULL,
  CONSTRAINT restore_require_pkey PRIMARY KEY (id),
  CONSTRAINT restore_require_account_id_fkey FOREIGN KEY (account_id)
      REFERENCES account (id) MATCH SIMPLE
      ON UPDATE NO ACTION ON DELETE NO ACTION
)
WITH (OIDS=FALSE);
ALTER TABLE restore_require OWNER TO postgres;
COMMENT ON TABLE restore_require IS 'password restore requirements';



/*==============================================================*/
/* Table: scene                                                 */
/*==============================================================*/
create table scene (
   id                   SERIAL not null,
   apartment_id         INT4                 not null,
   name                 VARCHAR(50)          null,
   constraint PK_SCENE primary key (id)
);

comment on table scene is
'场景头表';

/*==============================================================*/
/* Table: scene_content                                         */
/*==============================================================*/
create table scene_content (
   id                   SERIAL not null,
   scene_id             INT4                 not null,
   device_key_code_id   INT4                 not null,
   value                INT4                 null,
   constraint PK_SCENE_CONTENT primary key (id)
);

comment on table scene_content is
'场景生效的执行内容';

/*==============================================================*/
/* Table: sms_sender_head                                       */
/*==============================================================*/
create table sms_sender_head (
   id                   SERIAL not null,
   apartment_id         INT4                 not null,
   content              VARCHAR(256)         null,
   dt                   TIMESTAMP            null,
   constraint PK_SMS_SENDER_HEAD primary key (id)
);

comment on table sms_sender_head is
'content: 短信内容';

/*==============================================================*/
/* Index: sms_sender_head_PK                                    */
/*==============================================================*/
create unique index sms_sender_head_PK on sms_sender_head (
id
);

/*==============================================================*/
/* Table: sms_sender_list                                       */
/*==============================================================*/
create table sms_sender_list (
   id                   SERIAL not null,
   head_id              INT4                 not null,
   mobile_phone         VARCHAR(50)          null,
   result               INT4                 null,
   constraint PK_SMS_SENDER_LIST primary key (id)
);

comment on table sms_sender_list is
'result:
0, send success
other: send failed (具体原因文本写log文件)';

/*==============================================================*/
/* Index: sms_sender_list_PK                                    */
/*==============================================================*/
create unique index sms_sender_list_PK on sms_sender_list (
id
);

/*==============================================================*/
/* Table: superbox                                              */
/*==============================================================*/
create table superbox (
   id                   SERIAL not null,
   uni_code             VARCHAR(50)          null,
   constraint PK_SUPERBOX primary key (id)
);

comment on table superbox is
'uni_code: superbox二维码';

/*==============================================================*/
/* Index: superbox_PK                                           */
/*==============================================================*/
create unique index superbox_PK on superbox (
id
);

alter table Apartment_Superbox
   add constraint FK_APARTMEN_REFERENCE_APARTMEN foreign key (apartment_id)
      references apartment (id)
      on delete cascade on update cascade;

alter table Apartment_Superbox
   add constraint FK_APARTMEN_REFERENCE_SUPERBOX foreign key (superbox_id)
      references superbox (id)
      on delete cascade on update cascade;

alter table account
   add constraint FK_ACCOUNT_REFERENCE_LANGUAGE foreign key (language_id)
      references language (id)
      on delete restrict on update restrict;

alter table alarm_name
   add constraint FK_ALARM_NA_REFERENCE_DEVICE_T foreign key (device_type_id)
      references device_type (id)
      on delete cascade on update cascade;

alter table alarm_name
   add constraint FK_ALARM_NA_REFERENCE_LANGUAGE foreign key (language_id)
      references language (id)
      on delete cascade on update cascade;

alter table apartment
   add constraint FK_APARTMEN_REFERENCE_ACCOUNT foreign key (account_id)
      references account (id)
      on delete cascade on update cascade;

alter table apartment
   add constraint FK_APARTMEN_REFERENCE_SCENE foreign key (scene_id)
      references scene (id)
      on delete set default on update cascade;

alter table contactor
   add constraint FK_CONTACTO_HOUSE_INC_APARTMEN foreign key (apartment_id)
      references apartment (id)
      on delete cascade on update cascade;

alter table device
   add constraint FK_DEVICE_REFERENCE_SUPERBOX foreign key (superbox_id)
      references superbox (id)
      on delete cascade on update cascade;

alter table device
   add constraint "FK_DEVICE_SENSOR TY_DEVICE_M" foreign key (device_model_id)
      references device_model (id)
      on delete restrict on update restrict;

alter table device_cmd
   add constraint FK_DEVICE_C_REFERENCE_DEVICE_K foreign key (device_key_id)
      references device_key (id)
      on delete restrict on update restrict;

alter table device_key
   add constraint FK_DEVICE_K_REFERENCE_DEVICE_M foreign key (device_model_id)
      references device_model (id)
      on delete cascade on update cascade;

alter table device_key_code
   add constraint FK_DEVICE_K_REFERENCE_DEVICE foreign key (device_id)
      references device (id)
      on delete cascade on update cascade;

alter table device_key_code
   add constraint FK_DEVICE_K_REFERENCE_DEVICE_K foreign key (device_key_id)
      references device_key (id)
      on delete cascade on update cascade;

alter table device_model
   add constraint FK_DEVICE_M_REFERENCE_DEVICE_T foreign key (device_type_id)
      references device_type (id)
      on delete restrict on update restrict;

alter table device_state
   add constraint FK_DEVICE_S_REFERENCE_DEVICE_K foreign key (device_key_id)
      references device_key (id)
      on delete cascade on update cascade;

alter table event
   add constraint FK_EVENT_REFERENCE_DEVICE_K foreign key (device_key_id)
      references device_key (id)
      on delete restrict on update restrict;

alter table language_device_state
   add constraint FK_LANGUAGE_REFERENCE_DEVICE_S foreign key (device_state_id)
      references device_state (id)
      on delete cascade on update cascade;

alter table language_device_state
   add constraint FK_LANGUAGE_REFERENCE_LANGUAGE foreign key (language_id)
      references language (id)
      on delete cascade on update cascade;

alter table message_template
   add constraint FK_MESSAGE__REFERENCE_DEVICE_M foreign key (sensor_model_id)
      references device_model (id)
      on delete cascade on update cascade;

alter table message_template
   add constraint FK_MESSAGE__REFERENCE_ACCOUNT foreign key (account_id)
      references account (id)
      on delete cascade on update cascade;

alter table message_template
   add constraint FK_MESSAGE__LANGUAGE__LANGUAGE foreign key (language_id)
      references language (id)
      on delete restrict on update cascade;

alter table scene
   add constraint FK_SCENE_REFERENCE_APARTMEN foreign key (apartment_id)
      references apartment (id)
      on delete cascade on update cascade;

alter table scene_content
   add constraint FK_SCENE_CO_REFERENCE_SCENE foreign key (scene_id)
      references scene (id)
      on delete cascade on update cascade;

alter table scene_content
   add constraint FK_SCENE_CO_REFERENCE_DEVICE_K foreign key (device_key_code_id)
      references device_key_code (id)
      on delete cascade on update cascade;

alter table sms_sender_head
   add constraint "FK_SMS_SEND_SEND HIST_APARTMEN" foreign key (apartment_id)
      references apartment (id)
      on delete restrict on update restrict;

alter table sms_sender_list
   add constraint "FK_SMS_SEND_SEND LIST_SMS_SEND" foreign key (head_id)
      references sms_sender_head (id)
      on delete cascade on update cascade;







insert into language(id,language) values(1,'en-US');
insert into language(id,language) values(2,'zh-CN');
insert into language(id,language) values(3,'zh-TW');


insert into message_template(language_id,account_id,sensor_model_id,template) values(null,null,null,'[apartment]于[time]发生[type]告警');

insert into device_type(id,name) values(1,'light');
insert into device_type(id,name) values(2,'curtain');
insert into device_type(id,name) values(100,'fire');
insert into device_type(id,name) values(101,'gas');
insert into device_type(id,name) values(200,'magnetic');
insert into device_type(id,name) values(201,'Infrared');
insert into device_type(id,name) values(301,'gas controller');

insert into device_model(id,device_type_id,name) values(1,1,'HRMS-2111S');
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(1,1,0,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(10,1,1,1,'on',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(11,1,0,0,'off',0);


insert into device_model(id,device_type_id,name) values(2,2,'HRMS-2131D');
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(2,2,0,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(20,2,1,1,'on',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(21,2,0,0,'off',0);


insert into device_model(id,device_type_id,name) values(3,1,'HRMS-2122D');
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(3,3,0,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(30,3,128,254,'dimmer',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(31,3,255,255,'on',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(32,3,127,127,'off',0);
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(4,3,1,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(33,4,128,254,'dimmer',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(34,4,255,255,'on',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(35,4,127,127,'off',0);


insert into device_model(id,device_type_id,name) values(4,101,'HRMS-2690S');
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(5,4,0,'',true,100,0,20);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(40,5,0,63,'normal',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(41,5,64,999,'emergency',1);

insert into device_model(id,device_type_id,name) values(5,200,'HRMS-5816');
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(6,5,0,'',true,100,0,10);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(50,6,0,31,'normal',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(51,6,32,63,'open',1);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(52,6,64,127,'damaged',1);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(53,6,128,999,'normal',0);


insert into device_model(id,device_type_id,name) values(6,201,'HRMS-5890');
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(7,6,0,'',true,100,0,10);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(60,7,0,63,'normal',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(61,7,64,127,'damage',1);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(62,7,128,999,'intrusion',1);

insert into device_model(id,device_type_id,name) values(7,1,'HRMS-2112S');
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(8,7,0,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(70,8,0,0,'off',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(71,8,1,1,'on',0);
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(9,7,1,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(72,9,0,0,'off',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(73,9,1,1,'on',0);

insert into device_model(id,device_type_id,name) values(8,1,'HRMS-2113S');
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(10,8,0,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(80,10,0,0,'off',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(81,10,1,1,'on',0);
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(11,8,1,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(82,11,0,0,'off',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(83,11,1,1,'on',0);
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(12,8,2,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(84,12,0,0,'off',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(85,12,1,1,'on',0);


insert into device_model(id,device_type_id,name) values(9,1,'HRMS-2114S');
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(13,9,0,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(90,13,0,0,'off',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(91,13,1,1,'on',0);
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(14,9,1,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(92,14,0,0,'off',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(93,14,1,1,'on',0);
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(15,9,2,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(94,15,0,0,'off',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(95,15,1,1,'on',0);
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(16,9,3,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(96,16,0,0,'off',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(97,16,1,1,'on',0);


insert into device_model(id,device_type_id,name) values(10,1,'HRMS-2111D');
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(17,10,0,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(100,17,1,1,'on',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(101,17,0,0,'off',0);


insert into device_model(id,device_type_id,name) values(11,1,'HRMS-2112D');
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(18,11,0,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(110,18,0,0,'off',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(111,18,1,1,'on',0);
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(19,11,1,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(112,19,0,0,'off',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(113,19,1,1,'on',0);

insert into device_model(id,device_type_id,name) values(12,1,'HRMS-2113D');
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(20,12,0,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(120,20,0,0,'off',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(121,20,1,1,'on',0);
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(21,12,1,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(122,21,0,0,'off',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(123,21,1,1,'on',0);
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(22,12,2,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(124,22,0,0,'off',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(125,22,1,1,'on',0);


insert into device_model(id,device_type_id,name) values(13,1,'HRMS-2114D');
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(23,13,0,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(130,23,0,0,'off',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(131,23,1,1,'on',0);
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(24,13,1,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(132,24,0,0,'off',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(133,24,1,1,'on',0);
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(25,13,2,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(134,25,0,0,'off',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(135,25,1,1,'on',0);
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(26,13,3,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(136,26,0,0,'off',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(137,26,1,1,'on',0);

insert into device_model(id,device_type_id,name) values(14,1,'HRMS-2121D');
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(27,14,0,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(140,27,128,254,'dimmer',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(141,27,255,255,'on',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(142,27,127,127,'off',0);

insert into device_model(id,device_type_id,name) values(15,2,'HRMS-2132D');
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(28,15,0,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(150,28,1,1,'on',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(151,28,0,0,'off',0);
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(29,15,1,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(152,29,1,1,'on',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(153,29,0,0,'off',0);

insert into device_model(id,device_type_id,name) values(16,301,'HRMS-2141A');
insert into device_key(id,device_model_id,seq,name,can_enum,max_state_value,min_state_value,alarm_type) values(30,16,0,'',true,100,0,0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(160,30,1,1,'on',0);
insert into device_state(id,device_key_id,value_begin,value_end,name,alarm_level) values(161,30,0,0,'off',0);

update device_model set name=substring(name from 6) where name like 'HRMS-%';


insert into server(id,type,address,status,dt_active) values(1,2,'192.168.11.59','offline','1999-1-1');

