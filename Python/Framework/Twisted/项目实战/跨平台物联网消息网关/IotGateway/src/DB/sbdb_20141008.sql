-- MySQL dump 10.13  Distrib 5.6.21, for Win32 (x86)
--
-- Host: localhost    Database: sbdb
-- ------------------------------------------------------
-- Server version	5.6.21-log

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `account`
--

DROP TABLE IF EXISTS `account`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `account` (
  `id` int(32) NOT NULL AUTO_INCREMENT,
  `language_id` int(32) NOT NULL,
  `user_name` char(20) DEFAULT NULL,
  `password` char(100) DEFAULT NULL,
  `email` char(255) DEFAULT NULL,
  `mobile_phone` char(50) DEFAULT NULL,
  `version` int(32) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_account_reference_language` (`language_id`),
  CONSTRAINT `fk_account_reference_language` FOREIGN KEY (`language_id`) REFERENCES `language` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `account`
--

LOCK TABLES `account` WRITE;
/*!40000 ALTER TABLE `account` DISABLE KEYS */;
/*!40000 ALTER TABLE `account` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `alarm`
--

DROP TABLE IF EXISTS `alarm`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `alarm` (
  `id` int(32) NOT NULL AUTO_INCREMENT,
  `event_id` int(32) NOT NULL,
  `apartment_device_id` int(32) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `alarm_event_id_fkey` (`event_id`),
  KEY `alarm_apartment_device_id_fkey` (`apartment_device_id`),
  CONSTRAINT `alarm_apartment_device_id_fkey` FOREIGN KEY (`apartment_device_id`) REFERENCES `apartment_device` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `alarm_event_id_fkey` FOREIGN KEY (`event_id`) REFERENCES `event` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `alarm`
--

LOCK TABLES `alarm` WRITE;
/*!40000 ALTER TABLE `alarm` DISABLE KEYS */;
/*!40000 ALTER TABLE `alarm` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `alarm_name`
--

DROP TABLE IF EXISTS `alarm_name`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `alarm_name` (
  `language_id` int(32) NOT NULL,
  `device_type_id` int(32) NOT NULL,
  `name` char(50) DEFAULT NULL,
  PRIMARY KEY (`device_type_id`,`language_id`),
  KEY `fk_alarm_na_reference_language` (`language_id`),
  CONSTRAINT `fk_alarm_na_reference_device_t` FOREIGN KEY (`device_type_id`) REFERENCES `device_type` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_alarm_na_reference_language` FOREIGN KEY (`language_id`) REFERENCES `language` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `alarm_name`
--

LOCK TABLES `alarm_name` WRITE;
/*!40000 ALTER TABLE `alarm_name` DISABLE KEYS */;
/*!40000 ALTER TABLE `alarm_name` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `apartment`
--

DROP TABLE IF EXISTS `apartment`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `apartment` (
  `id` int(32) NOT NULL AUTO_INCREMENT,
  `account_id` int(32) NOT NULL,
  `name` char(50) DEFAULT NULL,
  `arm_state` int(32) DEFAULT NULL,
  `scene_id` int(32) DEFAULT NULL,
  `dt_arm` timestamp NULL DEFAULT NULL,
  `version` int(32) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `apartment_pk` (`id`),
  KEY `fk_apartmen_reference_account` (`account_id`),
  CONSTRAINT `fk_apartmen_reference_account` FOREIGN KEY (`account_id`) REFERENCES `account` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `apartment`
--

LOCK TABLES `apartment` WRITE;
/*!40000 ALTER TABLE `apartment` DISABLE KEYS */;
/*!40000 ALTER TABLE `apartment` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `apartment_device`
--

DROP TABLE IF EXISTS `apartment_device`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `apartment_device` (
  `id` int(32) NOT NULL AUTO_INCREMENT,
  `apartment_id` int(32) DEFAULT NULL,
  `name` char(50) NOT NULL,
  `superbox_id` int(32) DEFAULT '0',
  `device_id` int(32) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `apartment_device_apartment_id_fkey` (`apartment_id`),
  KEY `apartment_device_device_id_fkey` (`device_id`),
  CONSTRAINT `apartment_device_apartment_id_fkey` FOREIGN KEY (`apartment_id`) REFERENCES `apartment` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `apartment_device_device_id_fkey` FOREIGN KEY (`device_id`) REFERENCES `device` (`id`) ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `apartment_device`
--

LOCK TABLES `apartment_device` WRITE;
/*!40000 ALTER TABLE `apartment_device` DISABLE KEYS */;
/*!40000 ALTER TABLE `apartment_device` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `apartment_device_key`
--

DROP TABLE IF EXISTS `apartment_device_key`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `apartment_device_key` (
  `id` int(32) NOT NULL AUTO_INCREMENT,
  `apartment_device_id` int(32) DEFAULT NULL,
  `device_key_code_id` int(32) DEFAULT NULL,
  `name` char(50) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `apartment_device_key_apartment_device_id_fkey` (`apartment_device_id`),
  KEY `apartment_device_key_device_key_code_id_fkey` (`device_key_code_id`),
  CONSTRAINT `apartment_device_key_apartment_device_id_fkey` FOREIGN KEY (`apartment_device_id`) REFERENCES `apartment_device` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `apartment_device_key_device_key_code_id_fkey` FOREIGN KEY (`device_key_code_id`) REFERENCES `device_key_code` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `apartment_device_key`
--

LOCK TABLES `apartment_device_key` WRITE;
/*!40000 ALTER TABLE `apartment_device_key` DISABLE KEYS */;
/*!40000 ALTER TABLE `apartment_device_key` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `apartment_superbox`
--

DROP TABLE IF EXISTS `apartment_superbox`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `apartment_superbox` (
  `apartment_id` int(32) NOT NULL,
  `superbox_id` int(32) NOT NULL,
  `name` char(50) DEFAULT NULL,
  PRIMARY KEY (`superbox_id`,`apartment_id`),
  KEY `fk_apartmen_reference_apartmen` (`apartment_id`),
  CONSTRAINT `fk_apartmen_reference_apartmen` FOREIGN KEY (`apartment_id`) REFERENCES `apartment` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_apartmen_reference_superbox` FOREIGN KEY (`superbox_id`) REFERENCES `superbox` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `apartment_superbox`
--

LOCK TABLES `apartment_superbox` WRITE;
/*!40000 ALTER TABLE `apartment_superbox` DISABLE KEYS */;
/*!40000 ALTER TABLE `apartment_superbox` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `client`
--

DROP TABLE IF EXISTS `client`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `client` (
  `id` int(32) NOT NULL AUTO_INCREMENT,
  `account_id` int(32) NOT NULL,
  `device_token` char(100) DEFAULT NULL,
  `enable_alarm` tinyint(1) DEFAULT NULL,
  `os` char(50) DEFAULT NULL,
  `dt_auth` timestamp NULL DEFAULT NULL,
  `dt_active` timestamp NULL DEFAULT NULL,
  `server_id` int(32) DEFAULT NULL,
  `terminal_code` char(50) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `ak_client_ur_client` (`device_token`,`account_id`),
  UNIQUE KEY `client_device_token_key` (`device_token`),
  KEY `fk_client_reference_account` (`account_id`),
  CONSTRAINT `fk_client_reference_account` FOREIGN KEY (`account_id`) REFERENCES `account` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `client`
--

LOCK TABLES `client` WRITE;
/*!40000 ALTER TABLE `client` DISABLE KEYS */;
/*!40000 ALTER TABLE `client` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `contactor`
--

DROP TABLE IF EXISTS `contactor`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `contactor` (
  `id` int(32) NOT NULL AUTO_INCREMENT,
  `apartment_id` int(32) NOT NULL,
  `name` char(50) DEFAULT NULL,
  `mobile_phone` char(50) DEFAULT NULL,
  `email_addr` char(200) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `contactor_pk` (`id`),
  KEY `fk_contacto_house_inc_apartmen` (`apartment_id`),
  CONSTRAINT `fk_contacto_house_inc_apartmen` FOREIGN KEY (`apartment_id`) REFERENCES `apartment` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `contactor`
--

LOCK TABLES `contactor` WRITE;
/*!40000 ALTER TABLE `contactor` DISABLE KEYS */;
/*!40000 ALTER TABLE `contactor` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `device`
--

DROP TABLE IF EXISTS `device`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `device` (
  `id` int(32) NOT NULL AUTO_INCREMENT,
  `device_model_id` int(32) NOT NULL,
  `uni_code` char(50) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `device_pk` (`id`),
  KEY `FK_DEVICE_SENSOR TY_DEVICE_M` (`device_model_id`),
  CONSTRAINT `FK_DEVICE_SENSOR TY_DEVICE_M` FOREIGN KEY (`device_model_id`) REFERENCES `device_model` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `device`
--

LOCK TABLES `device` WRITE;
/*!40000 ALTER TABLE `device` DISABLE KEYS */;
INSERT INTO `device` VALUES (1,1,'26');
/*!40000 ALTER TABLE `device` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `device_cmd`
--

DROP TABLE IF EXISTS `device_cmd`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `device_cmd` (
  `id` int(32) NOT NULL,
  `device_key_id` int(32) DEFAULT NULL,
  `value` int(32) DEFAULT NULL,
  `name` char(50) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_device_c_reference_device_k` (`device_key_id`),
  CONSTRAINT `fk_device_c_reference_device_k` FOREIGN KEY (`device_key_id`) REFERENCES `device_key` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `device_cmd`
--

LOCK TABLES `device_cmd` WRITE;
/*!40000 ALTER TABLE `device_cmd` DISABLE KEYS */;
/*!40000 ALTER TABLE `device_cmd` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `device_key`
--

DROP TABLE IF EXISTS `device_key`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `device_key` (
  `id` int(32) NOT NULL,
  `device_model_id` int(32) NOT NULL,
  `seq` int(32) DEFAULT NULL,
  `name` char(50) DEFAULT NULL,
  `can_enum` tinyint(1) DEFAULT NULL,
  `max_state_value` int(32) DEFAULT NULL,
  `min_state_value` int(32) DEFAULT NULL,
  `alarm_type` int(32) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_device_k_reference_device_m` (`device_model_id`),
  CONSTRAINT `fk_device_k_reference_device_m` FOREIGN KEY (`device_model_id`) REFERENCES `device_model` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `device_key`
--

LOCK TABLES `device_key` WRITE;
/*!40000 ALTER TABLE `device_key` DISABLE KEYS */;
INSERT INTO `device_key` VALUES (1,1,0,'',1,100,0,0),(2,2,0,'',1,100,0,0),(3,3,0,'',1,100,0,0),(4,3,1,'',1,100,0,0),(5,4,0,'',1,100,0,20),(6,5,0,'',1,100,0,10),(7,6,0,'',1,100,0,10),(8,7,0,'',1,100,0,0),(9,7,1,'',1,100,0,0),(10,8,0,'',1,100,0,0),(11,8,1,'',1,100,0,0),(12,8,2,'',1,100,0,0),(13,9,0,'',1,100,0,0),(14,9,1,'',1,100,0,0),(15,9,2,'',1,100,0,0),(16,9,3,'',1,100,0,0),(17,10,0,'',1,100,0,0),(18,11,0,'',1,100,0,0),(19,11,1,'',1,100,0,0),(20,12,0,'',1,100,0,0),(21,12,1,'',1,100,0,0),(22,12,2,'',1,100,0,0),(23,13,0,'',1,100,0,0),(24,13,1,'',1,100,0,0),(25,13,2,'',1,100,0,0),(26,13,3,'',1,100,0,0),(27,14,0,'',1,100,0,0),(28,15,0,'',1,100,0,0),(29,15,1,'',1,100,0,0),(30,16,0,'',1,100,0,0);
/*!40000 ALTER TABLE `device_key` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `device_key_code`
--

DROP TABLE IF EXISTS `device_key_code`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `device_key_code` (
  `id` int(32) NOT NULL AUTO_INCREMENT,
  `device_id` int(32) NOT NULL,
  `device_key_id` int(32) NOT NULL,
  `key_code` char(50) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_device_k_reference_device` (`device_id`),
  KEY `fk_device_k_reference_device_k` (`device_key_id`),
  CONSTRAINT `fk_device_k_reference_device` FOREIGN KEY (`device_id`) REFERENCES `device` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_device_k_reference_device_k` FOREIGN KEY (`device_key_id`) REFERENCES `device_key` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `device_key_code`
--

LOCK TABLES `device_key_code` WRITE;
/*!40000 ALTER TABLE `device_key_code` DISABLE KEYS */;
INSERT INTO `device_key_code` VALUES (1,1,1,'12345678');
/*!40000 ALTER TABLE `device_key_code` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `device_model`
--

DROP TABLE IF EXISTS `device_model`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `device_model` (
  `id` int(32) NOT NULL,
  `device_type_id` int(32) DEFAULT NULL,
  `name` char(50) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `sensor_type_pk` (`id`),
  KEY `fk_device_m_reference_device_t` (`device_type_id`),
  CONSTRAINT `fk_device_m_reference_device_t` FOREIGN KEY (`device_type_id`) REFERENCES `device_type` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `device_model`
--

LOCK TABLES `device_model` WRITE;
/*!40000 ALTER TABLE `device_model` DISABLE KEYS */;
INSERT INTO `device_model` VALUES (1,1,'2111S'),(2,2,'2131D'),(3,1,'2122D'),(4,101,'2690S'),(5,200,'5816'),(6,201,'5890'),(7,1,'2112S'),(8,1,'2113S'),(9,1,'2114S'),(10,1,'2111D'),(11,1,'2112D'),(12,1,'2113D'),(13,1,'2114D'),(14,1,'2121D'),(15,2,'2132D'),(16,301,'2141A');
/*!40000 ALTER TABLE `device_model` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `device_state`
--

DROP TABLE IF EXISTS `device_state`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `device_state` (
  `id` int(32) NOT NULL,
  `device_key_id` int(32) NOT NULL,
  `value_begin` int(32) DEFAULT NULL,
  `value_end` int(32) DEFAULT NULL,
  `name` char(50) DEFAULT NULL,
  `alarm_level` int(32) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `sensor_state_pk` (`id`),
  KEY `fk_device_s_reference_device_k` (`device_key_id`),
  CONSTRAINT `fk_device_s_reference_device_k` FOREIGN KEY (`device_key_id`) REFERENCES `device_key` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `device_state`
--

LOCK TABLES `device_state` WRITE;
/*!40000 ALTER TABLE `device_state` DISABLE KEYS */;
INSERT INTO `device_state` VALUES (10,1,1,1,'on',0),(11,1,0,0,'off',0),(20,2,1,1,'on',0),(21,2,0,0,'off',0),(30,3,128,254,'dimmer',0),(31,3,255,255,'on',0),(32,3,127,127,'off',0),(33,4,128,254,'dimmer',0),(34,4,255,255,'on',0),(35,4,127,127,'off',0),(40,5,0,63,'normal',0),(41,5,64,999,'emergency',1),(50,6,0,31,'normal',0),(51,6,32,63,'open',1),(52,6,64,127,'damaged',1),(53,6,128,999,'normal',0),(60,7,0,63,'normal',0),(61,7,64,127,'damage',1),(62,7,128,999,'intrusion',1),(70,8,0,0,'off',0),(71,8,1,1,'on',0),(72,9,0,0,'off',0),(73,9,1,1,'on',0),(80,10,0,0,'off',0),(81,10,1,1,'on',0),(82,11,0,0,'off',0),(83,11,1,1,'on',0),(84,12,0,0,'off',0),(85,12,1,1,'on',0),(90,13,0,0,'off',0),(91,13,1,1,'on',0),(92,14,0,0,'off',0),(93,14,1,1,'on',0),(94,15,0,0,'off',0),(95,15,1,1,'on',0),(96,16,0,0,'off',0),(97,16,1,1,'on',0),(100,17,1,1,'on',0),(101,17,0,0,'off',0),(110,18,0,0,'off',0),(111,18,1,1,'on',0),(112,19,0,0,'off',0),(113,19,1,1,'on',0),(120,20,0,0,'off',0),(121,20,1,1,'on',0),(122,21,0,0,'off',0),(123,21,1,1,'on',0),(124,22,0,0,'off',0),(125,22,1,1,'on',0),(130,23,0,0,'off',0),(131,23,1,1,'on',0),(132,24,0,0,'off',0),(133,24,1,1,'on',0),(134,25,0,0,'off',0),(135,25,1,1,'on',0),(136,26,0,0,'off',0),(137,26,1,1,'on',0),(140,27,128,254,'dimmer',0),(141,27,255,255,'on',0),(142,27,127,127,'off',0),(150,28,1,1,'on',0),(151,28,0,0,'off',0),(152,29,1,1,'on',0),(153,29,0,0,'off',0),(160,30,1,1,'on',0),(161,30,0,0,'off',0);
/*!40000 ALTER TABLE `device_state` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `device_type`
--

DROP TABLE IF EXISTS `device_type`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `device_type` (
  `id` int(32) NOT NULL,
  `name` char(50) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `device_type`
--

LOCK TABLES `device_type` WRITE;
/*!40000 ALTER TABLE `device_type` DISABLE KEYS */;
INSERT INTO `device_type` VALUES (1,'light'),(2,'curtain'),(100,'fire'),(101,'gas'),(200,'magnetic'),(201,'Infrared'),(301,'gas controller');
/*!40000 ALTER TABLE `device_type` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `event`
--

DROP TABLE IF EXISTS `event`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `event` (
  `id` int(32) NOT NULL AUTO_INCREMENT,
  `device_key_code_id` int(32) DEFAULT NULL,
  `value` int(32) DEFAULT NULL,
  `dt` timestamp NULL DEFAULT NULL,
  `alarm_level` int(32) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `event_device_key_code_id_fkey` (`device_key_code_id`),
  CONSTRAINT `event_device_key_code_id_fkey` FOREIGN KEY (`device_key_code_id`) REFERENCES `device_key_code` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `event`
--

LOCK TABLES `event` WRITE;
/*!40000 ALTER TABLE `event` DISABLE KEYS */;
/*!40000 ALTER TABLE `event` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `language`
--

DROP TABLE IF EXISTS `language`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `language` (
  `id` int(32) NOT NULL,
  `language` char(50) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `language_pk` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `language`
--

LOCK TABLES `language` WRITE;
/*!40000 ALTER TABLE `language` DISABLE KEYS */;
INSERT INTO `language` VALUES (1,'en-US'),(2,'zh-CN'),(3,'zh-TW');
/*!40000 ALTER TABLE `language` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `language_device_state`
--

DROP TABLE IF EXISTS `language_device_state`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `language_device_state` (
  `language_id` int(32) NOT NULL,
  `device_state_id` int(32) NOT NULL,
  `name` char(50) DEFAULT NULL,
  PRIMARY KEY (`device_state_id`,`language_id`),
  KEY `fk_language_reference_language` (`language_id`),
  CONSTRAINT `fk_language_reference_device_s` FOREIGN KEY (`device_state_id`) REFERENCES `device_state` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_language_reference_language` FOREIGN KEY (`language_id`) REFERENCES `language` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `language_device_state`
--

LOCK TABLES `language_device_state` WRITE;
/*!40000 ALTER TABLE `language_device_state` DISABLE KEYS */;
/*!40000 ALTER TABLE `language_device_state` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `message_template`
--

DROP TABLE IF EXISTS `message_template`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `message_template` (
  `id` int(32) NOT NULL AUTO_INCREMENT,
  `language_id` int(32) DEFAULT NULL,
  `account_id` int(32) DEFAULT NULL,
  `sensor_model_id` int(32) DEFAULT NULL,
  `template` longtext,
  PRIMARY KEY (`id`),
  KEY `language_id_fk` (`language_id`),
  KEY `message_template_pk` (`id`),
  KEY `fk_message__reference_device_m` (`sensor_model_id`),
  KEY `fk_message__reference_account` (`account_id`),
  CONSTRAINT `fk_message__language__language` FOREIGN KEY (`language_id`) REFERENCES `language` (`id`) ON UPDATE CASCADE,
  CONSTRAINT `fk_message__reference_account` FOREIGN KEY (`account_id`) REFERENCES `account` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_message__reference_device_m` FOREIGN KEY (`sensor_model_id`) REFERENCES `device_model` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `message_template`
--

LOCK TABLES `message_template` WRITE;
/*!40000 ALTER TABLE `message_template` DISABLE KEYS */;
INSERT INTO `message_template` VALUES (1,NULL,NULL,NULL,'[apartment]Óm[time]·tÉm[type]¸t¾p');
/*!40000 ALTER TABLE `message_template` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `param`
--

DROP TABLE IF EXISTS `param`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `param` (
  `param_name` longtext NOT NULL,
  `param_value` longtext,
  PRIMARY KEY (`param_name`(10))
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `param`
--

LOCK TABLES `param` WRITE;
/*!40000 ALTER TABLE `param` DISABLE KEYS */;
/*!40000 ALTER TABLE `param` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `restore_require`
--

DROP TABLE IF EXISTS `restore_require`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `restore_require` (
  `id` int(32) NOT NULL AUTO_INCREMENT,
  `account_id` int(32) NOT NULL,
  `uuid` char(50) NOT NULL,
  `finished` tinyint(1) NOT NULL,
  `dt` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `restore_require_account_id_fkey` (`account_id`),
  CONSTRAINT `restore_require_account_id_fkey` FOREIGN KEY (`account_id`) REFERENCES `account` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `restore_require`
--

LOCK TABLES `restore_require` WRITE;
/*!40000 ALTER TABLE `restore_require` DISABLE KEYS */;
/*!40000 ALTER TABLE `restore_require` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `scene`
--

DROP TABLE IF EXISTS `scene`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `scene` (
  `id` int(32) NOT NULL AUTO_INCREMENT,
  `apartment_id` int(32) NOT NULL,
  `name` char(50) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_scene_reference_apartmen` (`apartment_id`),
  CONSTRAINT `fk_scene_reference_apartmen` FOREIGN KEY (`apartment_id`) REFERENCES `apartment` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `scene`
--

LOCK TABLES `scene` WRITE;
/*!40000 ALTER TABLE `scene` DISABLE KEYS */;
/*!40000 ALTER TABLE `scene` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `scene_content`
--

DROP TABLE IF EXISTS `scene_content`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `scene_content` (
  `id` int(32) NOT NULL AUTO_INCREMENT,
  `scene_id` int(32) NOT NULL,
  `device_key_code_id` int(32) NOT NULL,
  `value` int(32) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_scene_co_reference_scene` (`scene_id`),
  KEY `fk_scene_co_reference_device_k` (`device_key_code_id`),
  CONSTRAINT `fk_scene_co_reference_device_k` FOREIGN KEY (`device_key_code_id`) REFERENCES `device_key_code` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_scene_co_reference_scene` FOREIGN KEY (`scene_id`) REFERENCES `scene` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `scene_content`
--

LOCK TABLES `scene_content` WRITE;
/*!40000 ALTER TABLE `scene_content` DISABLE KEYS */;
/*!40000 ALTER TABLE `scene_content` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `server`
--

DROP TABLE IF EXISTS `server`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `server` (
  `id` int(32) NOT NULL,
  `type` int(32) NOT NULL,
  `address` char(50) DEFAULT NULL,
  `status` char(50) DEFAULT NULL,
  `dt_active` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `server`
--

LOCK TABLES `server` WRITE;
/*!40000 ALTER TABLE `server` DISABLE KEYS */;
INSERT INTO `server` VALUES (1,2,'192.168.11.59','offline','2014-10-08 08:06:37');
/*!40000 ALTER TABLE `server` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `sms_sender_head`
--

DROP TABLE IF EXISTS `sms_sender_head`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `sms_sender_head` (
  `id` int(32) NOT NULL AUTO_INCREMENT,
  `apartment_id` int(32) NOT NULL,
  `content` longtext,
  `dt` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `sms_sender_head_pk` (`id`),
  KEY `FK_SMS_SEND_SEND HIST_APARTMEN` (`apartment_id`),
  CONSTRAINT `FK_SMS_SEND_SEND HIST_APARTMEN` FOREIGN KEY (`apartment_id`) REFERENCES `apartment` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `sms_sender_head`
--

LOCK TABLES `sms_sender_head` WRITE;
/*!40000 ALTER TABLE `sms_sender_head` DISABLE KEYS */;
/*!40000 ALTER TABLE `sms_sender_head` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `sms_sender_list`
--

DROP TABLE IF EXISTS `sms_sender_list`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `sms_sender_list` (
  `id` int(32) NOT NULL AUTO_INCREMENT,
  `head_id` int(32) NOT NULL,
  `mobile_phone` char(50) DEFAULT NULL,
  `result` int(32) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `sms_sender_list_pk` (`id`),
  KEY `FK_SMS_SEND_SEND LIST_SMS_SEND` (`head_id`),
  CONSTRAINT `FK_SMS_SEND_SEND LIST_SMS_SEND` FOREIGN KEY (`head_id`) REFERENCES `sms_sender_head` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `sms_sender_list`
--

LOCK TABLES `sms_sender_list` WRITE;
/*!40000 ALTER TABLE `sms_sender_list` DISABLE KEYS */;
/*!40000 ALTER TABLE `sms_sender_list` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `superbox`
--

DROP TABLE IF EXISTS `superbox`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `superbox` (
  `id` int(32) NOT NULL AUTO_INCREMENT,
  `uni_code` char(50) DEFAULT NULL,
  `dt_auth` timestamp NULL DEFAULT NULL,
  `dt_active` timestamp NULL DEFAULT NULL,
  `server_id` int(32) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `superbox_pk` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `superbox`
--

LOCK TABLES `superbox` WRITE;
/*!40000 ALTER TABLE `superbox` DISABLE KEYS */;
INSERT INTO `superbox` VALUES (1,'sb_code11','2014-10-08 07:32:33',NULL,1);
/*!40000 ALTER TABLE `superbox` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2014-10-08 16:08:48
