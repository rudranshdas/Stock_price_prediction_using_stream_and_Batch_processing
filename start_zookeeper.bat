@echo off
title Zookeeper Server
cd C:\kafka\kafka_2.12-3.2.0
.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
pause