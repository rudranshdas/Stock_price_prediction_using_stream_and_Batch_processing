@echo off
title Kafka Server
cd C:\kafka\kafka_2.12-3.2.0
.\bin\windows\kafka-server-start.bat .\config\server.properties
pause