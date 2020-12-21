#!/usr/bin/python3

# sudo apt-get install python3-paho-mqtt
# paho mqtt client used to stop the tensorflow mqtt client
# The mqtt client loops for ever to listen to the topic "tensorflow"
# from the home assistant broker
import paho.mqtt.client as mqtt
import os
import argparse
import configparser

def run(mqtt_host, mqtt_port, mqtt_user, mqtt_pwd):
  client = mqtt.Client()
  client.username_pw_set(username=mqtt_user, password=mqtt_pwd)
  client.connect(mqtt_host, int(mqtt_port), 60)
  client.publish("tensorflow","disconnect")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mqtt_host', default='localhost', help='MQTT broker host name', type=str)
    parser.add_argument('--mqtt_port', default=1883, help='MQTT broker port')
    parser.add_argument('--mqtt_user', help='MQTT broker username')
    parser.add_argument('--mqtt_pwd', help='MQTT broker password')
    parser.add_argument('--config', help='OpsHome configuration file', type=str)

    # parse command line
    args = parser.parse_args()
    # translate parse command line to dict: ns['mqtt_host']
    ns = vars(args)

    # if --config is specified, overrides args parameters
    if ns['config']:
      # parse config file override
      config = configparser.ConfigParser()
      if os.path.isfile(args.config):
        config.read(args.config)
        for section in config.sections():
          for (key, val) in config.items(section):
            ns[key] = val

    run(ns['mqtt_host'], ns['mqtt_port'], ns['mqtt_user'], ns['mqtt_pwd'])

