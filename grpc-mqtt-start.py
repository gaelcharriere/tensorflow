#!/usr/bin/python3

# gRPC client detecting only the following objects on images:
# (person=1, bicycle=2, car=3, motorcycle=4, bus=6, truck=8)
# The other objects detected by the model are filtered out.
# Images are loaded locally and sent to the gRPC server to
# be analyzed. The gRPC server returns an output containing:
# * classes: The type of image detected
#            Classes are used to filter out the objects that
#            we are not interested in
# * boxes:   The box coordinates surrounding the object
#            Used by the client to draw the boxes around
#            the detected object
# * scores   The score of the detection
#            Below a defined minimum score we can filter out
#            the object detected
#
# gRPC client connects to a gRPC ModelServer at 0.0.0.0:8500.
# The gRPC ModelServer is a based out of Tensorflow Serving with Docker.
# We tell TensorFlow serving to load the model named: efficientdet_d0
#   docker run --rm -p 8500:8500 --mount type=bind,source=$(pwd),target=$(pwd) \ 
#     -e MODEL_BASE_PATH=$(pwd)/models -e MODEL_NAME=efficientdet_d0 -t tensorflow/serving
# 
# This script is tested and validated only for 
#   'efficientdet_d0 model' that accepts a tensor proto as input.
#
# "input_tensor": {
#   "dtype": "DT_UINT8",
#   "tensor_shape": {
#     "dim": [
#       { "size": "1", "name": ""},
#       { "size": "-1", "name": ""},
#       { "size": "-1", "name": ""},
#       { "size": "3", "name": "" }
# ]}}
# "outputs": { 
#   "detection_classes": {
#     "dtype": "DT_FLOAT",
#       "tensor_shape": {
#         "dim": [
#           { "size": "1", "name": ""},
#           { "size": "100", "name": "" }
#   ]}},
#   "detection_boxes": {
#     "dtype": "DT_FLOAT",
#       "tensor_shape": {
#         "dim": [
#           { "size": "1", "name": "" },
#           { "size": "100", "name": "" },
#           { "size": "4", "name": "" }
#   ]}},
#   "detection_scores": {
#     "dtype": "DT_FLOAT",
#       "tensor_shape": {
#         "dim": [
#           { "size": "1", "name": "" },
#           { "size": "100", "name": "" }
#  ]}}}
from __future__ import print_function
import argparse
import configparser
import logging
import multiprocessing
import numpy as np
import time
import math
from datetime import datetime

import tensorflow as tf
import grpc
import os
import json
import paho.mqtt.client as mqtt
from PIL import Image
from six import BytesIO
from six.moves.urllib.request import urlopen

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# label classes and draw boxes helper
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from influxdb import InfluxDBClient

# Get the absolute directory of the current script
dir_path = os.path.dirname(os.path.realpath(__file__) )

# tgt image directory for processed images
# object detected surrounded by boxes
img_path = '/docker/home-assistant/config/tmp'
# log file
logfile = '/var/log/tensorflow.log'
logging.basicConfig(filename=logfile, format='%(asctime)s - %(message)s', level=logging.INFO)

# ### Load label map data (for plotting).
# 
# Label maps correspond index numbers to category names, 
# so that when our convolution network predicts `5`, 
# we know that this corresponds to `airplane`.  
# Here we use internal utility functions, 
# but anything that returns a dictionary mapping integers 
# to appropriate string labels would be fine.
# 
# We are going, for simplicity, to load from the repository that we loaded the Object Detection API code
PATH_TO_LABELS = dir_path + '/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Load cameras urls
PATH_TO_CAMERAS = dir_path + '/data/cam_snap.json'
with open(PATH_TO_CAMERAS) as json_file:
  cams = json.load(json_file)

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("tensorflow")

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8")
    print("MSG RECEIVED: " + msg.topic + ", " + payload)
    logging.info("MSG RECEIVED: %s, %s", msg.topic, payload)
    
    # start timer and image process counter
    tt = time.time()

    if payload == 'disconnect':
      # exit program
      print('client disconnecting...')
      client.disconnect()
      exit(0)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    if payload == 'all':
      # load all images once asyn
      for c in cams['camera']:
        # define run detection process per image
        p = multiprocessing.Process(target=run_detection, args=(stub, request, c['url'], c['name'], c['classes'], c['score'], return_dict))
        jobs.append(p)
        # start the running detection process
        p.start()
    else:
      # load a single image
      for c in cams['camera']:
        if c['name'] == payload:
          # define run detection process per image
          p = multiprocessing.Process(target=run_detection, args=(stub, request, c['url'], c['name'],c['classes'], c['score'], return_dict))
          jobs.append(p)
          # start the running detection process
          p.start()
    
    # wait all the jobs to be completed      
    for j in jobs:
      j.join()

    # update ha sensor
    # each camera has an sensor associated containing the number
    # of objects per class of object
    # loop over return keys
    for k, v in return_dict.items():
      # ['chemin'] = [0,0,0,0,0,1]
      # number of objects per class
      MQTT_MSG=json.dumps({"detected":"on","person": v[0],"bicycle":v[1],"car":v[2],"motorcycle":v[3],"bus":v[4],"truck":v[5]})
      client.publish("detection/" + k, MQTT_MSG)

    elapsed = round(time.time()-tt,2)
    print(len(jobs),"images processed in",elapsed,"secs")
    logging.info("%d images processed in %.2f secs", len(jobs), elapsed)

# ## Utilities
# 
# Load an image from file or http location into a numpy array.
# Puts image into numpy array to feed into tensorflow graph.
# Note that by convention we put it into a numpy arrray with shape
# (height, width, channels), where channels=3 for RGB.
# 
# Args: 
#   path: the file path to the image (local or remote)
# 
# Returns:
#   uint8 numpy array with shape (img_height, img_width, 3)
#  
def load_image_into_numpy_array(path):
  image = None
  if(path.startswith('http')):
    response = urlopen(path)
    image_data = response.read()
    image_data = BytesIO(image_data)
    image = Image.open(image_data)
    maxwidth = 640
    (im_width, im_height) = image.size
    ratio = maxwidth/im_width
    image = image.resize((int(im_width*ratio), int(im_height*ratio)), Image.ANTIALIAS)
  else:
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))

  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (1, im_height, im_width, 3)).astype(np.uint8)

# ## Extract the index of classes of interest
# The classes input contains all the classes of all
# the objects detected in the image: [1 2 1 5]
# we want to get the index of images of interest.
# Let's say class 1.
# [1 2 1 5] -> [0, 2]
# The output of the function is an array pointing to
# the index of interest. It will used to squeeze the
# original scores and boxes to the index of interest only
# Args:
#   filtered_classes: the classes we want to filter
#   classes: All the classes detected in the image
#   scores: All the scores detected in the image
#   min_score: the minimum score to keep
#
# Returns:
#   Array containing the index of interest
def find_index(filtered_classes, classes, scores, min_score):
  # extract the indexes of class IDS of interest
  # we want to get the index of classes
  # (person=1, bicycle=2, car=3, motorcycle=4, bus=6, truck=8)
  indexes = []
  # loop over classes detected in the image
  i = 0
  while i < len(classes):
    if classes[i] in filtered_classes:
      # object found in a class we are interested in
      # ensure the score is above the min score to keep it
      if scores[i] >= min_score:
        # save the index
        indexes.append(i)
    i += 1

  return indexes


# ###  Process Tensorflow Serving results per image
# Keep only interesting classes of objects.
# Extract boxes, classes, scores and the number of objects
# into a list.
#  
# Args:
#   res: results to process
#   img_length: the img length linked to the result
#   filtered_classes: the classes we want to filter
#   min_score: the minimum score to keep
#
# Returns:
#   [boxes, classes, scores]
#   
def process_results(res, img_length, filtered_classes, min_score):
  # "outputs": { 
  #   "detection_classes": {
  #     "dtype": "DT_FLOAT",
  #       "tensor_shape": {
  #         "dim": [
  #           { "size": "1", "name": ""},
  #           { "size": "100", "name": "" }
  #   ]}},
  #   "detection_boxes": {
  #     "dtype": "DT_FLOAT",
  #       "tensor_shape": {
  #         "dim": [
  #           { "size": "1", "name": "" },
  #           { "size": "100", "name": "" },
  #           { "size": "4", "name": "" }
  #   ]}},
  #   "detection_scores": {
  #     "dtype": "DT_FLOAT",
  #       "tensor_shape": {
  #         "dim": [
  #           { "size": "1", "name": "" },
  #           { "size": "100", "name": "" }
  #  ]}}}
  boxes = np.reshape(res.outputs['detection_boxes'].float_val, [img_length, 100, 4])[0]
  classes = np.reshape(res.outputs['detection_classes'].float_val, [img_length, 100])[0]
  scores = np.reshape(res.outputs['detection_scores'].float_val, [img_length, 100])[0]
  # convert class float to int
  # [1. 2.] -> [1 2]
  classes = (classes).astype(int)

  # extract the indexes of class IDS of interest
  # we want to get the index of classes
  # (person=1, bicycle=2, car=3, motorcycle=4, bus=6, truck=8)
  indexes = find_index(filtered_classes, classes, scores, min_score)
  
  # filter out the scores boxes and classes we are not interested in
  # we keep only the values at specific indexes
  # squeeze reduce the dimension of arrays with a single dimension
  # that's why we fix a min dimension
  scores = np.squeeze(scores[indexes])
  scores = np.array(scores, ndmin=1)
  boxes = np.squeeze(boxes[indexes])
  boxes = np.array(boxes, ndmin=2)
  classes = np.squeeze(classes[indexes])
  classes = np.array(classes, ndmin=1)

  return [boxes, classes, scores]

# ###  Save np array as image
# The image is saved in home assistant directory
# to be attached to email notification by HA.
#  
# Args:
#   img_as_np: the image to save as np
#   name: the name of the file - camera name
# 
def save_img(img_as_np, name):
  # build image from np
  img = Image.fromarray(img_as_np)

  # save processed image into the target directory
  out = os.path.join(img_path, name + '.jpg')
  img.save(out, format="JPEG", quality=100)

# ### Insert object into database
#
# Args:
#  obj: the new object to insert into the database
#  name: the camera name
#  trigger: true if the object is linked with an alert, false otherwise
def insert_obj_db(obj, name, trigger):
  # obj [boxes, classes, scores]
  # loop over object detected
  # one score per object
  i = 0
  json_body = []
  # use current datime as the id for all the objects of the current image
  current = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
  while i < len(obj[2]):
    data = {}
    data['measurement'] = 'image'
    data['tags'] = {}
    data['tags']['camera'] = name
    data['tags']['score'] = str(obj[2][i])
    data['tags']['class'] = str(obj[1][i])
    data['tags']['trigger'] = trigger
    data['fields'] = {}
    # use current datime as the id for all the objects of the current image
    data['fields']['value'] = current
    # loop over 4 boxes points per object
    data['tags']['ymin'] = str(obj[0][i][0])
    data['tags']['xmin'] = str(obj[0][i][1])
    data['tags']['ymax'] = str(obj[0][i][2])
    data['tags']['xmax'] = str(obj[0][i][3])
    json_body.append(data)
    i += 1

  # insert objects into db
  dbclient.write_points(json_body)


# Try to find the object into the object list
#
# Args:
#  objclass: the class of the new object to find
#  ymin: the ymin of the new object to find
#  xmin: the xmin of the new object to find
#  ymax: the ymax of the new object to find
#  xmax: the xmax of the new object to find
#  objects: the list of objects from the db to look for
def find_obj(objclass, score, ymin, xmin, ymax, xmax, objects):
  # last array of objects
  # [{'time', 'camera', 'class', 'score', 'xmax', 'xmin', 'ymax', 'ymin'}]
  
  # loop over the objects saved into the db
  for obj in objects:
    if obj['class'] != objclass:
      continue
    if not math.isclose(float(ymin), float(obj['ymin']), abs_tol= 0.02):
      print(ymin)
      print(obj['ymin'])
      print('different ymin')
      continue
    if not math.isclose(float(xmin), float(obj['xmin']), abs_tol= 0.02):
      print(xmin)
      print(obj['xmin'])
      print('different xmin')
      continue
    if not math.isclose(float(ymax), float(obj['ymax']), abs_tol= 0.02):
      print(ymax)
      print(obj['ymax'])
      print('different ymax')
      continue
    if not math.isclose(float(xmax), float(obj['xmax']), abs_tol= 0.02):
      print(xmax)
      print(obj['xmax'])
      print('different xmax')
      continue
    # similar object found
    print('Object already detected')
    logging.info("Object already detected")
    return True
  # object not found
  return False

# Find new objects. The goal of this function is
# to detect if new objects were found or not.
#
# Args:
#  last: the last objects from the db
#  new: the current objects detected on the image
# Return: True if new object found, false otherwise
def new_obj_found(last, new):
  # last array of objects
  # [{'time', 'camera', 'class', 'score', 'xmax', 'xmin', 'ymax', 'ymin'}]
  # new array of objects
  # ymin: str(new[0][i][0])
  # xmin: str(new[0][i][1])
  # ymax: str(new[0][i][2])
  # xmax: str(new[0][i][3])
  # class: str(new[1][i])
  # score: str(new[2][i])

  # check if there is at least one object found
  if len(new[2]) == 0:
    return False

  # if no previous event found, we now for
  # sure that the event is a new one
  if len(last) == 0:
    return True
  
  # first compare the number of objects found
  if len(new[2]) > len(last):
    return True

  # loop over new objects
  i = 0
  while i < len(new[2]):
    # class of object new[1][i]
    # (person=1, bicycle=2, car=3, motorcycle=4, bus=6, truck=8)

    # let's try to identify
    # if the object was already discovered.
    # An object with same class, similar score, xmax, xmin, ymax and ymin
    # is considered as identical
    if not find_obj(
      str(new[1][i]),
      str(new[2][i]),
      str(new[0][i][0]),
      str(new[0][i][1]),
      str(new[0][i][2]),
      str(new[0][i][3]), last
    ):
      # if the object is not found, it means
      # that we discovered a new object...
      return True

    # move to next object
    i += 1

  # no new object found
  return False

# get last objects detected from db for a camera
#
# Args:
#  name: the name of the camera
# Return: last objects found for this camera
def get_last_obj(name):
  # get last id found for this camera
  query = "SELECT last(value) FROM image WHERE camera='" + name + "';"
  rs = dbclient.query(query)
  if len(rs) == 0:
    # no previous object found, return empty array
    return []
  else:
    lastid = list(rs.get_points(measurement='image'))[0]['last']
    # find all the objects detected for the last id
    query = "SELECT * FROM image WHERE camera='" + name + "' AND value='" + lastid + "';"
    rs = dbclient.query(query)
    last = list(rs.get_points(measurement='image'))
    # [{'time', 'camera', 'class', 'score', 'xmax', 'xmin', 'ymax', 'ymin'}]
    return last

# determine if a new alert is triggered or not based on conditions
# condition 1: no alert already triggered last minute
# condition 2: no more than x alerts during an interval of time in minutes
#
# Args:
#  name: the name of the camera
#  threshold: the max number of alerts allowed per interval of time
#  min: the interval of time in minutes from now
# Return: true if the alert is triggered, false otherwise
def trigger_alert(name, threshold, min):
  # get number of events from the same image during the last interval
  query = "SELECT COUNT(DISTINCT(value)) FROM image WHERE camera='" + name + "' AND trigger='True' AND time > now()-" + str(min) + "m;"
  rs = dbclient.query(query)
  if len(rs) > 0:
    # result set not empty
    count = list(rs.get_points(measurement='image'))[0]['count']
    if count > threshold:
      print('events already detected above threshold: ',count,'/5 (last hour)')
      logging.info("events already detected above threshold: %d/5 (last hour)", count)
      return False

  # check if a detection alert was already triggered during the last minute
  query = "SELECT COUNT(DISTINCT(value)) FROM image WHERE camera='" + name + "' AND trigger='True' AND time > now()-1m;"
  rs = dbclient.query(query)
  if len(rs) > 0:
    # result set not empty
    count = list(rs.get_points(measurement='image'))[0]['count']
    if count > 0:
      print('events already detected above threshold: ',count,'/1 (last min)')
      logging.info("events already detected above threshold: %d/1 (last min)", count)
      return False
  
  # trigger alert
  return True

# ### Run detection for an image
# - convert the image into a tensor
# - send the request to tensorfow serving
# - process the result
# - save the output image with boxes
#
# Args:
#   stub: the stub used to send the gRPC request
#   request: the tensorflow serving request
#   image: the local or remote image path
#   name: the name of the camera
#   classes: the list of classes to look for
#   min_score: the minimum score
#   return_dict: return values
#
def run_detection(stub, request, image, name, classes, min_score, return_dict):
  # read image into numpy array
  img  = load_image_into_numpy_array(image)

  # convert to tensor proto and make request
  # shape is in NHWC (num_samples x height x width x channels) format
  tensor = tf.make_tensor_proto(img, shape=img.shape)

  # inference
  request.inputs['input_tensor'].CopyFrom(tensor)
  res = stub.Predict(request, 60.0) # 60secs timeout

  # extract filtered boxes, classes and scores
  # [boxes, classes, scores]
  obj = process_results(res, len(img), classes, min_score)

  # get previous objects detected from db
  last = get_last_obj(name)

  # ensure new objects found
  if new_obj_found(last, obj):
    # determine if a new alert should be triggered
    # based on previous events already detected
    # more than 5 events during last hour: alert = false
    # already an event last minute: alert = false
    alert = trigger_alert(name, 5, 60)

    # print number of objects found
    print("Found",len(obj[2]),"objects in",name)
    logging.info("Found %d objects in %s.", len(obj[2]), name)
    
    # insert new object in db
    insert_obj_db(obj, name, alert)

    if alert:
      # generate alert
      # copy np image
      img_with_boxes = img.copy()[0]

      # draw boxes on top of image
      viz_utils.visualize_boxes_and_labels_on_image_array(
        img_with_boxes,
        obj[0],     # boxes
        obj[1],     # classes
        obj[2],     # scores
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=min_score,
        agnostic_mode=False,
        keypoints=None,
        keypoint_scores=None,
        keypoint_edges=None)

      # save image with boxes
      save_img(img_with_boxes, name)

      # loop over object detected and count them
      persons = 0
      bicycles = 0
      cars = 0
      motorcycles = 0
      bus = 0
      trucks = 0
      for i in range(len(obj[2])):
        # (person=1, bicycle=2, car=3, motorcycle=4, bus=6, truck=8)
        cat = category_index.get(obj[1][i]).get('name')
        if cat == 'person':
          persons += 1
        elif cat == 'bicycle':
          bicycles += 1
        elif cat == 'car':
          cars += 1
        elif cat == 'motorcycle':
          motorcycles += 1
        elif cat == 'bus':
          bus += 1
        elif cat == 'truck':
          trucks += 1
        print(cat,':',round(obj[2][i]*100,2),'%')
        logging.info("%s: %.2f%%", cat, round(obj[2][i]*100,2))
    
      # return number of object detected for this camera
      return_dict[name] = [persons, bicycles, cars, motorcycles, bus, trucks]

def run(mqtt_host, mqtt_port, mqtt_user, mqtt_pwd, db_host, db_port, db_user, db_pwd, db_name, host, port, model, signature_name):
    # create the gRPC stub
    # specify max receive msg length to avoid max length exception
    options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
    channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port), options=options)
    # global variable initialized once
    # the stub and request are loaded once at startup
    # and then re-used for every image to analyze
    # much more performant since the initial loading
    # takes time
    global stub 
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    global request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = signature_name

    # create mqtt client
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.username_pw_set(username=mqtt_user, password=mqtt_pwd)
    client.connect(mqtt_host, int(mqtt_port), 60)

    # db connexion
    global dbclient
    dbclient = InfluxDBClient(db_host, db_port, db_user, db_pwd, db_name)

    # Blocking call that processes network traffic, dispatches callbacks and
    # handles reconnecting.
    # Other loop*() functions are available that give a threaded interface and a
    # manual interface.
    client.loop_forever()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mqtt_host', default='localhost', help='MQTT broker host name', type=str)
    parser.add_argument('--mqtt_port', default=1883, help='MQTT broker port')
    parser.add_argument('--mqtt_user', help='MQTT broker username')
    parser.add_argument('--mqtt_pwd', help='MQTT broker password')
    parser.add_argument('--db_host', default='localhost', help='InfluxDB host name', type=str)
    parser.add_argument('--db_port', default=8086, help='InfluxDB host port')
    parser.add_argument('--db_user', help='InfluxDB username')
    parser.add_argument('--db_pwd', help='InfluxDB password')
    parser.add_argument('--db_tf', default='tensorflow', help='InfluxDB tensorflow database name')
    parser.add_argument('--tf_host', default='0.0.0.0', help='Tensorflow server host name', type=str)
    parser.add_argument('--tf_port', default=8500, help='Tensorflow server port number', type=int)
    parser.add_argument('--model', default='efficientdet', help='model name', type=str)
    parser.add_argument('--signature_name', default='serving_default', help='Signature name of saved TF model', type=str)
    parser.add_argument('--config', help='Configuration file', type=str)

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

    run(ns['mqtt_host'], ns['mqtt_port'], ns['mqtt_user'], ns['mqtt_pwd'], ns['db_host'], ns['db_port'], ns['db_user'], ns['db_pwd'], ns['db_tf'], ns['tf_host'], ns['tf_port'], ns['model'], ns['signature_name'])
