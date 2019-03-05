# This codes bases on: 
# https://pythonprogramming.net/tensorflow-object-detection-api-self-driving-car/ 
# https://github.com/tensorflow/models/tree/master/research/object_detection
# and is also available in its on repo.
# https://github.com/patdring/CarND-Term3-TrafficLightDetection
import rospy
from styx_msgs.msg import TrafficLight
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
#from matplotlib import pyplot as plt
from PIL import Image
from os import path
from utils import label_map_util
#from utils import visualization_utils as vis_util
import time
import cv2

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = r'light_classification/model/frozen_inference_graph.pb'

# number of classes for my dataset
NUM_CLASSES = 4

class TLClassifier(object):
    def __init__(self):
        self.detection_graph = None
        self.current_tl_light = TrafficLight.UNKNOWN               
        # Load a (frozen) Tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        self.category_index = {1: {'id': 1, 'name': 'Red'}, 
                               2: {'id': 2, 'name': 'Yellow'},
                               3: {'id': 3, 'name': 'Green'}, 
                               4: {'id': 4, 'name': 'off'}}

        # create tensorflow session for detection
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # end
        self.sess = tf.Session(graph=self.detection_graph, config=config)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
    
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.expand_dims(image, axis=0)

        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores,
                 self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        red_cnt = 0
        detect_cnt = 0

        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > 0.6:            
                class_name = self.category_index[classes[i]]['name']
                detect_cnt += 1
                # Traffic light thing
                if class_name == 'Red':
                    red_cnt += 1

        if  detect_cnt - red_cnt > red_cnt:
            self.current_light = TrafficLight.GREEN
        else:
            self.current_light = TrafficLight.RED

        return self.current_light