# This codes bases on: 
# https://pythonprogramming.net/tensorflow-object-detection-api-self-driving-car/ 
# https://github.com/tensorflow/models/tree/master/research/object_detection
# and is also available in its on repo.
# https://github.com/patdring/CarND-Term3-TrafficLightDetection

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
PATH_TO_CKPT = 'model/frozen_inference_graph.pb'

# number of classes for my dataset
NUM_CLASSES = 4

class TLClassifier(object):
    def __init__(self):
        self.detection_graph = None
                       
        # Load a (frozen) Tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
    
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor =self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                image_np = load_image_into_numpy_array(image)
                image_expanded = np.expand_dims(image_np, axis=0)
                
                (boxes, scores, classes, num) = sess.run(
                    [detect_boxes, detect_scores, detect_classes, num_detections],
                    feed_dict={image_tensor: image_expanded})
                              
                if (scores[0][0] >= 0.5):
                    if(classes[0][0] == 1):
                        return TrafficLight.RED
                    if(classes[0][0] == 2):
                        return TrafficLight.YELLOW
                    if(classes[0][0] == 3):
                        return TrafficLight.GREEN
                    if(classes[0][0] == 4):
                        return TrafficLight.UNKNOWN
                else:
                    return TrafficLight.UNKNOWN
 
        return TrafficLight.UNKNOWN
